"""Contract tests for the rc publish in publish-staging.yml.

The selected rc is recorded in uv.lock. PyPI lists an upload as soon as it
accepts it. uv resolves from the Simple index, which can trail by minutes. The
publish has to wait for the index before anything resolves the pinned rc, or
the wheel check fails on every platform for a wheel that is not missing.
"""

import os
import subprocess
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_STAGING = ROOT / ".github/workflows/publish-staging.yml"
CHECK_WHEELS = ROOT / "scripts/check_wheels_available.sh"
WHEELS_AVAILABLE = ROOT / ".github/workflows/wheels-available.yml"

PIN_STEP = "Select anton-agent from the reviewed lock"
WAIT_STEP = "Wait until uv resolves the pinned anton-agent rc"
RC = "2.26.10.19.1rc2"

# What uv prints when the Simple index does not serve the pinned rc yet.
UV_NO_VERSION = f"""\
echo "error: No solution found when resolving dependencies" >&2
echo "  cause: Because there is no version of anton-agent=={RC}" >&2
exit 1
"""


def _publish_steps(workflow_path=PUBLISH_STAGING) -> list[dict]:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    job = "publish" if workflow_path == PUBLISH_STAGING else "wheels-available"
    return workflow["jobs"][job]["steps"]


def _step(name: str, workflow_path=PUBLISH_STAGING) -> dict:
    return next(step for step in _publish_steps(workflow_path) if step.get("name") == name)


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _run(
    tmp_path: Path, argv: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess:
    # The stubs come first on PATH, so the scripts reach them instead of the
    # real uv and sleep.
    return subprocess.run(
        argv,
        cwd=tmp_path,
        env={**os.environ, **env, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_wait_step(
    tmp_path: Path, *, uv_failures: int, workflow_path=PUBLISH_STAGING
) -> tuple[subprocess.CompletedProcess, list[str], list[str]]:
    """Run the wait step as Actions runs it, with uv failing `uv_failures` times."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "log"
    log.mkdir()
    _stub(
        bin_dir,
        "uv",
        f"""\
printf '%s <<< %s\\n' "$*" "$(cat)" >> "{log}/uv"
if [ "$(( $(wc -l < "{log}/uv") ))" -le {uv_failures} ]; then
{UV_NO_VERSION}fi
echo "anton-agent=={RC}"
""",
    )
    _stub(bin_dir, "sleep", f'echo "$1" >> "{log}/sleep"\n')

    script = tmp_path / "step.sh"
    script.write_text(_step(WAIT_STEP, workflow_path)["run"], encoding="utf-8")
    # `bash -e {0}` is the shell Actions uses for a `run:` with no `shell:`.
    result = _run(tmp_path, ["bash", "-e", str(script)], {"ANTON_RC": RC})

    def lines(name: str) -> list[str]:
        path = log / name
        return path.read_text().splitlines() if path.exists() else []

    return result, lines("uv"), lines("sleep")


@pytest.mark.parametrize("workflow_path", [PUBLISH_STAGING, WHEELS_AVAILABLE])
@pytest.mark.parametrize("version", [RC, "2.26.10.21.1"])
def test_release_selects_reviewed_anton_without_reading_latest(tmp_path, version, workflow_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "curl", "echo 'must not select a new release from PyPI' >&2; exit 99\n")
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname = "anton-agent"\nversion = "{version}"\n'
    )
    output = tmp_path / "output"
    output.touch()
    pin = next(step for step in _publish_steps(workflow_path) if step.get("id") == "pin")
    result = _run(tmp_path, ["bash", "-e", "-c", pin["run"]], {"GITHUB_OUTPUT": str(output)})
    assert result.returncode == 0, result.stderr
    assert f"anton-agent=={version}" in result.stdout
    assert output.read_text() == (f"rc={version}\n" if "rc" in version else "")


def test_the_rc_wait_runs_between_the_pin_and_the_build():
    names = [step.get("name") for step in _publish_steps()]
    pin = _step(PIN_STEP)
    wait = _step(WAIT_STEP)

    # Every step after the build resolves the pin through uv, so the wait has
    # to finish before the build starts.
    assert names.index(PIN_STEP) < names.index(WAIT_STEP) < names.index("Build package")

    # The pin step exports the rc only on the branch that writes the pin, so a
    # stable anton-agent leaves the output empty and skips the wait.
    assert pin["id"] == "pin"
    assert wait["if"] == "steps.pin.outputs.rc != ''"
    assert wait["env"] == {"ANTON_RC": "${{ steps.pin.outputs.rc }}"}


@pytest.mark.parametrize("workflow_path", [PUBLISH_STAGING, WHEELS_AVAILABLE])
def test_the_rc_wait_retries_until_uv_resolves_the_pin(tmp_path, workflow_path):
    result, uv_calls, sleeps = _run_wait_step(tmp_path, uv_failures=2, workflow_path=workflow_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(uv_calls) == 3
    for call in uv_calls:
        args, stdin = call.split(" <<< ")
        assert args.startswith("pip compile - ")
        # Without the refresh, uv can answer from the cache setup-uv restored,
        # which may predate the upload no matter how long the step waits.
        assert "--refresh-package anton-agent" in args
        assert stdin == f"anton-agent=={RC}"
    assert sleeps == ["30", "30"]
    assert f"uv resolves anton-agent=={RC}" in result.stdout


@pytest.mark.parametrize("workflow_path", [PUBLISH_STAGING, WHEELS_AVAILABLE])
def test_the_rc_wait_gives_up_with_uvs_error(tmp_path, workflow_path):
    result, uv_calls, sleeps = _run_wait_step(tmp_path, uv_failures=1000, workflow_path=workflow_path)

    assert result.returncode == 1
    assert len(uv_calls) == 30
    # No sleep after the last attempt.
    assert sleeps == ["30"] * 29
    assert f"there is no version of anton-agent=={RC}" in result.stderr
    assert "still does not serve" in result.stderr


def _run_wheel_check(tmp_path: Path, uv_body: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "uv", uv_body)
    target = tmp_path / "pyproject.toml"
    target.write_text("", encoding="utf-8")
    return _run(tmp_path, ["bash", str(CHECK_WHEELS), str(target)], {})


def test_the_wheel_check_does_not_blame_a_wheel_when_nothing_resolved(tmp_path):
    result = _run_wheel_check(tmp_path, UV_NO_VERSION)

    assert result.returncode == 1
    assert "FAILED (could not resolve at all)" in result.stdout
    assert "could not be resolved at all, so no wheel was checked" in result.stdout
    assert "publishes no wheel" not in result.stdout


def test_the_wheel_check_gives_the_wheel_advice_when_a_wheel_is_missing(tmp_path):
    # Step 1 resolves; step 2, the wheel-only re-resolve, fails.
    result = _run_wheel_check(
        tmp_path,
        """\
case " $* " in
  *" --only-binary "*)
    echo "error: cryptography==49.0.0 has no wheel for this platform" >&2
    exit 1
    ;;
esac
""",
    )

    assert result.returncode == 1
    assert "FAILED (resolves to a version with no wheel)" in result.stdout
    assert "publishes no wheel" in result.stdout
    assert "could not be resolved at all" not in result.stdout


@pytest.mark.parametrize("filename", ["publish.yml", "publish-staging.yml"])
def test_both_publishers_check_the_built_artifact_before_upload(tmp_path, filename):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    steps = workflow["jobs"]["publish"]["steps"]
    step = next(item for item in steps if item.get("name") == "Check the built wheel resolves to wheels on every shipped platform")
    assert steps.index(step) < next(i for i, item in enumerate(steps) if item.get("uses", "").startswith("pypa/gh-action-pypi-publish"))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check_wheels_available.sh").write_text('test "$1" = dist/reviewed.whl\n')
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "reviewed.whl").touch()
    result = _run(tmp_path, ["bash", "-c", step["run"]], {})
    assert result.returncode == 0, result.stdout + result.stderr
    (dist / "second.whl").touch()
    result = _run(tmp_path, ["bash", "-c", step["run"]], {})
    assert result.returncode != 0


def test_selected_rc_wait_precedes_the_first_constrained_wheel_resolution():
    workflow = yaml.safe_load(PUBLISH_STAGING.read_text())
    assert "wheels-available" in workflow["jobs"]["release"]["needs"]
    assert "wheels-available" in workflow["jobs"]["publish"]["needs"]
    gate = yaml.safe_load(WHEELS_AVAILABLE.read_text())["jobs"]["wheels-available"]
    steps = gate["steps"]
    names = [step.get("name") for step in steps]
    check = "Check every shipped platform resolves to versions with wheels"
    assert names.index(PIN_STEP) < names.index(WAIT_STEP) < names.index(check)
    pin = next(step for step in steps if step.get("name") == PIN_STEP)
    wait = next(step for step in steps if step.get("name") == WAIT_STEP)
    assert pin["if"] == "!inputs.floating"
    assert wait["if"] == "steps.pin.outputs.rc != ''"
