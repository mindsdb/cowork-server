"""tests/skip_guard.py, driven through real pytest runs.

Each case writes a small test file to a temporary directory and runs pytest on
it in a subprocess with the guard loaded, so the hooks, the exit code and the
job summary are the ones CI sees.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.skip_guard import AllowedSkip, is_allowed, load_allowlist

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tests" / "skip_allowlist.yaml"

SKIPPING_FILE = """
import pytest

def test_runs():
    assert True

def test_skips():
    pytest.skip("no Redis reachable")
"""

DESELECT_FILE = """
def test_runs():
    assert True

def test_left_out():
    assert True
"""


def _run(
    tmp_path: Path,
    source: str,
    *,
    env: dict[str, str],
    allowed: list[str] = (),
    extra_args: list[str] = (),
) -> subprocess.CompletedProcess:
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text(
        yaml.safe_dump([{"test": test, "reason": "listed for this case"} for test in allowed]),
        encoding="utf-8",
    )
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COWORK_REQUIRE_INTEGRATION", "COWORK_REQUIRE_SERVICES", "GITHUB_STEP_SUMMARY"}
    }
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "tests.skip_guard", "-p", "no:cacheprovider",
            "--rootdir", str(tmp_path), "--skip-allowlist", str(allowlist),
            *extra_args, str(tmp_path / "test_sample.py"),
        ],
        cwd=tmp_path,
        env={**clean, "PYTHONPATH": str(ROOT), **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("flag", ["COWORK_REQUIRE_INTEGRATION", "COWORK_REQUIRE_SERVICES"])
def test_a_required_run_fails_on_a_skip_the_allowlist_does_not_name(tmp_path, flag):
    result = _run(tmp_path, SKIPPING_FILE, env={flag: "true"})

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout
    assert "NOT ALLOWED: test_sample.py::test_skips: no Redis reachable" in result.stdout


def test_a_required_run_passes_when_the_allowlist_names_the_skip(tmp_path):
    result = _run(
        tmp_path, SKIPPING_FILE,
        env={"COWORK_REQUIRE_SERVICES": "true"}, allowed=["test_sample.py::test_skips"],
    )

    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert "allowed: test_sample.py::test_skips: no Redis reachable" in result.stdout


def test_a_best_effort_run_lists_the_skip_and_still_passes(tmp_path):
    result = _run(tmp_path, SKIPPING_FILE, env={"COWORK_REQUIRE_INTEGRATION": "false"})

    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert "tests left out of this run" in result.stdout
    assert "NOT ALLOWED: test_sample.py::test_skips" in result.stdout


def test_a_run_with_no_flag_says_nothing(tmp_path):
    result = _run(tmp_path, SKIPPING_FILE, env={})

    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert "tests left out of this run" not in result.stdout


def test_a_required_run_fails_on_an_explicit_deselect_the_allowlist_does_not_name(tmp_path):
    deselect = ["--deselect", "test_sample.py::test_left_out"]

    refused = _run(tmp_path, DESELECT_FILE, env={"COWORK_REQUIRE_INTEGRATION": "true"}, extra_args=deselect)
    allowed = _run(
        tmp_path, DESELECT_FILE, env={"COWORK_REQUIRE_INTEGRATION": "true"},
        allowed=["test_sample.py::test_left_out"], extra_args=deselect,
    )

    assert refused.returncode == pytest.ExitCode.TESTS_FAILED, refused.stdout
    assert "NOT ALLOWED: test_sample.py::test_left_out: deselected with --deselect" in refused.stdout
    assert allowed.returncode == pytest.ExitCode.OK, allowed.stdout


def test_a_marker_deselect_is_not_a_skip(tmp_path):
    source = DESELECT_FILE.replace("def test_left_out", "import pytest\n\n@pytest.mark.slow\ndef test_left_out")

    result = _run(
        tmp_path, source, env={"COWORK_REQUIRE_INTEGRATION": "true"}, extra_args=["-m", "not slow"],
    )

    assert result.returncode == pytest.ExitCode.OK, result.stdout


def test_a_run_that_already_failed_keeps_its_own_exit_code(tmp_path):
    source = SKIPPING_FILE.replace("assert True", "assert False")

    result = _run(tmp_path, source, env={"COWORK_REQUIRE_SERVICES": "true"})

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout
    assert "1 failed" in result.stdout


def test_the_job_summary_names_each_left_out_test_and_its_reason(tmp_path):
    summary = tmp_path / "summary.md"

    _run(tmp_path, SKIPPING_FILE, env={"COWORK_REQUIRE_SERVICES": "true", "GITHUB_STEP_SUMMARY": str(summary)})

    text = summary.read_text(encoding="utf-8")
    assert "| `test_sample.py::test_skips` | NOT RUN (not allowed) | no Redis reachable |" in text
    assert "1 not allowed, so the run fails." in text


def test_an_entry_covers_every_parametrized_case_of_its_test():
    allowlist = [AllowedSkip(test="tests/x.py::test_a", reason="r")]

    assert is_allowed("tests/x.py::test_a[redis]", allowlist)
    assert is_allowed("tests/x.py::test_a", allowlist)
    assert not is_allowed("tests/x.py::test_ab", allowlist)


def test_every_allowlist_entry_names_a_test_that_exists():
    for entry in load_allowlist(ALLOWLIST):
        path, _, name = entry.test.partition("::")
        source = (ROOT / path).read_text(encoding="utf-8")
        assert f"def {name.split('[')[0]}(" in source, entry.test
        assert entry.reason.strip(), entry.test


def test_two_replicas_fails_without_redis_where_the_services_job_promised_one(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         "tests/integration/test_two_replicas.py"],
        cwd=ROOT,
        env={
            **os.environ,
            # Nothing listens on port 9 (discard), so the ping fails at once.
            "COWORK_TEST_REDIS_URL": "redis://127.0.0.1:9/1",
            "COWORK_REQUIRE_SERVICES": "true",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout
    assert "COWORK_REQUIRE_SERVICES is set" in result.stdout
