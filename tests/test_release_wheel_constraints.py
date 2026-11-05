"""Exercise the release and floating wheel gates against a local wheel index."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
UV = shutil.which("uv")
pytestmark = pytest.mark.skipif(UV is None, reason="requires the CI-pinned uv executable")


def package_wheel(directory, version):
    name = "release_fixture"
    metadata = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(metadata + "/METADATA", f"Metadata-Version: 2.3\nName: release-fixture\nVersion: {version}\n")
        archive.writestr(metadata + "/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(metadata + "/RECORD", "")
    return path


def gate_fixture(tmp_path, script=None):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    gate = scripts / "check_wheels_available.sh"
    gate.write_text(script or (ROOT / "scripts/check_wheels_available.sh").read_text())
    contract = tmp_path / "cowork" / "release-requirements.txt"
    contract.parent.mkdir()
    contract.write_text("release-fixture==1.0\n")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "server-fixture"\nversion = "1.0"\ndependencies = ["release-fixture>=1"]\n')
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    package_wheel(wheels, "1.0")
    package_wheel(wheels, "2.0")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "resolutions.jsonl"
    # Run the real uv against local wheels. Record the first pass's output file
    # before the shell script removes its work directory, so the assertion sees
    # the versions actually selected for every platform and interpreter.
    wrapper = bin_dir / "uv"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, subprocess, sys\n"
        f"result = subprocess.run([{UV!r}, *sys.argv[1:], '--no-index', '--find-links', {str(wheels)!r}])\n"
        "if result.returncode == 0 and '--output-file' in sys.argv:\n"
        "    output = pathlib.Path(sys.argv[sys.argv.index('--output-file') + 1]).read_text()\n"
        f"    with open({str(log)!r}, 'a') as stream:\n"
        "        stream.write(json.dumps(output) + '\\n')\n"
        "raise SystemExit(result.returncode)\n"
    )
    wrapper.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "UV_OVERRIDE": "",
        "UV_CONSTRAINT": "",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    return gate, log, environment


@pytest.mark.parametrize(("floating", "version"), [(False, "1.0"), (True, "2.0")])
def test_release_keeps_reviewed_version_while_floating_probe_can_advance(tmp_path, floating, version):
    gate, log, environment = gate_fixture(tmp_path)
    args = ["bash", str(gate)] + (["--floating"] if floating else [])
    result = subprocess.run(args, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stdout + result.stderr
    resolutions = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(resolutions) == 8  # Four shipped platforms, Python 3.12 and 3.13.
    for output in resolutions:
        selected = [line for line in output.splitlines() if line.startswith("release-fixture==")]
        assert selected == [f"release-fixture=={version}"]


def test_missing_reviewed_version_fails_instead_of_selecting_available_newer_wheel(tmp_path):
    gate, log, environment = gate_fixture(tmp_path)
    (tmp_path / "wheels" / "release_fixture-1.0-py3-none-any.whl").unlink()
    result = subprocess.run(["bash", str(gate)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not resolve at all" in result.stdout
    assert not log.exists()
