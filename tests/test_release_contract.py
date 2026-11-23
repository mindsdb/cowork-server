"""Reject drift between the reviewed lock and the artifacts users install."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_contract_under_test", ROOT / "scripts/release_contract.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
CONTRACT = "fixture-dependency==1.0\n"


def distributions(directory, *, wheel="valid", sdist="valid"):
    member = release.CONTRACT.relative_to(release.ROOT).as_posix()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(directory / "fixture-1.0-py3-none-any.whl", "w") as archive:
            if wheel != "missing":
                archive.writestr(member, CONTRACT if wheel != "altered" else "fixture-dependency==2.0\n")
            if wheel == "duplicate":
                archive.writestr(member, CONTRACT)
    with tarfile.open(directory / "fixture-1.0.tar.gz", "w:gz") as archive:
        for _ in range(2 if sdist == "duplicate" else 1):
            if sdist == "missing":
                continue
            info = tarfile.TarInfo("fixture-1.0/" + member)
            if sdist == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "outside-contract.txt"
                archive.addfile(info)
            else:
                data = (CONTRACT if sdist != "altered" else "fixture-dependency==2.0\n").encode()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def test_identical_contract_is_accepted_in_both_artifacts(tmp_path):
    distributions(tmp_path)
    release.check_distributions(tmp_path, CONTRACT)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "altered"])
def test_wheel_contract_must_be_present_once_and_unchanged(tmp_path, kind):
    distributions(tmp_path, wheel=kind)
    with pytest.raises(ValueError, match="Wheel runtime contract"):
        release.check_distributions(tmp_path, CONTRACT)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "altered", "symlink"])
def test_sdist_contract_must_be_one_unchanged_regular_file(tmp_path, kind):
    distributions(tmp_path, sdist=kind)
    with pytest.raises(ValueError, match="Sdist"):
        release.check_distributions(tmp_path, CONTRACT)


def test_check_rejects_stale_export_without_rewriting_it(tmp_path, monkeypatch, capsys):
    contract = tmp_path / "requirements.txt"
    original = "fixture-dependency==0.9\n"
    contract.write_text(original)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=CONTRACT))
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check"])

    assert release.main() == 1
    assert "stale" in capsys.readouterr().err
    assert contract.read_text() == original


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires the CI-pinned uv executable")
def test_export_refuses_changed_project_metadata_without_rewriting_lock(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "contract-fixture"\nversion = "1.0"\nrequires-python = ">=3.11"\ndependencies = []\n')
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    subprocess.run(["uv", "lock", "--offline"], cwd=tmp_path, check=True, capture_output=True)
    lock = tmp_path / "uv.lock"
    original = lock.read_bytes()
    # This can resolve offline without changing any package version. An unlocked
    # export would silently repair the lock instead of rejecting this mismatch.
    project.write_text(project.read_text().replace(">=3.11", ">=3.12"))
    contract = tmp_path / "requirements.txt"
    contract.write_text("")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check"])

    assert release.main() != 0
    assert lock.read_bytes() == original
    assert contract.read_text() == ""


def test_final_pins_pass_the_production_check():
    export = (
        "anton-agent==2.26.11.15.1 \\\n"
        "    --hash=sha256:abc\n"
        "pywin32==311 ; sys_platform == 'win32' \\\n"
        "    --hash=sha256:def\n"
        "certifi==2026.10.5.post1\n"
        "torch==2.9.0+cpu\n"
    )
    assert release.unreleased_pins(export) == []


@pytest.mark.parametrize(
    "version", ["2.26.11.13.1rc3", "2.26.10.29.1rc4", "1.0a1", "1.0b2", "1.0.dev3", "1.0rc1.post1", "not-a-version"]
)
def test_production_check_refuses_pre_release_dev_and_unparseable_pins(version):
    export = f"aiohttp==3.14.0 \\\n    --hash=sha256:abc\nanton-agent=={version} \\\n    --hash=sha256:def\n"
    assert release.unreleased_pins(export) == [f"anton-agent=={version}"]


@pytest.mark.parametrize("version, refused", [
    ("1!2.0+linux.1", False),
    ("1!2.0rc1+linux.1", True),
    ("2.0.post1.dev2", True),
])
def test_production_check_handles_epoch_and_local_versions(version, refused):
    pin = f"dependency=={version}"
    assert release.unreleased_pins(pin + ' ; python_version >= "3.12"') == ([pin] if refused else [])


def test_require_final_rejects_current_rc_export_without_rewriting(tmp_path, monkeypatch):
    contract = tmp_path / "requirements.txt"
    export = "anton-agent==2.26.11.13.1rc3\n"
    contract.write_text(export)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=export))
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check", "--require-final"])
    assert release.main() == 1
    assert contract.read_text() == export


def project(dependencies, extras=None):
    text = f'[project]\nname = "fixture"\ndependencies = {json.dumps(dependencies)}\n'
    if extras:
        text += "[project.optional-dependencies]\n"
        text += "".join(f"{name} = {json.dumps(values)}\n" for name, values in extras.items())
    return text


def test_final_bounds_pass_the_production_check():
    pyproject = project([
        "anton-agent>=2.26.11.15.1,<3",
        "pywin32>=311; sys_platform == 'win32'",
        'tomli>=2.0; python_full_version < "3.13.0rc1"',
        "certifi>=2026.10.5.post1",
        "uvicorn[standard]~=0.30.0",
        "fixture-series==1.*",
        "fixture-tool @ https://example.com/fixture-tool-1.0rc1.tar.gz",
        "unbounded",
    ], {"channels": ["slack-sdk>=3.27.0"]})
    assert release.unreleased_requirements(pyproject) == []


@pytest.mark.parametrize("requirement", [
    "anton-agent>=2.26.11.20.1rc3,<3",
    "fixture>=1.0a1",
    "fixture[extra]>=1.0b2",
    "fixture==1.0.dev3",
    "fixture>=1.0,<2.0rc1",
    "fixture>=not-a-version",
])
def test_production_check_refuses_pre_release_dev_and_unparseable_bounds(requirement):
    pyproject = project(["aiohttp>=3.14", requirement])
    assert release.unreleased_requirements(pyproject) == [requirement]


def test_production_check_reads_optional_dependencies():
    pyproject = project([], {"channels": ["slack-sdk>=3.27.0rc1"]})
    assert release.unreleased_requirements(pyproject) == ["slack-sdk>=3.27.0rc1"]


def run_check(tmp_path, monkeypatch, export, dependencies, *flags, allowance=""):
    contract = tmp_path / "requirements.txt"
    contract.write_text(export)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(project(dependencies) + allowance)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(release, "PYPROJECT", pyproject)
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=export))
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check", *flags])
    return release.main()


def test_require_final_refuses_a_candidate_floor_under_a_final_lock(tmp_path, monkeypatch, capsys):
    floor = "anton-agent>=2.26.11.20.1rc3,<3"
    assert run_check(tmp_path, monkeypatch, "anton-agent==2.26.11.20.1\n", [floor], "--require-final") == 1
    err = capsys.readouterr().err
    assert f"pyproject.toml requires:\n  {floor}\n" in err
    assert "The lock pins" not in err


def test_require_final_names_both_a_candidate_lock_and_floor(tmp_path, monkeypatch, capsys):
    export = "anton-agent==2.26.11.20.1rc3\n"
    assert run_check(tmp_path, monkeypatch, export, ["anton-agent>=2.26.11.20.1rc3,<3"], "--require-final") == 1
    err = capsys.readouterr().err
    assert "The lock pins:\n  anton-agent==2.26.11.20.1rc3\n" in err
    assert "pyproject.toml requires:\n  anton-agent>=2.26.11.20.1rc3,<3\n" in err


def test_require_final_accepts_a_final_lock_and_floor(tmp_path, monkeypatch):
    export = "anton-agent==2.26.11.20.1\n"
    assert run_check(tmp_path, monkeypatch, export, ["anton-agent>=2.26.11.20.1,<3"], "--require-final") == 0


def test_staging_check_accepts_a_candidate_lock_and_floor(tmp_path, monkeypatch):
    export = "anton-agent==2.26.11.20.1rc3\n"
    assert run_check(tmp_path, monkeypatch, export, ["anton-agent>=2.26.11.20.1rc3,<3"]) == 0


ALLOW_RC3 = '[tool.cowork.release.allow-prerelease]\nanton-agent = "2.26.11.20.1rc3"\n'


def test_an_allowed_pre_release_passes_in_the_lock_and_the_floor(tmp_path, monkeypatch, capsys):
    export = "anton-agent==2.26.11.20.1rc3\n"
    floor = ["anton-agent>=2.26.11.20.1rc3,<3"]
    assert run_check(tmp_path, monkeypatch, export, floor, "--require-final", allowance=ALLOW_RC3) == 0
    assert "allows these pre-releases in a production release:\n  anton-agent==2.26.11.20.1rc3\n" in capsys.readouterr().err


def test_an_allowance_names_one_exact_version(tmp_path, monkeypatch, capsys):
    export = "anton-agent==2.26.11.20.1rc4\n"
    floor = ["anton-agent>=2.26.11.20.1rc4,<3"]
    assert run_check(tmp_path, monkeypatch, export, floor, "--require-final", allowance=ALLOW_RC3) == 1
    err = capsys.readouterr().err
    assert "The lock pins:\n  anton-agent==2.26.11.20.1rc4\n" in err
    assert "pyproject.toml requires:\n  anton-agent>=2.26.11.20.1rc4,<3\n" in err


def test_an_allowance_covers_only_its_own_project(tmp_path, monkeypatch, capsys):
    export = "anton-agent==2.26.11.20.1rc3\nfixture==2.26.11.20.1rc3\n"
    dependencies = ["anton-agent>=2.26.11.20.1rc3,<3", "fixture>=2.26.11.20.1rc3"]
    assert run_check(tmp_path, monkeypatch, export, dependencies, "--require-final", allowance=ALLOW_RC3) == 1
    err = capsys.readouterr().err
    assert "The lock pins:\n  fixture==2.26.11.20.1rc3\n" in err
    assert "pyproject.toml requires:\n  fixture>=2.26.11.20.1rc3\n" in err


def test_allowance_names_compare_as_pep503_names():
    text = project(["Anton_Agent>=1.0rc1"]) + '[tool.cowork.release.allow-prerelease]\n"anton.agent" = "1.0rc1"\n'
    assert release.allowed_prereleases(text) == {"anton-agent": "1.0rc1"}
    assert release.unreleased_requirements(text) == []
    assert release.unreleased_pins("anton-agent==1.0rc1\n", release.allowed_prereleases(text)) == []


def test_an_allowance_must_name_one_version():
    text = project([]) + '[tool.cowork.release.allow-prerelease]\nanton-agent = ["1.0rc1"]\n'
    with pytest.raises(ValueError, match="must name one version"):
        release.allowed_prereleases(text)


def test_no_allowance_table_allows_nothing():
    assert release.allowed_prereleases(project(["anton-agent>=1.0rc1"])) == {}
