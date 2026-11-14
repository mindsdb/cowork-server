"""The cached app settings follow the environment from one test to the next.

A test that switches tenancy through the environment must not leave its
settings cached for the next test on the same process, whatever order its
fixtures tear down in. Under pytest-xdist the next test can be any test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_TWO_TESTS = """
import pytest

from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture
def reads_settings_at_teardown():
    yield
    get_app_settings()


@pytest.fixture
def org_tenancy(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_switches_to_org(monkeypatch, reads_settings_at_teardown, org_tenancy):
    assert get_app_settings().tenancy_mode == "org"


def test_the_next_test_reads_local_tenancy():
    assert get_app_settings().tenancy_mode == "local"
"""


def test_a_settings_read_during_teardown_does_not_carry_org_tenancy_to_the_next_test(tmp_path):
    """``monkeypatch`` is set up before ``org_tenancy``, so it restores the env
    after ``reads_settings_at_teardown`` has cached org settings again."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_two.py").write_text(_TWO_TESTS)
    env = {key: value for key, value in os.environ.items() if key != "COWORK_TENANCY_MODE"}

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-p", "conftest",
         "--rootdir", str(tmp_path), "tests"],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": f"{_ROOT / 'tests'}{os.pathsep}{_ROOT}"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert "2 passed" in result.stdout, result.stdout
    assert result.returncode == pytest.ExitCode.OK, result.stdout
