"""Connector probe's plaintext credential file must not land on shared EFS.

_write_credentials_env() writes connector credentials in plaintext (DS_*
assignments) to a UUID-named file under _probe_tmp_dir(). Before this fix
that directory was a module-level constant, cowork_home() / "tmp". In org
mode cowork_home() is the shared EFS tree with no org_id segment under this
path, so the plaintext file was readable by any organization sharing the
mount and, unlike the old ephemeral container disk, survived a pod restart.
"""
import os
import stat
from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.connectors.probe import CredentialProbe, _probe_tmp_dir


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_app_settings.cache_clear()


def _probe(credentials):
    return CredentialProbe(
        engine="postgres",
        credentials=credentials,
        llm_client=None,
        workspace=None,
    )


def test_credentials_file_stays_under_cowork_home_tmp_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    path, var_names = _probe({"password": "hunter2"})._write_credentials_env()

    assert Path(path).parent == tmp_path / "tmp"
    assert var_names == ["DS_PASSWORD"]


def test_credentials_file_is_not_written_under_shared_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    path, _ = _probe({"password": "hunter2"})._write_credentials_env()

    resolved = Path(path).resolve()
    assert home.resolve() not in resolved.parents


def test_probe_tmp_dir_org_mode_is_not_under_cowork_home(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = _probe_tmp_dir()

    assert home not in resolved.parents


def test_credentials_file_is_written_with_owner_only_permissions(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    path, _ = _probe({"password": "hunter2"})._write_credentials_env()

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600
