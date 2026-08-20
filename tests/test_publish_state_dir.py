"""publish's state.json must not land on shared EFS in org mode.

_cowork_state_dir() falls back to bare cowork_home() unless
ANTON_COWORK_STATE_DIR is set, and no values.yaml sets that var. state.json
holds publish_history (_save_state / list_publish_history), which carries no
org_id segment, so every organization would read every other organization's
publish history off the shared tree.
"""
import pytest

import cowork.services.publish as publish_mod
from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_app_settings.cache_clear()


def test_state_dir_stays_under_cowork_home_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    assert publish_mod._cowork_state_dir() == tmp_path


def test_state_dir_is_not_under_shared_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = publish_mod._cowork_state_dir().resolve()

    assert home.resolve() not in resolved.parents
    assert resolved != home.resolve()


def test_explicit_state_dir_env_still_wins_in_org_mode(monkeypatch, tmp_path):
    # An operator-set ANTON_COWORK_STATE_DIR is an explicit override and must
    # keep taking precedence, in org mode as in local mode.
    explicit = tmp_path / "explicit-state"
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "cowork-shared"))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(explicit))
    get_app_settings.cache_clear()

    assert publish_mod._cowork_state_dir() == explicit
