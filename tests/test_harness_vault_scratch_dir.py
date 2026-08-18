"""Anton harness's temporary filtered data-vault directory must not land on
shared EFS in org mode.

_build_chat_session stages a per-turn filtered copy of the connector vault
under _vault_scratch_dir() (cleaned up via shutil.rmtree afterwards) whenever
a turn disables one or more connections. That directory carries no org_id
segment, so left on cowork_home() it would put every organization's
temporary vault contents under the same shared, readable location.
"""
import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.anton_harness.harness import _vault_scratch_dir


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_app_settings.cache_clear()


def test_vault_scratch_dir_stays_under_cowork_home_tmp_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    assert _vault_scratch_dir() == tmp_path / "tmp"


def test_vault_scratch_dir_is_not_under_shared_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = _vault_scratch_dir()

    assert home not in resolved.parents
    assert resolved != home / "tmp"
