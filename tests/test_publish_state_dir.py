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
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, TenantScope

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "55e92829-e701-599b-88f9-69f5bf1165eb"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
SUB_B = "8a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _person(*, partition: str, sub: str, resources=frozenset(PersonalResource)) -> TenantScope:
    return TenantScope(
        personal=True, org_id=partition, user_id=sub, personal_resources=frozenset(resources)
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_app_settings.cache_clear()


def test_state_dir_stays_under_cowork_home_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    assert publish_mod._cowork_state_dir(scope=LOCAL_SCOPE) == tmp_path


def test_state_dir_is_not_under_shared_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    org_scope = TenantScope(org_mode=True, org_id=ORG, user_id=SUB_A)
    resolved = publish_mod._cowork_state_dir(scope=org_scope).resolve()

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

    org_scope = TenantScope(org_mode=True, org_id=ORG, user_id=SUB_A)
    assert publish_mod._cowork_state_dir(scope=org_scope) == explicit


# ── personal partitions (local tenancy with trusted identity headers) ──────


def test_each_person_keeps_their_own_publish_history(personal_deployment, monkeypatch, tmp_path):
    """Publish history holds each published artifact's path, name and public
    URL, and GET /api/v1/publish returns it, so a person whose projects are
    personal reads and writes only their own."""
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    bob = _person(partition=PARTITION_B, sub=SUB_B)

    assert publish_mod._cowork_state_dir(scope=alice) == tmp_path / "shared" / PARTITION_A / "publish"
    publish_mod._save_state({"publish_history": [{"url": "https://view/alice"}]}, scope=alice)

    assert publish_mod._load_state(scope=bob) == {}
    assert publish_mod._load_state(scope=alice)["publish_history"] == [{"url": "https://view/alice"}]
    # The operator view keeps the deployment's own history, not anyone's.
    assert publish_mod._cowork_state_dir(scope=LOCAL_SCOPE) == tmp_path
    assert publish_mod._load_state(scope=LOCAL_SCOPE) == {}


def test_publish_history_is_shared_where_projects_are(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()
    shared = _person(partition=PARTITION_A, sub=SUB_A, resources=frozenset())

    assert publish_mod._cowork_state_dir(scope=shared) == tmp_path


def test_publish_history_refuses_an_unbound_caller_on_a_partitioned_deployment(
    personal_deployment, monkeypatch, tmp_path
):
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()

    with pytest.raises(MissingTenantScopeError):
        publish_mod._cowork_state_dir(scope=None)
