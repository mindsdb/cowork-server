"""Vault and scope fail-closed behaviour under shared storage.

The persisted connector vault holds saved credentials. On an org deployment
COWORK_HOME is a filesystem every organization can read, so a vault path that
is not org-keyed puts every tenant's credentials in one directory.
"""
from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
import cowork.server as cowork_server
from cowork.db.scoped import MissingTenantScopeError, TenantScope, scoped_storage_root

ORG_A = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def org_deployment(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture
def local_deployment(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


def test_no_scope_fails_closed_on_an_org_deployment(org_deployment):
    """A caller that forgot to thread its scope must not silently receive the
    shared namespace root, which is readable by every organization."""
    with pytest.raises(MissingTenantScopeError):
        scoped_storage_root(org_deployment / "data-vault", None, store="data-vault")


def test_no_scope_is_still_the_bare_path_on_a_desktop_install(local_deployment):
    """Desktop passes None everywhere and must be completely unaffected."""
    assert scoped_storage_root(local_deployment / "data-vault", None, store="data-vault") == local_deployment / "data-vault"


def test_org_scope_keys_the_vault_per_org(org_deployment):
    scope = TenantScope(org_mode=True, org_id=ORG_A, user_id="u1")
    assert scoped_storage_root(org_deployment / "data-vault", scope, store="data-vault") == org_deployment / ORG_A / "data-vault"


def test_two_orgs_never_share_a_vault_directory(org_deployment):
    other = "22222222-2222-4222-8222-222222222222"
    a = scoped_storage_root(org_deployment / "data-vault", TenantScope(org_mode=True, org_id=ORG_A, user_id="u"), store="data-vault")
    b = scoped_storage_root(org_deployment / "data-vault", TenantScope(org_mode=True, org_id=other, user_id="u"), store="data-vault")
    assert a != b


def test_single_tenant_harness_is_refused_in_org_mode(org_deployment):
    """available_harness_ids() only hides it from the picker. The harness is an
    org-scoped user setting, so a stored row naming one still reached get_harness
    and ran it against unscoped shared-storage paths."""
    from cowork.harnesses.base import get_harness

    with pytest.raises(ValueError, match="does not support multi-tenant"):
        get_harness("hermes")


def test_single_tenant_harness_still_loads_on_desktop(local_deployment):
    from cowork.harnesses.base import _registry, get_harness

    if "hermes" not in _registry:
        pytest.skip("hermes harness not installed in this environment")
    assert get_harness("hermes") is not None


def test_bearer_token_mirror_is_refused_in_org_mode(org_deployment, monkeypatch):
    """The token is mirrored into cowork_home()/.env, which on an org deployment
    is shared storage every organization can read and overwrite."""
    # Imported at module scope on purpose: cowork.server builds an app at
    # import time, so importing it here would raise before pytest.raises saw
    # the call and the test would pass for the wrong reason.
    monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
    get_app_settings.cache_clear()

    with pytest.raises(RuntimeError, match="not supported in org tenancy mode"):
        cowork_server.create_app()


def test_persisted_vault_is_org_keyed(org_deployment, monkeypatch):
    """The saved-credential vault, not the probe's transient copy. Two orgs
    must never resolve to the same directory."""
    monkeypatch.setenv("COWORK_CONNECTOR__VAULT_DIR", str(org_deployment / "data-vault"))
    get_app_settings.cache_clear()
    from cowork.services.connectors.persist import vault_for_scope

    other = "22222222-2222-4222-8222-222222222222"
    a = vault_for_scope(TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))
    b = vault_for_scope(TenantScope(org_mode=True, org_id=other, user_id="u"))

    assert Path(a._dir) != Path(b._dir)
    assert ORG_A in str(a._dir)


def test_persisted_vault_refuses_an_unscoped_save_in_org_mode(org_deployment):
    """An unscoped call used to resolve to the shared namespace root, which is
    where every organization's credentials ended up in one directory."""
    from cowork.services.connectors.persist import vault_for_scope

    with pytest.raises(MissingTenantScopeError):
        vault_for_scope(None)


def test_persisted_vault_is_unchanged_on_desktop(local_deployment, monkeypatch):
    monkeypatch.setenv("COWORK_CONNECTOR__VAULT_DIR", str(local_deployment / "data-vault"))
    get_app_settings.cache_clear()
    from cowork.services.connectors.persist import vault_for_scope

    assert Path(vault_for_scope(None)._dir) == local_deployment / "data-vault"
