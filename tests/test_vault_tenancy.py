"""Vault and scope fail-closed behaviour under shared storage.

The persisted connector vault holds saved credentials. On an org deployment
COWORK_HOME is a filesystem every organization can read, so a vault path that
is not org-keyed puts every tenant's credentials in one directory.
"""
from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import MissingTenantScopeError, TenantScope, scoped_storage_root

ORG_A = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def org_deployment(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
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
        scoped_storage_root(org_deployment / "data-vault", None)


def test_no_scope_is_still_the_bare_path_on_a_desktop_install(local_deployment):
    """Desktop passes None everywhere and must be completely unaffected."""
    assert scoped_storage_root(local_deployment / "data-vault", None) == local_deployment / "data-vault"


def test_org_scope_keys_the_vault_per_org(org_deployment):
    scope = TenantScope(org_mode=True, org_id=ORG_A, user_id="u1")
    assert scoped_storage_root(org_deployment / "data-vault", scope) == org_deployment / ORG_A / "data-vault"


def test_two_orgs_never_share_a_vault_directory(org_deployment):
    other = "22222222-2222-4222-8222-222222222222"
    a = scoped_storage_root(org_deployment / "data-vault", TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))
    b = scoped_storage_root(org_deployment / "data-vault", TenantScope(org_mode=True, org_id=other, user_id="u"))
    assert a != b
