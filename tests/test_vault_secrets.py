from __future__ import annotations

import asyncio
import logging

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import _reset_registered_ds_vars, begin_ds_turn_scope, scrub_credentials

import cowork.services.connectors.vault_secrets as vault_secrets_mod
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, TenantScope
from cowork.services.connectors.vault_secrets import register_vault_secrets


async def test_an_unreadable_vault_falls_back_to_shape_scrubbing(monkeypatch, caplog):
    """Registration runs on every turn, so a vault it cannot read must not
    fail the turn; the history is still scrubbed by API-key shape."""
    def missing_scope(scope):
        raise MissingTenantScopeError("no org in scope")

    monkeypatch.setattr(vault_secrets_mod, "vault_for_scope", missing_scope)
    # The migration tests' fileConfig disables every logger created before it.
    monkeypatch.setattr(vault_secrets_mod.logger, "disabled", False)

    with caplog.at_level(logging.WARNING, logger=vault_secrets_mod.__name__):
        await register_vault_secrets(None)

    assert "Could not register vault secrets" in caplog.text
    leaked_key = "sk-" + "a" * 30
    assert scrub_credentials(f"my key is {leaked_key}") == "my key is [REDACTED_API_KEY]"


async def test_concurrent_orgs_each_redact_only_their_own_secret(monkeypatch, tmp_path, request):
    """Two orgs' turns overlap on one server, under a parent context that
    already holds a DS_* scope (a long-lived task reused across turns). Both
    orgs name the connection the same, so only per-org vaults and per-request
    scopes keep each turn on its own password."""
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    get_app_settings.cache_clear()
    request.addfinalizer(get_app_settings.cache_clear)
    passwords = {"org-a": "alphaSecret1", "org-b": "bravoSecret2"}
    for org_id, password in passwords.items():
        LocalDataVault(tmp_path / org_id / "data-vault").save("postgres", "mydb", {
            "host": "db.example.com", "port": "5432", "database": "app",
            "user": "svc", "password": password,
        })
    request.addfinalizer(_reset_registered_ds_vars)
    begin_ds_turn_scope()
    both_registered = asyncio.Barrier(2)

    async def turn(org_id: str) -> str:
        await register_vault_secrets(TenantScope(org_mode=True, org_id=org_id))
        await both_registered.wait()
        return scrub_credentials(" ".join(passwords.values()))

    seen_by_a, seen_by_b = await asyncio.gather(turn("org-a"), turn("org-b"))

    assert "alphaSecret1" not in seen_by_a and "bravoSecret2" in seen_by_a
    assert "bravoSecret2" not in seen_by_b and "alphaSecret1" in seen_by_b


async def test_an_empty_vault_skips_the_registry_rebuild(monkeypatch, tmp_path):
    """An org with no connections has nothing to register, so its turns must
    not pay for anton re-parsing the datasource registry."""
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    rebuilds = []
    monkeypatch.setattr(vault_secrets_mod, "restore_namespaced_env", rebuilds.append)

    await register_vault_secrets(LOCAL_SCOPE)

    assert rebuilds == []


async def test_a_busy_file_store_refuses_the_registration_instead_of_scrubbing_by_shape(
    monkeypatch, tmp_path, request
):
    """A saved DSN password has no API-key shape: only its registered value
    redacts it. With every file store slot of the caller's share held, the
    registration refuses with FilesystemBusy rather than let the turn go on
    scrubbing by shape alone."""
    import cowork.db.session as db_session
    from cowork.db import units
    from cowork.db.units import FilesystemBusy
    from test_run_efs import holding

    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    vault = LocalDataVault(tmp_path / "data-vault")
    vault.save("postgres", "mydb", {
        "host": "db.example.com", "port": "5432", "database": "app",
        "user": "svc", "password": "hunter2DsnPass",
    })
    monkeypatch.setattr(vault_secrets_mod, "vault_for_scope", lambda scope: vault)
    request.addfinalizer(_reset_registered_ds_vars)
    alice = TenantScope(org_mode=True, org_id="org-1", user_id="alice")

    async with holding(units.EFS_SHARE, alice):
        with pytest.raises(FilesystemBusy):
            await register_vault_secrets(alice)

    await register_vault_secrets(alice)
    assert "hunter2DsnPass" not in scrub_credentials("connect with hunter2DsnPass")


async def test_a_web_turn_answers_503_when_the_file_store_is_too_busy_to_register_its_secrets(monkeypatch):
    """The refusal reaches the client as the busy answer it can retry: 503
    server_busy with Retry-After, before the routing gate reads any history."""
    import httpx

    import cowork.db.session as db_session
    import cowork.handlers.responses as responses_mod
    from cowork.db import units
    from cowork.server import create_app
    from test_run_efs import holding

    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    routed = []

    async def route_request(self, **kwargs):
        routed.append(kwargs)
        raise RuntimeError("the gate ran")

    monkeypatch.setattr(responses_mod.ResponsesHandler, "_route_request", route_request)
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
            response = await client.post("/api/v1/responses/", json={"input": "hi", "stream": True})

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "server_busy"
    assert response.headers["retry-after"] == "1"
    assert routed == []
