"""delete_connection()'s local-mode branch. Became `async def` so the
org-mode branch can await auth_proxy.proxy_delete - regression coverage for
the local branch's blocking oauth_service.revoke() call, which must stay off
the event loop (run_in_threadpool), not run on it now that FastAPI no longer
threadpools this handler itself.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.db.scoped import LOCAL_SCOPE
from tests._fakes import FakeRequest


@pytest.mark.asyncio
async def test_revoke_runs_off_loop_and_still_deletes(monkeypatch):
    calls = []

    def fake_revoke(engine, name, connector_settings, oauth_settings, *, scope=None):
        calls.append((engine, name, scope))

    monkeypatch.setattr(connections_endpoints.oauth_service, "revoke", fake_revoke)
    monkeypatch.setattr(connections_endpoints.ConnectionsService, "delete", lambda self, engine, name: True)

    result = await connections_endpoints.delete_connection("gmail", "work", LOCAL_SCOPE, FakeRequest())

    assert result is None
    assert calls == [("gmail", "work", LOCAL_SCOPE)]


@pytest.mark.asyncio
async def test_404_when_not_found(monkeypatch):
    monkeypatch.setattr(connections_endpoints.oauth_service, "revoke", lambda *a, **k: None)
    monkeypatch.setattr(connections_endpoints.ConnectionsService, "delete", lambda self, engine, name: False)

    with pytest.raises(HTTPException) as exc_info:
        await connections_endpoints.delete_connection("gmail", "ghost", LOCAL_SCOPE, FakeRequest())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_revoke_failure_is_logged_not_raised(monkeypatch):
    def raising_revoke(*a, **k):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(connections_endpoints.oauth_service, "revoke", raising_revoke)
    monkeypatch.setattr(connections_endpoints.ConnectionsService, "delete", lambda self, engine, name: True)

    result = await connections_endpoints.delete_connection("gmail", "work", LOCAL_SCOPE, FakeRequest())

    assert result is None
