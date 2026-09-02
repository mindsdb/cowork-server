"""Org-mode branches of the connections endpoints — no durable local vault
to read/write in org mode, so these proxy to auth instead (same mechanism
as the OAuth Connector Lifecycle endpoints). Direct function calls, no
TestClient/app fixture — matches this repo's existing convention (see
test_oauth_picker_token.py).
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.db.scoped import TenantScope
from cowork.schemas.connectors import ConnectionSummaryResponse
from tests._fakes import FakeRequest

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")


@pytest.mark.asyncio
async def test_list_connections_flattens_the_catalogue(monkeypatch):
    async def fake_proxy_catalogue(request, settings):
        return {
            "items": [
                {
                    "id": "google_drive",
                    "connections": [{"engine": "google_drive", "name": "work", "user_label": "a@b.com"}],
                },
                {"id": "gmail", "connections": []},
            ]
        }

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_catalogue", fake_proxy_catalogue)

    result = await connections_endpoints.list_connections(ORG_SCOPE, FakeRequest())

    assert result == [ConnectionSummaryResponse(engine="google_drive", name="work", user_label="a@b.com")]


@pytest.mark.asyncio
async def test_get_connection_surfaces_non_secret_fields_only(monkeypatch):
    async def fake_proxy_connection_detail(engine, name, request, settings):
        assert engine == "google_drive" and name == "work"
        return {
            "status": "active",
            "account_email": "a@b.com",
            "token_type": "Bearer",
            "scope": "drive.file",
            "expires_at": "2026-08-27T00:00:00Z",
            "picked_files": [],
        }

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_connection_detail", fake_proxy_connection_detail)

    detail = await connections_endpoints.get_connection("google_drive", "work", ORG_SCOPE, FakeRequest())

    assert detail.engine == "google_drive"
    assert detail.name == "work"
    assert detail.fields == {
        "status": "active",
        "account_email": "a@b.com",
        "token_type": "Bearer",
        "scope": "drive.file",
        "expires_at": "2026-08-27T00:00:00Z",
        "_picked_files": "[]",
    }
    assert "access_token" not in detail.fields


@pytest.mark.asyncio
async def test_get_connection_includes_picked_files_as_json_string(monkeypatch):
    """ENG-2097: the actual regression. auth's Data Vault stores
    _picked_files as a native list (see auth's merge_picked_files
    docstring); desktop's local vault stores it as a JSON-encoded string
    (ConnectionsService.merge_picked_files), and CustomizeView.jsx's
    JSON.parse(fields._picked_files) only understands the latter — so the
    org-mode bridge has to convert, not just pass the field through."""
    async def fake_proxy_connection_detail(engine, name, request, settings):
        return {
            "status": "active",
            "account_email": "a@b.com",
            "token_type": "Bearer",
            "scope": "drive.file",
            "expires_at": "2026-08-27T00:00:00Z",
            "picked_files": [{"id": "f1", "name": "a.txt", "projects": []}],
        }

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_connection_detail", fake_proxy_connection_detail)

    detail = await connections_endpoints.get_connection("google_drive", "work", ORG_SCOPE, FakeRequest())

    assert detail.fields["_picked_files"] == '[{"id": "f1", "name": "a.txt", "projects": []}]'


@pytest.mark.asyncio
async def test_get_connection_surfaces_needs_reconnect_status(monkeypatch):
    """The other half of ENG-2097's fix: proxy_connection_detail (unlike the
    proxy_token this replaced) never raises on a stuck connection, so its
    real status reaches the panel instead of a 403."""
    async def fake_proxy_connection_detail(engine, name, request, settings):
        return {
            "status": "needs_reconnect",
            "account_email": "a@b.com",
            "token_type": "Bearer",
            "scope": "",
            "expires_at": "",
            "picked_files": [],
        }

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_connection_detail", fake_proxy_connection_detail)

    detail = await connections_endpoints.get_connection("google_drive", "work", ORG_SCOPE, FakeRequest())

    assert detail.fields["status"] == "needs_reconnect"


@pytest.mark.asyncio
async def test_get_connection_propagates_auths_404(monkeypatch):
    async def fake_proxy_connection_detail(engine, name, request, settings):
        raise HTTPException(status_code=404, detail="No google_drive connection named 'ghost' for this user.")

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_connection_detail", fake_proxy_connection_detail)

    with pytest.raises(HTTPException) as exc_info:
        await connections_endpoints.get_connection("google_drive", "ghost", ORG_SCOPE, FakeRequest())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_connection_proxies_to_auth(monkeypatch):
    calls = []

    async def fake_proxy_delete(engine, name, request, settings):
        calls.append((engine, name))

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_delete", fake_proxy_delete)

    result = await connections_endpoints.delete_connection("google_drive", "work", ORG_SCOPE, FakeRequest())

    assert result is None
    assert calls == [("google_drive", "work")]


@pytest.mark.asyncio
async def test_delete_connection_propagates_auths_404(monkeypatch):
    async def fake_proxy_delete(engine, name, request, settings):
        raise HTTPException(status_code=404, detail="No google_drive connection named 'ghost' for this user.")

    monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_delete", fake_proxy_delete)

    with pytest.raises(HTTPException) as exc_info:
        await connections_endpoints.delete_connection("google_drive", "ghost", ORG_SCOPE, FakeRequest())
    assert exc_info.value.status_code == 404


def test_patch_connection_token_is_a_501_in_org_mode():
    """No org-mode caller reaches this route today (Electron main's
    token-refresh.ts always targets the local desktop sidecar) — guarded
    explicitly so a future org-mode caller fails loudly instead of writing
    to the wrong vault."""
    body = connections_endpoints.PatchTokenBody(access_token="t")
    with pytest.raises(HTTPException) as exc_info:
        connections_endpoints.patch_connection_token("google_drive", "work", body, ORG_SCOPE)
    assert exc_info.value.status_code == 501
