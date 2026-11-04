"""Drive file search behind the in-app file chooser: the Drive API call it
makes, and how the route gets the token and reports refusals."""
from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import google_drive as drive_endpoints
from cowork.db.scoped import TenantScope
from cowork.services.connectors import google_drive_files
from cowork.services.connectors.google_drive_files import (
    DriveAccessDenied,
    DriveSearchError,
    drive_query,
    search_files,
)
from tests._fakes import FakeRequest

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
LOCAL_SCOPE = TenantScope(org_mode=False)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_empty_query_lists_files_not_folders_or_trash():
    q = drive_query("  ")
    assert q == "trashed = false and mimeType != 'application/vnd.google-apps.folder'"


def test_name_search_escapes_quotes_and_backslashes():
    q = drive_query("Q3 'plan' \\ draft")
    assert q.endswith("name contains 'Q3 \\'plan\\' \\\\ draft'")


@pytest.mark.asyncio
async def test_search_asks_drive_for_recent_files_across_all_drives():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={
            "nextPageToken": "next-1",
            "files": [
                {
                    "id": "f1", "name": "Roadmap", "mimeType": "application/vnd.google-apps.document",
                    "iconLink": "https://icon", "webViewLink": "https://docs/f1", "resourceKey": "rk",
                    "modifiedTime": "2026-11-01T10:00:00Z", "owners": [{"displayName": "Ana"}],
                },
                {"name": "no id, dropped"},
            ],
        })

    async with _client(handler) as client:
        result = await search_files("tok", "road", "page-0", client=client)

    assert seen["auth"] == "Bearer tok"
    params = seen["params"]
    assert params["orderBy"].startswith("viewedByMeTime desc")
    assert params["corpora"] == "allDrives"
    assert params["supportsAllDrives"] == "true"
    assert params["includeItemsFromAllDrives"] == "true"
    assert params["pageToken"] == "page-0"
    assert "name contains 'road'" in params["q"]
    assert result == {
        "files": [{
            "id": "f1", "name": "Roadmap", "mimeType": "application/vnd.google-apps.document",
            "iconUrl": "https://icon", "url": "https://docs/f1", "resourceKey": "rk",
            "modifiedTime": "2026-11-01T10:00:00Z", "owner": "Ana",
        }],
        "next_page_token": "next-1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403])
async def test_refused_token_is_access_denied(code):
    async with _client(lambda r: httpx.Response(code, json={})) as client:
        with pytest.raises(DriveAccessDenied):
            await search_files("tok", client=client)


@pytest.mark.asyncio
async def test_other_drive_errors_are_search_errors():
    async with _client(lambda r: httpx.Response(500, json={})) as client:
        with pytest.raises(DriveSearchError) as exc:
            await search_files("tok", client=client)
    assert not isinstance(exc.value, DriveAccessDenied)


def _stub_search(monkeypatch, *, raises=None):
    seen = {}

    async def fake_search(access_token, query="", page_token=""):
        seen.update(token=access_token, query=query, page_token=page_token)
        if raises:
            raise raises
        return {"files": [], "next_page_token": ""}

    monkeypatch.setattr(drive_endpoints, "search_files", fake_search)
    return seen


@pytest.mark.asyncio
async def test_local_search_uses_the_connections_own_token(monkeypatch):
    monkeypatch.setattr(
        drive_endpoints.ConnectionsService, "runtime_fields",
        lambda self, engine, name: {"access_token": f"{engine}:{name}"},
    )
    seen = _stub_search(monkeypatch)
    body = drive_endpoints.DriveSearchRequest(name="work", query="road")

    result = await drive_endpoints.search_drive_files(body, FakeRequest(), LOCAL_SCOPE)

    assert result == {"files": [], "next_page_token": ""}
    assert seen == {"token": "google_drive:work", "query": "road", "page_token": ""}


@pytest.mark.asyncio
async def test_org_search_gets_the_token_through_the_proxy_for_that_connection(monkeypatch):
    asked = {}

    async def fake_proxy_token(engine, request, settings, *, name=""):
        asked.update(engine=engine, name=name)
        return {"access_token": "org-token"}

    monkeypatch.setattr(drive_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
    seen = _stub_search(monkeypatch)
    body = drive_endpoints.DriveSearchRequest(name="work")

    await drive_endpoints.search_drive_files(body, FakeRequest(headers={"authorization": "Bearer u"}), ORG_SCOPE)

    assert asked == {"engine": "google_drive", "name": "work"}
    assert seen["token"] == "org-token"


@pytest.mark.asyncio
async def test_unknown_local_connection_is_404(monkeypatch):
    monkeypatch.setattr(drive_endpoints.ConnectionsService, "runtime_fields", lambda self, e, n: None)
    _stub_search(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        await drive_endpoints.search_drive_files(
            drive_endpoints.DriveSearchRequest(name="gone"), FakeRequest(), LOCAL_SCOPE,
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(("raises", "code"), [
    (DriveAccessDenied("403"), 409),
    (DriveSearchError("500"), 502),
])
async def test_drive_refusals_map_to_reconnect_or_bad_gateway(monkeypatch, raises, code):
    monkeypatch.setattr(
        drive_endpoints.ConnectionsService, "runtime_fields", lambda self, e, n: {"access_token": "t"},
    )
    _stub_search(monkeypatch, raises=raises)
    with pytest.raises(HTTPException) as exc:
        await drive_endpoints.search_drive_files(
            drive_endpoints.DriveSearchRequest(name="work"), FakeRequest(), LOCAL_SCOPE,
        )
    assert exc.value.status_code == code
    if code == 409:
        assert "See and download all your Google Drive files" in exc.value.detail


def test_module_keeps_the_drive_api_url_literal():
    assert google_drive_files.DRIVE_FILES_URL == "https://www.googleapis.com/drive/v3/files"
