"""Browsing a saved SharePoint connection reaches only the sites it names.

Graph is answered by a fake Microsoft at the HTTP transport. Browse returns
fixed fields, one page per call with a cursor that can only page on, and
never a pre-authenticated download link.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import sharepoint as sharepoint_ep
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.principal import personal_partition_id
from cowork.services.connectors import sharepoint as sharepoint_service
from cowork.services.connectors.persist import persist_connection

SECRET = "sp-browse-secret-4e8b"
TENANT = "contoso.onmicrosoft.com"
CLIENT_ID = "11111111-2222-3333-4444-555555555555"
SITE = "https://contoso.sharepoint.com/sites/team"
OTHER_SITE = "https://contoso.sharepoint.com/sites/finance"
SITE_ID = "contoso.sharepoint.com,aaaa-1111,bbbb-2222"
DRIVE_ID = "b!Lib_1"
SITE_PATH = "/v1.0/sites/contoso.sharepoint.com:/sites/team"
DRIVES_PATH = f"/v1.0/sites/{SITE_ID}/drives"
CHILDREN_PATH = f"{DRIVES_PATH}/{DRIVE_ID}/items/root/children"
STORED = {"tenant_id": TENANT, "client_id": CLIENT_ID, "client_secret": SECRET, "site_urls": SITE}
BASE = "/api/v1/connectors/sharepoint/connections"

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
ALICE = TenantScope(personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_A), user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)
BOB = TenantScope(personal=True, org_id=personal_partition_id(DEPLOYMENT, SUB_B), user_id=SUB_B, personal_resources=ALL_PERSONAL_RESOURCES)


def _file(n: int) -> dict:
    return {
        "id": f"01F{n:04d}",
        "name": f"file-{n:03d}.txt",
        "size": n,
        "file": {},
        "lastModifiedDateTime": "2026-10-26T00:00:00Z",
        "@microsoft.graph.downloadUrl": f"https://contoso.sharepoint.com/_layouts/download.aspx?tempauth=leak{n}",
    }


class Graph:
    """A fake Entra and Graph for one granted site with one library holding 201 files."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.token_response = httpx.Response(200, json={"access_token": "browse-token", "expires_in": 3600})
        self.site_response: httpx.Response | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "login.microsoftonline.com":
            return self.token_response
        path = request.url.path
        if path == SITE_PATH:
            return self.site_response or httpx.Response(200, json={"id": SITE_ID})
        if path == DRIVES_PATH:
            return httpx.Response(200, json={"value": [{"id": DRIVE_ID, "name": "Documents"}]})
        if path == CHILDREN_PATH:
            if request.url.params.get("$skiptoken") == "p2":
                return httpx.Response(200, json={"value": [_file(200)]})
            if request.url.params.get("$skiptoken") == "stale":
                return httpx.Response(400, json={"error": {"code": "invalidRequest"}})
            next_link = f"https://graph.microsoft.com{CHILDREN_PATH}?$top=200&$skiptoken=p2"
            return httpx.Response(200, json={"value": [_file(n) for n in range(200)], "@odata.nextLink": next_link})
        return httpx.Response(404, json={"error": {"code": "itemNotFound"}})

    def graph_paths(self) -> list[str]:
        return [r.url.path for r in self.requests if r.url.host == "graph.microsoft.com"]


@pytest.fixture
def graph(monkeypatch, sharepoint_stores):
    fake = Graph()
    monkeypatch.setattr(
        sharepoint_ep,
        "SharePointClient",
        lambda key: sharepoint_service.SharePointClient(key, transport=httpx.MockTransport(fake.handler)),
    )
    return fake


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def _save(fields: dict = STORED, name: str = "corp", scope: TenantScope = LOCAL_SCOPE) -> None:
    persist_connection("sharepoint", None, name, dict(fields), scope=scope)


def test_libraries_of_a_named_site_are_listed(graph):
    _save()

    response = _client().get(f"{BASE}/corp/libraries", params={"site": "https://Contoso.sharepoint.com/sites/Team/"})

    assert response.status_code == 200, response.text
    assert response.json() == {"libraries": [{"id": DRIVE_ID, "name": "Documents"}]}


def test_a_folder_of_201_items_is_read_over_two_pages(graph):
    _save()
    client = _client()

    first = client.get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID})
    second = client.get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID, "cursor": first.json()["cursor"]})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert len(first.json()["items"]) == 200
    assert first.json()["cursor"] == "p2"
    assert [item["name"] for item in second.json()["items"]] == ["file-200.txt"]
    assert second.json()["cursor"] is None
    item = first.json()["items"][0]
    assert item == {"id": "01F0000", "name": "file-000.txt", "size": 0, "kind": "file", "modified": "2026-10-26T00:00:00Z"}


def test_browse_never_returns_a_download_link(graph):
    _save()

    response = _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID})

    assert "tempauth" not in response.text
    assert "downloadUrl" not in response.text
    assert SECRET not in response.text


@pytest.mark.parametrize(
    "cursor",
    [
        "https://evil.example/v1.0/sites/other/drives?$skiptoken=p2",
        "../../../sites/other/drives",
        "p2&$filter=x",
    ],
)
def test_a_forged_cursor_cannot_change_the_request_path(graph, cursor):
    _save()

    _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID, "cursor": cursor})

    children = [r for r in graph.requests if r.url.path.endswith("/children")]
    assert [r.url.path for r in children] == [CHILDREN_PATH]
    assert children[0].url.host == "graph.microsoft.com"
    assert children[0].url.params["$skiptoken"] == cursor
    assert all(r.url.host != "evil.example" for r in graph.requests)


def test_a_site_the_connection_does_not_name_is_refused_before_any_graph_call(graph):
    _save()

    response = _client().get(f"{BASE}/corp/libraries", params={"site": OTHER_SITE})

    assert response.status_code == 403
    assert graph.requests == []


def test_a_library_from_another_site_is_not_found(graph):
    _save()

    response = _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": "b!SomeOtherSitesLibrary"})

    assert response.status_code == 404
    assert not any(path.endswith("/children") for path in graph.graph_paths())


@pytest.mark.parametrize(("drive_id", "item_id"), [("../x", "root"), (DRIVE_ID, "../../sites/x"), (DRIVE_ID, "a b")])
def test_ids_outside_graphs_charset_are_refused(graph, drive_id, item_id):
    _save()

    response = _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": drive_id, "item_id": item_id})

    assert response.status_code == 400
    assert not any(path.endswith("/children") for path in graph.graph_paths())


def test_a_cursor_graph_no_longer_accepts_asks_to_reopen_the_folder(graph):
    _save()

    response = _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID, "cursor": "stale"})

    assert response.status_code == 400
    assert response.json()["detail"] == "This folder listing is out of date. Open the folder again."


def test_an_overlong_cursor_is_refused(graph):
    _save()

    response = _client().get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID, "cursor": "x" * 4097})

    assert response.status_code == 400
    assert graph.requests == []


@pytest.mark.parametrize("name", ["corp_", "CORP", "missing"])
def test_a_name_that_is_not_exactly_stored_is_not_found(graph, name):
    _save()

    response = _client().get(f"{BASE}/{name}/libraries", params={"site": SITE})

    assert response.status_code == 404
    assert graph.requests == []


@pytest.mark.parametrize(
    "fields",
    [
        {**STORED, "tenant_id": "contoso.com/evil"},
        {**STORED, "site_urls": "https://evil.example/sites/team"},
        {**STORED, "site_urls": "https://contoso.sharepoint.com/sites/team:/x"},
        {k: v for k, v in STORED.items() if k != "client_secret"},
    ],
)
def test_a_stored_record_that_fails_validation_is_refused_before_any_request(graph, fields):
    _save(fields)

    response = _client().get(f"{BASE}/corp/libraries", params={"site": SITE})

    assert response.status_code == 409
    assert graph.requests == []


@pytest.mark.parametrize(
    ("token_response", "site_response", "expected"),
    [
        (httpx.Response(401, json={"error_codes": [7000222]}), None, 409),
        (httpx.Response(400, json={"error_codes": [700016]}), None, 409),
        (None, httpx.Response(429, headers={"Retry-After": "600"}), 429),
        (None, httpx.Response(500, json={}), 502),
        (None, httpx.Response(403, json={}), 404),
    ],
)
def test_graph_failures_map_to_browse_statuses(graph, token_response, site_response, expected):
    _save()
    if token_response is not None:
        graph.token_response = token_response
    graph.site_response = site_response

    response = _client().get(f"{BASE}/corp/libraries", params={"site": SITE})

    assert response.status_code == expected
    assert SECRET not in response.text
    assert "graph.microsoft.com" not in response.text
    if expected == 409:
        assert response.json()["detail"] == sharepoint_ep._KEY_NO_LONGER_WORKS


def test_another_testers_private_connection_is_not_found(personal_deployment, graph):
    get_app_settings.cache_clear()
    _save(scope=ALICE)

    with pytest.raises(HTTPException) as refused:
        sharepoint_ep.list_libraries("corp", SITE, BOB)

    assert refused.value.status_code == 404
    assert graph.requests == []


def test_a_connection_shared_through_the_registry_is_browsable(personal_deployment, managed_dir, graph):
    get_app_settings.cache_clear()
    _save(scope=ALICE)
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(
        json.dumps([{"user_id": SUB_A, "partition_id": ALICE.org_id, "engine": "sharepoint", "name": "corp"}])
    )

    libraries = sharepoint_ep.list_libraries("corp", SITE, BOB)

    assert [library.id for library in libraries.libraries] == [DRIVE_ID]


def test_org_mode_refuses_browse(monkeypatch, graph):
    _save()
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    client = _client()

    libraries = client.get(f"{BASE}/corp/libraries", params={"site": SITE})
    children = client.get(f"{BASE}/corp/children", params={"site": SITE, "drive_id": DRIVE_ID})

    assert libraries.status_code == 403
    assert children.status_code == 403
    assert graph.requests == []
