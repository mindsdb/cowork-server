"""Notion page search behind the chat's "Add pages from Notion" picker."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from anton.core.datasources.data_vault import LocalDataVault
from cowork.api.v1.endpoints.connectors.notion import NotionSearchRequest, search_notion_pages
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.notion_pages import MAX_RESULTS, describe_project_pages, parse_search_results

SEARCH_RESPONSE = json.dumps({
    "type": "workspace_search",
    "results": [
        {"id": "p1", "title": "ML Research", "url": "https://www.notion.so/ML-Research-p1", "type": "page",
         "timestamp": "2025-07-22T10:00:00Z", "highlight": "benchmarks"},
        {"id": "d1", "title": "Experiments", "url": "https://www.notion.so/d1", "type": "database"},
        {"id": "u1", "title": "Martyna", "url": "https://www.notion.so/u1", "type": "user"},
        {"id": "", "title": "No id", "url": "https://www.notion.so/x"},
        {"id": "p2", "title": "No url"},
    ],
})


class TestParseSearchResults:
    def test_keeps_pages_and_databases_with_an_id_and_url(self):
        pages = parse_search_results(SEARCH_RESPONSE)
        assert [p["id"] for p in pages] == ["p1", "d1"]
        assert pages[0] == {
            "id": "p1", "title": "ML Research", "url": "https://www.notion.so/ML-Research-p1",
            "type": "page", "timestamp": "2025-07-22T10:00:00Z",
        }

    def test_reads_json_out_of_mcp_text_blocks(self):
        pages = parse_search_results([{"type": "text", "text": SEARCH_RESPONSE}])
        assert [p["id"] for p in pages] == ["p1", "d1"]

    def test_accepts_a_bare_list(self):
        pages = parse_search_results(json.dumps([{"id": "p1", "url": "https://www.notion.so/p1"}]))
        assert pages == [{"id": "p1", "title": "Untitled", "url": "https://www.notion.so/p1", "type": "page", "timestamp": ""}]

    @pytest.mark.parametrize("content", ["not json", json.dumps({"type": "ai_search"}), None, 42])
    def test_unrecognised_shapes_return_nothing(self, content):
        assert parse_search_results(content) == []

    def test_results_are_capped(self):
        many = [{"id": f"p{i}", "url": f"https://www.notion.so/p{i}"} for i in range(MAX_RESULTS + 5)]
        assert len(parse_search_results(json.dumps({"results": many}))) == MAX_RESULTS


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path))
    v = LocalDataVault(tmp_path)
    v.save("notion", "mcp-ws", {"access_token": "tok-mcp", "_method": "mcp", "account_email": "m@x.io:ws1"})
    v.save("notion", "secret-ws", {"internal_integration_secret": "secret_x", "_method": "internal-integration-secret"})
    return v


@pytest.fixture
def notion_server(monkeypatch):
    # Independent of whether the installed anton already lists Notion's MCP URL.
    monkeypatch.setattr("anton.core.mcp.servers.mcp_server_url", lambda engine: "https://mcp.notion.com/mcp")


@pytest.fixture
def mcp_calls(monkeypatch, notion_server):
    calls = []

    async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
        calls.append((engine, access_token, tool_name, kwargs))
        return SEARCH_RESPONSE

    monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)
    return calls


async def test_searches_with_the_connections_stored_token(vault, mcp_calls):
    out = await search_notion_pages(NotionSearchRequest(name="mcp-ws", query=" ML research "), LOCAL_SCOPE, None)
    assert [p["title"] for p in out["results"]] == ["ML Research", "Experiments"]
    assert mcp_calls == [("notion", "tok-mcp", "notion-search", {"query": "ML research"})]


async def test_a_non_mcp_connection_cannot_search(vault, mcp_calls):
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="secret-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 404
    assert mcp_calls == []


async def test_a_blank_query_is_rejected_before_any_call(vault, mcp_calls):
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="   "), LOCAL_SCOPE, None)
    assert exc.value.status_code == 422
    assert mcp_calls == []


async def test_a_rejected_token_asks_for_a_reconnect(vault, notion_server, monkeypatch):
    from anton.core.mcp.errors import McpPermanentError

    async def rejected(*args, **kwargs):
        raise McpPermanentError("401")

    monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", rejected)
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 409
    assert "reconnect" in exc.value.detail


async def test_other_failures_are_a_502(vault, notion_server, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", broken)
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 502


async def test_an_anton_without_the_notion_server_is_a_501(vault, monkeypatch):
    monkeypatch.setattr("anton.core.mcp.servers.mcp_server_url", lambda engine: None)
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 501


async def test_org_mode_takes_the_token_from_auth(monkeypatch, mcp_calls):
    seen = {}

    async def fake_proxy_token(engine, request, settings, *, name=""):
        seen.update(engine=engine, name=name)
        return {"access_token": "tok-auth"}

    monkeypatch.setattr("cowork.services.connectors.oauth.auth_proxy.proxy_token", fake_proxy_token)
    scope = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
    out = await search_notion_pages(NotionSearchRequest(name="ws-a", query="ML"), scope, object())
    assert seen == {"engine": "notion", "name": "ws-a"}
    assert mcp_calls[0][1] == "tok-auth"
    assert len(out["results"]) == 2


class TestProjectPages:
    def _add(self, engine, name, entries):
        assert ConnectionsService(LOCAL_SCOPE).merge_picked_files(engine, name, entries) is not None

    def test_pages_added_to_a_project_are_listed_for_that_project_only(self, vault, tmp_path):
        self._add("notion", "mcp-ws", [
            {"id": "p1", "name": "ML Research", "url": "https://www.notion.so/p1", "projects": ["alpha"]},
            {"id": "p2", "name": "Roadmap", "url": "https://www.notion.so/p2", "projects": ["beta"]},
        ])
        found = ConnectionsService(LOCAL_SCOPE).picked_files_by_project(LocalDataVault(tmp_path), "alpha", engine="notion")
        assert [p["id"] for p in found["mcp-ws"]] == ["p1"]

    def test_adding_a_page_to_a_second_project_keeps_the_first(self, vault, tmp_path):
        self._add("notion", "mcp-ws", [{"id": "p1", "name": "ML Research", "url": "u", "projects": ["alpha"]}])
        self._add("notion", "mcp-ws", [{"id": "p1", "name": "ML Research", "url": "u", "projects": ["beta"]}])
        service = ConnectionsService(LOCAL_SCOPE)
        for project in ("alpha", "beta"):
            assert service.picked_files_by_project(LocalDataVault(tmp_path), project, engine="notion")

    def test_engines_do_not_leak_into_each_other(self, vault, tmp_path):
        vault.save("google_drive", "me-gmail-com", {"access_token": "t"})
        self._add("google_drive", "me-gmail-com", [{"id": "d1", "name": "Sheet", "projects": ["alpha"]}])
        self._add("notion", "mcp-ws", [{"id": "p1", "name": "ML Research", "url": "u", "projects": ["alpha"]}])
        service = ConnectionsService(LOCAL_SCOPE)
        drive = service.picked_files_by_project(LocalDataVault(tmp_path), "alpha")
        notion = service.picked_files_by_project(LocalDataVault(tmp_path), "alpha", engine="notion")
        assert list(drive) == ["me-gmail-com"] and list(notion) == ["mcp-ws"]

    def test_the_agent_is_told_each_page_url_and_how_to_read_it(self):
        text = describe_project_pages({"mcp-ws": [{"id": "p1", "name": "ML Research", "url": "https://www.notion.so/p1"}]})
        assert "- ML Research (url: https://www.notion.so/p1, connection: mcp-ws)" in text
        assert "notion-fetch" in text

    def test_no_pages_adds_nothing_to_the_prompt(self):
        assert describe_project_pages({}) == ""
