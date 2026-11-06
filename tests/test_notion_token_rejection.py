"""A rejected Notion token through anton's real MCP client: the MCP SDK turns
an HTTP 401/403 into a JSON-RPC error with no status, so the reconnect and
admin-approval answers depend on anton's McpTokenRejected."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

errors = pytest.importorskip("anton.core.mcp.errors")
if not hasattr(errors, "McpTokenRejected"):  # pragma: no cover - older anton
    pytest.skip("installed anton has no McpTokenRejected", allow_module_level=True)
httpx2 = pytest.importorskip("httpx2")

from anton.core.datasources.data_vault import LocalDataVault  # noqa: E402
from anton.core.mcp import servers as anton_mcp_servers  # noqa: E402

from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints  # noqa: E402
from cowork.api.v1.endpoints.connectors.notion import NotionSearchRequest, search_notion_pages  # noqa: E402
from cowork.api.v1.endpoints.connectors.oauth import NOTION_ADMIN_APPROVAL_MESSAGE, McpIdentityRequest  # noqa: E402
from cowork.db.scoped import LOCAL_SCOPE  # noqa: E402


def _handler(*, open_status: int, call_status: int):
    def handle(request):
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        if message.get("method") == "initialize":
            if open_status != 200:
                return httpx2.Response(open_status)
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-notion", "version": "1"},
            }})
        if "id" not in message:
            return httpx2.Response(202)
        if call_status != 200:
            return httpx2.Response(call_status)
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {
            "content": [{"type": "text", "text": json.dumps({"results": []})}]}})

    return handle


@pytest.fixture()
def notion_mcp(monkeypatch):
    monkeypatch.setitem(anton_mcp_servers.MCP_SERVER_URLS, "notion", "https://mcp.notion.com/mcp")
    real_client = httpx2.AsyncClient

    def install(*, open_status: int = 200, call_status: int = 200):
        transport = httpx2.MockTransport(_handler(open_status=open_status, call_status=call_status))

        # A subclass, not a function: other libraries subclass AsyncClient on import.
        class FakeNotionClient(real_client):
            def __init__(self, **kw):
                kw.setdefault("transport", transport)
                super().__init__(**kw)

        monkeypatch.setattr(httpx2, "AsyncClient", FakeNotionClient)

    return install


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path))
    LocalDataVault(tmp_path).save("notion", "mcp-ws", {"access_token": "tok", "_method": "mcp"})


@pytest.mark.parametrize(("where", "status"), [("open", 401), ("open", 403), ("call", 401), ("call", 403)])
async def test_a_rejected_token_asks_for_a_reconnect(vault, notion_mcp, where, status):
    notion_mcp(**{f"{where}_status": status})
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 409


async def test_a_server_error_is_not_a_reconnect(vault, notion_mcp):
    notion_mcp(open_status=500)
    with pytest.raises(HTTPException) as exc:
        await search_notion_pages(NotionSearchRequest(name="mcp-ws", query="x"), LOCAL_SCOPE, None)
    assert exc.value.status_code == 502


@pytest.mark.parametrize("where", ["open", "call"])
async def test_a_403_on_connect_shows_the_admin_approval_message(notion_mcp, where):
    notion_mcp(**{f"{where}_status": 403})
    with pytest.raises(HTTPException) as exc:
        await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1"))
    assert exc.value.detail == NOTION_ADMIN_APPROVAL_MESSAGE


async def test_a_500_on_connect_is_not_an_admin_block(notion_mcp):
    notion_mcp(open_status=500)
    with pytest.raises(HTTPException) as exc:
        await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1"))
    assert exc.value.detail != NOTION_ADMIN_APPROVAL_MESSAGE
