"""Notion's MCP connector: the hidden `mcp` method, id-only credentials,
the identity bridge, and best-effort revoke."""
from __future__ import annotations

import json
import sys
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints
from cowork.api.v1.endpoints.connectors.oauth import (
    NOTION_ADMIN_APPROVAL_MESSAGE,
    McpIdentityRequest,
    get_oauth_credentials,
)
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.identity import connection_display_name
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.google import OAuthService
from cowork.services.connectors.specs._registry import ConnectorSpecRegistry

CIMD_URL = "https://mindshub.ai/.well-known/notion-oauth-client.json"


class TestSpec:
    @staticmethod
    def _methods():
        spec = ConnectorSpecRegistry().get_connector("notion")
        return {m.id: m for m in spec.form.methods}

    def test_mcp_method_is_recommended_but_ships_hidden(self):
        mcp = self._methods()["mcp"]
        assert mcp.recommended is True
        assert mcp.hidden is True

    def test_mcp_method_oauth_shape(self):
        oauth = self._methods()["mcp"].oauth
        assert oauth.auth_url == "https://mcp.notion.com/authorize"
        assert oauth.token_url == "https://mcp.notion.com/token"
        assert oauth.revoke_url == "https://mcp.notion.com/token"
        assert oauth.scopes == ["default"]
        assert oauth.redirect_port == 47294
        assert oauth.refresh_token_optional is True
        assert oauth.service_id == "notion"
        assert oauth.service_id in OAUTH_SERVICES

    def test_served_spec_keeps_refresh_token_optional(self):
        """Desktop reads the flag from the spec cowork-server serves, which is
        the model's dump, so an undeclared field would be silently dropped."""
        served = ConnectorSpecRegistry().get_connector("notion").model_dump()
        mcp = next(m for m in served["form"]["methods"] if m["id"] == "mcp")
        assert mcp["oauth"]["refresh_token_optional"] is True

    def test_older_methods_stay_defined_but_step_back(self):
        methods = self._methods()
        assert methods["internal-integration-secret"].recommended is False
        assert methods["oauth"].hidden is True

    def test_other_connectors_do_not_get_refresh_token_optional(self):
        hubspot = ConnectorSpecRegistry().get_connector("hubspot")
        mcp = next(m for m in hubspot.form.methods if m.id == "mcp")
        assert mcp.oauth.refresh_token_optional is False


def test_credentials_are_id_only(monkeypatch):
    monkeypatch.setenv("NOTION_CLIENT_ID", CIMD_URL)
    assert get_oauth_credentials("notion") == {"client_id": CIMD_URL, "client_secret": ""}


def test_credentials_missing_client_id_is_reported(monkeypatch):
    monkeypatch.setenv("NOTION_CLIENT_ID", "")
    with pytest.raises(HTTPException) as exc:
        get_oauth_credentials("notion")
    assert exc.value.status_code == 422


def test_tile_subtitle_prefers_workspace_name_over_composite_identity():
    fields = {"account_email": "ann@acme.com:ws-1", "account_name": "Acme"}
    assert connection_display_name(fields, "notion") == "Acme"


try:  # pragma: no cover - import probe, not logic
    from anton.core.mcp import servers as anton_mcp_servers

    HAS_ANTON_MCP = True
except ImportError:  # pragma: no cover
    HAS_ANTON_MCP = False


@pytest.mark.skipif(not HAS_ANTON_MCP, reason="installed anton has no anton.core.mcp")
class TestIdentityBridge:
    @pytest.fixture(autouse=True)
    def _notion_server_url(self, monkeypatch):
        # The anton release that adds Notion's URL may not be installed yet.
        monkeypatch.setitem(anton_mcp_servers.MCP_SERVER_URLS, "notion", "https://mcp.notion.com/mcp")

    @staticmethod
    def _tool_returns(monkeypatch, result, calls=None):
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            if calls is not None:
                calls.append((engine, access_token, tool_name, kwargs))
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

    async def test_identity_is_email_and_workspace_and_name_is_the_workspace(self, monkeypatch):
        calls = []
        self._tool_returns(
            monkeypatch,
            json.dumps({"results": [{"id": "u-1", "type": "person", "email": "ann@acme.com"}]}),
            calls,
        )
        result = await oauth_endpoints.get_mcp_identity(
            "notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1", workspace_name="Acme"),
        )
        assert result == {"account_email": "ann@acme.com:ws-1", "account_name": "Acme"}
        assert calls == [("notion", "tok", "notion-get-users", {"user_id": "self"})]

    async def test_name_falls_back_to_email_without_a_workspace_name(self, monkeypatch):
        self._tool_returns(monkeypatch, {"id": "u-1", "person": {"email": "ann@acme.com"}})
        result = await oauth_endpoints.get_mcp_identity(
            "notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1"),
        )
        assert result == {"account_email": "ann@acme.com:ws-1", "account_name": "ann@acme.com"}

    async def test_user_id_stands_in_when_notion_returns_no_email(self, monkeypatch):
        self._tool_returns(monkeypatch, json.dumps({"id": "u-1", "type": "person"}))
        result = await oauth_endpoints.get_mcp_identity(
            "notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1"),
        )
        assert result == {"account_email": "u-1:ws-1", "account_name": "u-1"}

    async def test_two_workspaces_get_distinct_identities(self, monkeypatch):
        self._tool_returns(monkeypatch, {"id": "u-1", "email": "ann@acme.com"})
        a = await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="t", workspace_id="ws-1"))
        b = await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="t", workspace_id="ws-2"))
        assert a["account_email"] != b["account_email"]

    async def test_no_identity_at_all_502s(self, monkeypatch):
        self._tool_returns(monkeypatch, "{}")
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok", workspace_id="ws-1"))
        assert exc.value.status_code == 502
        assert exc.value.detail != NOTION_ADMIN_APPROVAL_MESSAGE

    async def test_a_tool_error_returns_the_admin_approval_message(self, monkeypatch):
        self._tool_returns(monkeypatch, RuntimeError("notion MCP tool 'notion-get-users' returned an error: 'blocked'"))
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok"))
        assert exc.value.detail == NOTION_ADMIN_APPROVAL_MESSAGE

    async def test_a_wrapped_403_returns_the_admin_approval_message(self, monkeypatch):
        request = httpx.Request("POST", "https://mcp.notion.com/mcp")
        forbidden = httpx.HTTPStatusError("403", request=request, response=httpx.Response(403, request=request))
        self._tool_returns(monkeypatch, ExceptionGroup("task group", [forbidden]))
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok"))
        assert exc.value.detail == NOTION_ADMIN_APPROVAL_MESSAGE

    async def test_a_network_failure_is_not_reported_as_an_admin_block(self, monkeypatch):
        self._tool_returns(monkeypatch, httpx.ConnectError("connection refused"))
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok"))
        assert exc.value.status_code == 502
        assert exc.value.detail != NOTION_ADMIN_APPROVAL_MESSAGE


@pytest.mark.skipif(not HAS_ANTON_MCP, reason="installed anton has no anton.core.mcp")
async def test_501_when_the_installed_anton_has_no_notion_server_url(monkeypatch):
    """Otherwise call_mcp_tool's "no known MCP server URL" error would show
    users the admin-approval message."""
    monkeypatch.delitem(anton_mcp_servers.MCP_SERVER_URLS, "notion", raising=False)
    with pytest.raises(HTTPException) as exc:
        await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok"))
    assert exc.value.status_code == 501


async def test_501_when_the_installed_anton_has_no_mcp_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "anton.core.mcp.wiring", None)
    with pytest.raises(HTTPException) as exc:
        await oauth_endpoints.get_mcp_identity("notion", McpIdentityRequest(access_token="tok"))
    assert exc.value.status_code == 501


class _FakeVault:
    def __init__(self, path):
        pass

    def load(self, engine, name):
        return {"auth_type": "oauth", "access_token": "at-1", "refresh_token": "rt-1"}


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_revoke_posts_the_refresh_token_and_client_id_without_a_secret(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)
    captured = {}

    def _fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["body"] = parse_qs(request.data.decode())
        return _FakeResponse()

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    OAuthService().revoke("notion", "ann", ConnectorSettings(), OAuthSettings(_env_file=None, NOTION_CLIENT_ID=CIMD_URL))

    assert captured["url"] == "https://mcp.notion.com/token"
    assert captured["body"] == {"token": ["rt-1"], "client_id": [CIMD_URL]}


def test_revoke_failure_does_not_block_disconnect(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)

    def _fail(request, timeout=10):
        raise OSError("connection reset")

    monkeypatch.setattr(google_module, "urlopen", _fail)

    OAuthService().revoke("notion", "ann", ConnectorSettings(), OAuthSettings(_env_file=None, NOTION_CLIENT_ID=CIMD_URL))
