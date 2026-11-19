"""Slack's MCP connector: the `mcp` method, credentials, the REST identity,
best-effort revoke, and Code Projects reading through an MCP connection."""
from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.endpoints.connectors.oauth import get_oauth_credentials
from cowork.api.v1.router import api_router
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import SourceContextRequest
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors import app_held
from cowork.services.connectors.identity import connection_display_name
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.google import OAuthService
from cowork.services.connectors.persist import own_vault_for_scope, own_vault_root, persist_connection
from cowork.services.connectors.specs._registry import ConnectorSpecRegistry

SLACK_SCOPES = [
    "search:read.public", "search:read.private", "search:read.im", "search:read.mpim",
    "search:read.files", "search:read.users",
    "channels:history", "channels:read", "channels:write",
    "groups:history", "groups:read", "groups:write",
    "im:history", "im:read", "im:write",
    "mpim:history", "mpim:read", "mpim:write",
    "chat:write", "reactions:read", "reactions:write",
    "canvases:read", "canvases:write", "lists:read", "lists:write",
    "files:read", "files:write", "emoji:read", "users:read", "users:read.email",
]


def _methods():
    return {m.id: m for m in ConnectorSpecRegistry().get_connector("slack").form.methods}


@pytest.fixture
def slack_on(monkeypatch):
    monkeypatch.setenv("SLACK_CONNECTOR_ENABLED", "true")


@pytest.fixture
def slack_off(monkeypatch):
    monkeypatch.setenv("SLACK_CONNECTOR_ENABLED", "false")


@pytest.mark.usefixtures("slack_on")
class TestSpec:
    def test_mcp_method_is_recommended_and_visible(self):
        mcp = _methods()["mcp"]
        assert mcp.label == "In-Browser Connect"
        assert mcp.recommended is True
        assert mcp.hidden is False
        assert mcp.fields == []

    def test_mcp_method_oauth_shape(self):
        oauth = _methods()["mcp"].oauth
        assert oauth.auth_url == "https://slack.com/oauth/v2_user/authorize"
        assert oauth.token_url == "https://slack.com/api/oauth.v2.user.access"
        assert oauth.scopes == SLACK_SCOPES
        assert oauth.revoke_url is None
        assert oauth.supports_refresh is True
        assert oauth.supports_revoke is True
        assert oauth.redirect_host == "localhost"
        assert oauth.redirect_port == 47297
        assert oauth.service_id == "slack"
        assert oauth.service_id in OAUTH_SERVICES

    def test_older_methods_stay_defined_but_step_back(self):
        methods = _methods()
        assert methods["bot-token"].recommended is False
        assert methods["bot-token"].hidden is False
        assert methods["oauth"].hidden is True

    def test_slack_is_an_mcp_engine(self):
        assert "slack" in ConnectorSpecRegistry().mcp_engines()

    def test_no_other_connector_uses_the_loopback_port(self):
        users = [
            (engine, method.get("id"))
            for engine, raw in ConnectorSpecRegistry().get_connectors().items()
            for method in ((raw.get("form") or {}).get("methods") or [])
            if (method.get("oauth") or {}).get("redirect_port") == 47297
        ]
        assert users == [("slack", "mcp")]


class TestSwitchedOff:
    """SLACK_CONNECTOR_ENABLED off, as in prod for now: the spec served to
    clients shows the same methods as before the `mcp` method, and new
    In-Browser Connect sign-ins are refused."""

    def test_the_mcp_method_is_hidden_and_the_bot_token_is_the_default(self, slack_off):
        methods = _methods()
        assert methods["mcp"].hidden is True
        assert methods["mcp"].recommended is False
        assert methods["bot-token"].recommended is True
        assert methods["oauth"].hidden is False

    def test_switching_on_shows_the_mcp_method(self, slack_on):
        methods = _methods()
        assert methods["mcp"].hidden is False
        assert methods["mcp"].recommended is True
        assert methods["bot-token"].recommended is False
        assert methods["oauth"].hidden is True

    @staticmethod
    def _client() -> TestClient:
        app = FastAPI()
        app.include_router(api_router)
        return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))

    @staticmethod
    def _save(client: TestClient, name: str = "", replace_existing: bool = False):
        return client.post("/api/v1/connectors/connections/save", json={
            "connector_id": "slack", "method": "mcp", "name": name, "replace_existing": replace_existing,
            "values": {"access_token": "xoxp-1", "account_email": "ann@acme.com:T1", "account_name": "Acme"},
        })

    def test_a_new_mcp_connection_is_refused(self, sharepoint_stores, slack_off):
        response = self._save(self._client())
        assert response.status_code == 403
        assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []

    def test_a_new_mcp_connection_saves_when_switched_on(self, sharepoint_stores, slack_on):
        response = self._save(self._client())
        assert response.status_code == 200, response.text
        assert [c["engine"] for c in own_vault_for_scope(LOCAL_SCOPE).list_connections()] == ["slack"]

    def test_reconnecting_an_existing_mcp_connection_still_saves(self, sharepoint_stores, monkeypatch):
        client = self._client()
        monkeypatch.setenv("SLACK_CONNECTOR_ENABLED", "true")
        name = self._save(client).json()["name"]
        monkeypatch.setenv("SLACK_CONNECTOR_ENABLED", "false")

        response = self._save(client, name=name, replace_existing=True)

        assert response.status_code == 200, response.text
        assert own_vault_for_scope(LOCAL_SCOPE).load("slack", name)["access_token"] == "xoxp-1"

    def test_the_bot_token_method_still_saves(self, sharepoint_stores, slack_off):
        response = self._client().post("/api/v1/connectors/connections/save", json={
            "connector_id": "slack", "method": "bot-token", "values": {"bot_token": "xoxb-1"},
        })
        assert response.status_code == 200, response.text

    def test_a_bot_token_connection_cannot_be_replaced_with_an_mcp_one(self, sharepoint_stores, slack_off):
        client = self._client()
        name = client.post("/api/v1/connectors/connections/save", json={
            "connector_id": "slack", "method": "bot-token", "values": {"bot_token": "xoxb-1"},
        }).json()["name"]

        response = self._save(client, name=name, replace_existing=True)

        assert response.status_code == 403
        assert own_vault_for_scope(LOCAL_SCOPE).load("slack", name)["bot_token"] == "xoxb-1"

    def test_starting_a_slack_sign_in_is_refused(self, sharepoint_stores, slack_off):
        response = self._client().post("/api/v1/connectors/oauth/slack/start", json={})
        assert response.status_code == 404


def test_credentials_need_both_id_and_secret(monkeypatch):
    monkeypatch.setenv("SLACK_CLIENT_ID", "123.456")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "")
    with pytest.raises(HTTPException) as exc:
        get_oauth_credentials("slack")
    assert exc.value.status_code == 422

    monkeypatch.setenv("SLACK_CLIENT_SECRET", "s3cret")
    assert get_oauth_credentials("slack") == {"client_id": "123.456", "client_secret": "s3cret"}


class TestIdentity:
    def test_tile_subtitle_is_the_email_never_the_composite_identity(self):
        assert connection_display_name({"account_email": "ann@acme.com:T1", "account_name": "Acme"}, "slack") == "ann@acme.com"
        assert connection_display_name({"account_email": "U1:T1", "account_name": "Acme"}, "slack") == "Acme"


class _FakeVault:
    def load(self, engine, name):
        return {"auth_type": "oauth", "_method": "mcp", "access_token": "xoxp-at", "refresh_token": "xoxe-rt"}


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


SETTINGS = dict(_env_file=None, SLACK_CLIENT_ID="123.456", SLACK_CLIENT_SECRET="s3cret")


class TestRevoke:
    def test_revoke_sends_the_access_token_as_bearer_to_auth_revoke(self, monkeypatch):
        monkeypatch.setattr(google_module, "_vault_for", lambda *a, **k: _FakeVault())
        sent = []

        def fake_urlopen(request, timeout=10):
            sent.append(request)
            return _FakeResponse({"ok": True, "revoked": True})

        monkeypatch.setattr(google_module, "urlopen", fake_urlopen)

        OAuthService().revoke("slack", "acme", ConnectorSettings(), OAuthSettings(**SETTINGS))

        assert [r.full_url for r in sent] == ["https://slack.com/api/auth.revoke"]
        assert sent[0].get_header("Authorization") == "Bearer xoxp-at"

    def test_an_ok_false_revoke_is_logged_and_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setattr(google_module, "_vault_for", lambda *a, **k: _FakeVault())
        monkeypatch.setattr(
            google_module, "urlopen", lambda request, timeout=10: _FakeResponse({"ok": False, "error": "token_revoked"}),
        )

        with caplog.at_level(logging.WARNING, logger="cowork.oauth"):
            OAuthService().revoke("slack", "acme", ConnectorSettings(), OAuthSettings(**SETTINGS))

        assert any("Could not revoke" in r.message and "token_revoked" in r.message for r in caplog.records)

    def test_a_bring_your_own_app_connection_is_not_revoked_at_slack(self, monkeypatch):
        class _OwnAppVault:
            def load(self, engine, name):
                return {"auth_type": "oauth", "_method": "oauth", "access_token": "xoxb-shared"}

        monkeypatch.setattr(google_module, "_vault_for", lambda *a, **k: _OwnAppVault())
        sent = []
        monkeypatch.setattr(
            google_module, "urlopen", lambda request, timeout=10: sent.append(request) or _FakeResponse({"ok": True}),
        )

        OAuthService().revoke("slack", "acme", ConnectorSettings(), OAuthSettings(**SETTINGS))

        assert sent == []


class TestCodeProjects:
    @pytest.fixture
    def protected(self, sharepoint_stores, monkeypatch):
        monkeypatch.setattr(app_secret, "_secret", "ab" * 32)
        app_held._entries.clear()
        yield
        app_held._entries.clear()

    def test_an_mcp_slack_connection_reads_a_thread_with_its_held_token(self, protected, slack_off):
        root = own_vault_root(LOCAL_SCOPE)
        name = persist_connection(
            "slack", "mcp", "",
            {"auth_type": "oauth", "access_token": "xoxp-held", "account_email": "ann@acme.com:T1"},
            vault=app_held.AppHeldVault(source=own_vault_for_scope(LOCAL_SCOPE), root=root),
            scope=LOCAL_SCOPE,
        )
        # The token is held in server memory, not in the vault file.
        assert "access_token" not in own_vault_for_scope(LOCAL_SCOPE).read_record("slack", name)["fields"]

        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"ok": True, "messages": [{"text": "Ship it"}]})

        integration = DeveloperIntegrationService(LOCAL_SCOPE, transport=httpx.MockTransport(handler))
        try:
            context = integration.read(None, SourceContextRequest(
                provider="slack", kind="conversation",
                url="https://acme.slack.com/archives/C123/p1234567890123456",
            ))
        finally:
            integration.close()

        assert context.body == "Ship it"
        assert context.connection_name == name
        assert seen[0].headers["Authorization"] == "Bearer xoxp-held"
