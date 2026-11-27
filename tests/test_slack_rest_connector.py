"""Slack's REST sign-in: the same In-Browser Connect, with the token handed to
the agent for Slack's Web API. Slack grants the token every scope the app has,
so the connection always reads and writes."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors import app_held
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import OAuthService
from cowork.services.connectors.persist import own_vault_for_scope
from cowork.services.connectors.specs._registry import ConnectorSpecRegistry

SECRET = "ab" * 32
HEADERS = {app_secret.HEADER: SECRET}
BASE = "/api/v1/connectors/connections"


@pytest.fixture
def stores(sharepoint_stores, monkeypatch):
    monkeypatch.setenv("SLACK_CONNECTOR_ENABLED", "false")
    monkeypatch.setattr(app_secret, "_secret", SECRET)
    app_held._entries.clear()
    yield sharepoint_stores
    app_held._entries.clear()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _save() -> str:
    response = _client().post(f"{BASE}/save", json={
        "connector_id": "slack", "method": "browser_oauth_builtin",
        "values": {
            "access_token": "xoxp-1", "expires_at": "2099-01-01T00:00:00+00:00", "scope": "channels:read",
            "account_email": "ann@acme.com:T1", "account_name": "Acme",
        },
    }, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()["name"]


def _summary(name: str) -> dict:
    rows = [row for row in _client().get(f"{BASE}/", headers=HEADERS).json() if row["name"] == name]
    assert len(rows) == 1
    return rows[0]


def _rest_method():
    return {m.id: m for m in ConnectorSpecRegistry().get_connector("slack").form.methods}["browser_oauth_builtin"]


class TestSpec:
    def test_it_asks_for_reading_searching_and_posting(self):
        scopes = _rest_method().oauth.scopes
        assert {"search:read", "channels:history", "chat:write"} <= set(scopes)

    def test_canvases_and_lists_are_left_out(self):
        assert not [s for s in _rest_method().oauth.scopes if s.split(":")[0] in ("canvases", "lists")]

    def test_it_signs_in_like_the_mcp_method(self):
        methods = {m.id: m for m in ConnectorSpecRegistry().get_connector("slack").form.methods}
        rest, mcp = methods["browser_oauth_builtin"].oauth, methods["mcp"].oauth
        assert (rest.auth_url, rest.token_url, rest.redirect_host, rest.redirect_port, rest.service_id) == (
            mcp.auth_url, mcp.token_url, mcp.redirect_host, mcp.redirect_port, mcp.service_id
        )

    def test_it_comes_first_for_clients_that_take_the_first_sign_in(self):
        methods = ConnectorSpecRegistry().get_connector("slack").form.methods
        assert [m.id for m in methods if m.id in ("browser_oauth_builtin", "mcp")][0] == "browser_oauth_builtin"

    def test_usage_notes_cover_the_limits_and_asking_before_posting(self):
        notes = ConnectorSpecRegistry().usage_notes_for(["slack"])["slack"]
        for text in ("DS_SLACK_<CONNECTION>__ACCESS_TOKEN", "search.messages", "one request a minute",
                     "Retry-After", "url_private_download", "wait for their OK", "users.info"):
            assert text in notes


class TestConnection:
    def test_a_new_sign_in_saves_with_the_switch_off(self, stores):
        name = _save()
        fields = own_vault_for_scope(LOCAL_SCOPE).load("slack", name)
        assert fields["_method"] == "browser_oauth_builtin"
        # Not an MCP connection: the token stays in the file, for the agent's environment.
        assert fields["access_token"] == "xoxp-1"

    def test_the_tile_shows_read_and_write(self, stores):
        name = _save()
        assert _summary(name)["access_mode"] == "write"

    @pytest.mark.parametrize("mode", ["read", "write", "none"])
    def test_its_mode_cannot_be_changed(self, stores, mode):
        name = _save()
        response = _client().patch(f"{BASE}/slack/{name}/access-mode", json={"access_mode": mode}, headers=HEADERS)
        assert response.status_code == 409
        assert "_access_mode" not in own_vault_for_scope(LOCAL_SCOPE).load("slack", name)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps({"ok": True, "revoked": True}).encode()


def test_disconnect_revokes_a_rest_connection_at_slack(monkeypatch):
    class _Vault:
        def load(self, engine, name):
            return {"auth_type": "oauth", "_method": "browser_oauth_builtin", "access_token": "xoxp-rest"}

    monkeypatch.setattr(google_module, "_vault_for", lambda *a, **k: _Vault())
    sent = []
    monkeypatch.setattr(google_module, "urlopen", lambda request, timeout=10: sent.append(request) or _FakeResponse())

    OAuthService().revoke(
        "slack", "acme", ConnectorSettings(),
        OAuthSettings(_env_file=None, SLACK_CLIENT_ID="123.456", SLACK_CLIENT_SECRET="s3cret"),
    )

    assert [r.full_url for r in sent] == ["https://slack.com/api/auth.revoke"]
    assert sent[0].get_header("Authorization") == "Bearer xoxp-rest"
