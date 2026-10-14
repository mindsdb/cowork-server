"""Figma's in-browser OAuth connector: identity, credentials, spec shape and
disconnect.

Figma has no token revocation endpoint, so disconnect must clean up locally
without calling Figma, and its token endpoint authenticates the client with
HTTP Basic auth (the same `token_auth_style` Supabase uses).
"""

from __future__ import annotations

import json
from pathlib import Path

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import OAuthService, _fetch_userinfo_figma

_SPEC = Path(google_module.__file__).resolve().parents[1] / "specs" / "figma.json"


class _FakeJsonResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _stub_me(monkeypatch, payload: dict, captured: dict | None = None) -> None:
    def _fake_urlopen(request, timeout=20):
        if captured is not None:
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
        return _FakeJsonResponse(payload)

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)


def _builtin_method() -> dict:
    spec = json.loads(_SPEC.read_text())
    return next(m for m in spec["form"]["methods"] if m["id"] == "browser_oauth_builtin")


def test_identity_is_the_users_email(monkeypatch):
    captured: dict = {}
    _stub_me(monkeypatch, {"id": "123", "handle": "Ada", "email": "ada@example.com"}, captured)

    identity = _fetch_userinfo_figma("figma-token")

    assert identity == {"email": "ada@example.com", "name": "ada@example.com"}
    assert captured["url"] == "https://api.figma.com/v1/me"
    assert captured["headers"]["Authorization"] == "Bearer figma-token"


def test_identity_falls_back_to_handle_without_an_email(monkeypatch):
    _stub_me(monkeypatch, {"id": "123", "handle": "Ada", "email": ""})

    assert _fetch_userinfo_figma("figma-token") == {"email": "Ada", "name": "Ada"}


def test_credentials_resolve_from_figma_settings():
    settings = OAuthSettings(_env_file=None, FIGMA_CLIENT_ID="figma-cid", FIGMA_CLIENT_SECRET="figma-secret")

    assert OAuthService()._resolve_credentials("figma", settings) == ("figma-cid", "figma-secret")


def test_spec_uses_basic_auth_and_no_revoke():
    method = _builtin_method()
    oauth = method["oauth"]

    assert method["hidden"] is True
    assert oauth["token_url"] == "https://api.figma.com/v1/oauth/token"
    assert oauth["token_auth_style"] == "basic"
    assert oauth["supports_revoke"] is False
    assert "revoke_url" not in oauth
    assert oauth["service_id"] == "figma"
    assert oauth["redirect_port"] == 47295


def test_spec_requests_only_read_scopes():
    scopes = _builtin_method()["oauth"]["scopes"]

    assert "current_user:read" in scopes
    assert all(scope.endswith(":read") for scope in scopes)


class _FakeVault:
    def __init__(self, path):
        self._path = path

    def load(self, engine, name):
        return {"auth_type": "oauth", "access_token": "figma-access", "refresh_token": "figma-refresh"}


def test_disconnect_never_calls_figma(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)

    def _boom(request, timeout=10):
        raise AssertionError("Figma has no revoke endpoint; disconnect must stay local")

    monkeypatch.setattr(google_module, "urlopen", _boom)

    settings = OAuthSettings(_env_file=None, FIGMA_CLIENT_ID="figma-cid", FIGMA_CLIENT_SECRET="figma-secret")
    OAuthService().revoke("figma", "my-figma-conn", ConnectorSettings(), settings)
