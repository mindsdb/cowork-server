"""Regression: disconnecting a Supabase connection must actually revoke the
grant on Supabase's side, not just delete local credentials.

Supabase's OAuth revoke endpoint (`POST /v1/oauth/revoke`) doesn't fit the
generic RFC-7009 form-body pattern the other connectors use: it requires a
JSON body naming `client_id`, `client_secret`, and specifically the
`refresh_token` (revoking an access_token alone isn't supported and wouldn't
remove mindshub from the user's Supabase-side Authorized Apps list, since
that list reflects the underlying grant, not any one short-lived token).
Before the `_REVOKE_HANDLERS` entry was added, the spec's
`supports_revoke: false` meant `revoke()` returned immediately — disconnecting
never called Supabase at all, leaving the app listed as authorized
indefinitely (ENG bug: connecting a second Supabase org used to also fail
because of an unrelated loopback-callback issue; this covers the disconnect
side).
"""

import json

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import OAuthService


class _FakeVault:
    def __init__(self, path):
        self._path = path

    def load(self, engine, name):
        return {
            "auth_type": "oauth",
            "access_token": "sb-access-123",
            "refresh_token": "sb-refresh-456",
        }


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_revoke_calls_supabase_with_json_body_and_refresh_token(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)
    captured = {}

    def _fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        return _FakeResponse()

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    settings = OAuthSettings(_env_file=None, SUPABASE_CLIENT_ID="cid-123", SUPABASE_CLIENT_SECRET="csecret-456")
    svc = OAuthService()
    svc.revoke("supabase", "my-supabase-conn", ConnectorSettings(), settings)

    assert captured["url"] == "https://api.supabase.com/v1/oauth/revoke"
    assert captured["method"] == "POST"
    assert captured["headers"]["Content-type"] == "application/json"
    assert json.loads(captured["body"]) == {
        "client_id": "cid-123",
        "client_secret": "csecret-456",
        "refresh_token": "sb-refresh-456",
    }


def test_revoke_skips_network_call_without_oauth_settings(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)

    def _boom(request, timeout=10):
        raise AssertionError("must not call Supabase without credentials to authenticate the request")

    monkeypatch.setattr(google_module, "urlopen", _boom)

    svc = OAuthService()
    # Should not raise, just log and return.
    svc.revoke("supabase", "my-supabase-conn", ConnectorSettings(), None)
