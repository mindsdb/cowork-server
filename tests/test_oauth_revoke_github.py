"""Regression: disconnecting a GitHub connection in the app must actually
revoke the grant on GitHub's side, not just delete local credentials.

GitHub classic OAuth Apps don't support the generic RFC-7009 POST revoke_url
pattern the other connectors (Google, Linear) use. The real revoke call is
`DELETE https://api.github.com/applications/{client_id}/grant`, authenticated
with HTTP Basic auth (client_id:client_secret) and a JSON body naming the
token. Before the `_REVOKE_HANDLERS` dispatch was added, `revoke()` treated
GitHub's `supports_revoke: false` as "not supported" and returned immediately
— disconnecting never called GitHub at all, leaving the app listed under the
user's Authorized OAuth Apps indefinitely.
"""

import base64
import json

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import OAuthService


class _FakeVault:
    def __init__(self, path):
        self._path = path

    def load(self, engine, name):
        return {"auth_type": "oauth", "access_token": "gh-token-123"}


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_revoke_calls_github_grant_endpoint_with_basic_auth(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)
    captured = {}

    def _fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        return _FakeResponse()

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    settings = OAuthSettings(_env_file=None, GITHUB_CLIENT_ID="cid-123", GITHUB_CLIENT_SECRET="csecret-456")
    svc = OAuthService()
    svc.revoke("github", "my-github-conn", ConnectorSettings(), settings)

    assert captured["url"] == "https://api.github.com/applications/cid-123/grant"
    assert captured["method"] == "DELETE"
    expected_auth = "Basic " + base64.b64encode(b"cid-123:csecret-456").decode("ascii")
    assert captured["headers"]["Authorization"] == expected_auth
    assert json.loads(captured["body"]) == {"access_token": "gh-token-123"}


def test_revoke_skips_network_call_without_oauth_settings(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)

    def _boom(request, timeout=10):
        raise AssertionError("must not call GitHub without credentials to authenticate the request")

    monkeypatch.setattr(google_module, "urlopen", _boom)

    svc = OAuthService()
    # Should not raise, just log and return.
    svc.revoke("github", "my-github-conn", ConnectorSettings(), None)


def test_revoke_still_uses_generic_path_for_providers_without_a_custom_handler(monkeypatch):
    monkeypatch.setattr(google_module, "LocalDataVault", _FakeVault)
    captured = {}

    def _fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = request.data
        return _FakeResponse()

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    settings = OAuthSettings(google_drive_client_id="cid", google_drive_client_secret="csecret")
    svc = OAuthService()
    svc.revoke("google_drive", "my-drive-conn", ConnectorSettings(), settings)

    assert captured["url"] == "https://oauth2.googleapis.com/revoke"
    assert captured["method"] == "POST"
    assert captured["body"] == b"token=gh-token-123"
