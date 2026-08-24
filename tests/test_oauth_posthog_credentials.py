"""Regression: secret-less (public, PKCE-only) OAuth providers like PostHog
must not be treated as "credentials incomplete" just because client_secret
is empty — that's the correct, expected state for them, not a missing config.

Before the fix, three call sites checked `client_id and client_secret`
directly instead of accounting for `secret_attr is None`:
  - `start()`'s BYOK bypass silently discarded a caller-supplied client_id
    for a secret-less provider and fell back to server settings instead.
  - `callback()`'s cached-credentials branch could never fire for a
    secret-less provider, so it always re-resolved live settings instead of
    the client_id actually used to build the authorize URL.
Both now go through the shared `_credentials_complete` helper, same as
`_resolve_credentials` and `get_catalogue` already did.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from cowork.common.settings.app_settings import OAuthSettings
from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import OAuthService, _credentials_complete, _fetch_userinfo_posthog


@pytest.mark.parametrize(
    "client_id, client_secret, secret_attr, expected",
    [
        ("cid", "", None, True),  # secret-less provider, secret correctly empty
        ("", "", None, False),  # no client_id at all
        ("cid", "csecret", "x_client_secret", True),  # secretful provider, both present
        ("cid", "", "x_client_secret", False),  # secretful provider, secret missing
        ("", "csecret", "x_client_secret", False),  # secret present but no client_id
    ],
)
def test_credentials_complete(client_id, client_secret, secret_attr, expected):
    assert _credentials_complete(client_id, client_secret, secret_attr) is expected


def test_start_accepts_caller_supplied_client_id_for_secret_less_provider(tmp_path):
    # No posthog_client_id configured on settings — if the BYOK bypass fell
    # through to _resolve_credentials, this would raise HTTPException(400).
    settings = OAuthSettings(_env_file=None, state_path=str(tmp_path / "oauth_state.json"))
    svc = OAuthService()

    response = svc.start("posthog", settings, client_id="https://example.com/cimd.json", client_secret="")

    assert "client_id=https%3A%2F%2Fexample.com%2Fcimd.json" in response.auth_url


class _FakeStore:
    def __init__(self, pending: dict[str, dict[str, Any]]) -> None:
        self.pending = pending
        self.outcomes: dict[str, dict[str, Any]] = {}

    def get_pending(self, service: str) -> dict[str, Any] | None:
        return self.pending.get(service)

    def clear_pending(self, service: str, *, error: str = "") -> None:
        self.pending.pop(service, None)

    def set_outcome(self, state: str, outcome: dict[str, Any]) -> None:
        self.outcomes[state] = outcome

    def get_outcome(self, state: str) -> dict[str, Any] | None:
        return self.outcomes.get(state)


def test_callback_uses_cached_client_id_for_secret_less_provider(monkeypatch):
    store = _FakeStore(
        {
            "posthog": {
                "state": "s-456",
                "clientId": "https://example.com/cimd.json",
                "clientSecret": "",
                "redirectUri": "http://127.0.0.1/cb",
                "verifier": "v",
                "startedAt": "",
            }
        }
    )
    svc = OAuthService()
    monkeypatch.setattr(svc, "_store", lambda settings: store)

    captured = {}

    def _fake_exchange(**kwargs):
        captured["client_id"] = kwargs["client_id"]
        return {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}

    monkeypatch.setattr(svc, "_exchange_code", _fake_exchange)
    monkeypatch.setattr(
        "cowork.services.connectors.oauth.google._USERINFO_FETCHERS",
        {"posthog": lambda access_token: {"email": "a@b.com", "name": "A B"}},
    )
    monkeypatch.setattr(
        "cowork.services.connectors.oauth.google.persist_connection",
        lambda engine, method, name, fields: "posthog-conn",
    )

    # `_resolve_credentials` must not be reached — if the cached branch
    # weren't taken, this would raise (no posthog_client_id on settings)
    # and the connection would fail with a "not configured" page instead.
    def _boom(self, service, settings):
        raise AssertionError("cached client_id was not used — fell through to _resolve_credentials")

    monkeypatch.setattr(OAuthService, "_resolve_credentials", _boom)

    html = svc.callback("posthog", code="auth-code", state="s-456", error="", settings=OAuthSettings(_env_file=None))

    assert captured["client_id"] == "https://example.com/cimd.json"
    assert "connected" in html.lower()
    assert store.outcomes["s-456"]["status"] == "success"


class _FakeJsonResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_fetch_userinfo_posthog_falls_back_to_eu_host_on_us_failure(monkeypatch):
    def _fake_urlopen(request, timeout=20):
        if urlparse(request.full_url).hostname == "us.posthog.com":
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)
        assert urlparse(request.full_url).hostname == "eu.posthog.com"
        return _FakeJsonResponse({"email": "eu-user@example.com", "first_name": "EU", "last_name": "User"})

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    identity = _fetch_userinfo_posthog("eu-issued-token")

    assert identity == {"email": "eu-user@example.com", "name": "EU User"}


def test_fetch_userinfo_posthog_uses_us_host_when_it_succeeds(monkeypatch):
    def _fake_urlopen(request, timeout=20):
        assert urlparse(request.full_url).hostname == "us.posthog.com"
        return _FakeJsonResponse({"email": "us-user@example.com", "first_name": "US", "last_name": "User"})

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)

    identity = _fetch_userinfo_posthog("us-issued-token")

    assert identity == {"email": "us-user@example.com", "name": "US User"}
