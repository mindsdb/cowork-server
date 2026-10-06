"""Regression: a Google-family OAuth connection with no `name` claim must not
default its user_label to the bare engine id.

None of Drive/Calendar/Ads/Analytics/Gmail's granted scopes include
profile/openid, so Google's userinfo response never carries a `name` — before
the fix, `default_label=account_name or None` passed `None` in that case, and
`persist_connection` fell back to the bare engine id ("gmail"), de-duplicated
with a trailing counter on a second account ("gmail 2"). That label survives
connectionIdentity()'s "title, again" filter on the frontend and leaks into
the tile subtitle next to the email (e.g. "gmail 2 · user@example.com" instead
of just the email). Falling back to account_email keeps the label identical
to the subtitle's own identity value, so the frontend's dedup collapses them
back to just the email.
"""

from __future__ import annotations

from typing import Any

import pytest

from cowork.common.settings.app_settings import OAuthSettings
from cowork.services.connectors.oauth.google import OAuthService


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


def _run_gmail_callback(monkeypatch, *, userinfo: dict[str, str]) -> dict[str, Any]:
    store = _FakeStore(
        {
            "gmail": {
                "state": "s-1",
                "clientId": "gmail-client-id",
                "clientSecret": "gmail-client-secret",
                "redirectUri": "http://127.0.0.1/cb",
                "verifier": "v",
                "startedAt": "",
            }
        }
    )
    svc = OAuthService()
    monkeypatch.setattr(svc, "_store", lambda settings: store)
    monkeypatch.setattr(
        svc,
        "_exchange_code",
        lambda **kwargs: {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600},
    )
    monkeypatch.setattr(
        "cowork.services.connectors.oauth.google._USERINFO_FETCHERS",
        {"gmail": lambda access_token: userinfo},
    )

    captured: dict[str, Any] = {}

    def _fake_persist(engine, method, name, fields, **kwargs):
        captured["default_label"] = kwargs.get("default_label")
        return "gmail-conn"

    monkeypatch.setattr("cowork.services.connectors.oauth.google.persist_connection", _fake_persist)

    html = svc.callback("gmail", code="auth-code", state="s-1", error="", settings=OAuthSettings(_env_file=None))
    assert "connected" in html.lower()
    return captured


def test_default_label_falls_back_to_email_when_google_has_no_name_claim(monkeypatch):
    captured = _run_gmail_callback(monkeypatch, userinfo={"email": "user@example.com"})

    assert captured["default_label"] == "user@example.com"


def test_default_label_prefers_name_when_google_does_return_one(monkeypatch):
    captured = _run_gmail_callback(
        monkeypatch, userinfo={"email": "user@example.com", "name": "Real Name"}
    )

    assert captured["default_label"] == "Real Name"


@pytest.mark.parametrize("userinfo", [{}, {"email": ""}])
def test_default_label_is_none_when_google_has_neither_name_nor_email(monkeypatch, userinfo):
    captured = _run_gmail_callback(monkeypatch, userinfo=userinfo)

    assert captured["default_label"] is None
