"""Regression: the OAuth callback page must not leak internal exception text.

A bare `except Exception` in GoogleOAuthService.callback used to interpolate
str(exc) into the HTML returned to the user's browser (py/stack-trace-exposure).
The detail belongs in the server log and the internal outcome store only; the
page shown to the user stays generic.
"""

from typing import Any

from cowork.common.settings.app_settings import OAuthSettings
from cowork.services.connectors.oauth.google import OAuthService


class _FakeStore:
    def __init__(self) -> None:
        self.pending: dict[str, dict[str, Any]] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}

    def get_pending(self, service: str) -> dict[str, Any] | None:
        return self.pending.get(service)

    def clear_pending(self, service: str, *, error: str = "") -> None:
        self.pending.pop(service, None)

    def set_outcome(self, state: str, outcome: dict[str, Any]) -> None:
        self.outcomes[state] = outcome

    def get_outcome(self, state: str) -> dict[str, Any] | None:
        return self.outcomes.get(state)


def test_callback_exception_detail_is_not_disclosed_to_the_user(monkeypatch):
    secret = "SECRET-INTERNAL-boto3-NoCredentialsError-at-10.0.3.4"
    store = _FakeStore()
    # Credentials present so the flow skips _resolve_credentials and reaches
    # the token exchange, which is the code guarded by the bare `except`.
    store.pending["gmail"] = {
        "state": "s-123",
        "clientId": "cid",
        "clientSecret": "csecret",
        "redirectUri": "http://127.0.0.1/cb",
        "verifier": "v",
        "startedAt": "",
    }

    svc = OAuthService()
    monkeypatch.setattr(svc, "_store", lambda settings: store)
    # Token exchange blows up with a leaky message — the generic-except branch.
    def _boom(**kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(svc, "_exchange_code", _boom)

    html = svc.callback("gmail", code="auth-code", state="s-123", error="", settings=OAuthSettings())

    # The browser page is generic — no exception text.
    assert secret not in html
    # But the detail is still captured internally for debugging.
    assert store.outcomes["s-123"]["status"] == "error"
    assert secret in store.outcomes["s-123"]["error"]
