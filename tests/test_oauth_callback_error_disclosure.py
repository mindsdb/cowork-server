"""Regression: the OAuth callback page must not leak internal exception text.

A bare `except Exception` in GoogleOAuthService.callback used to interpolate
str(exc) into the HTML returned to the user's browser (py/stack-trace-exposure).
The detail belongs in the server log and the internal outcome store only; the
page shown to the user stays generic.
"""

import html as html_lib
import json
import logging
import re
from typing import Any

import pytest

from cowork.common.record_files import MAX_RECORD_BYTES
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors.oauth import google
from cowork.services.connectors.oauth.google import OAuthService
from cowork.services.connectors.persist import own_vault_for_scope
from cowork.services.connectors.sealed_vault import SEALED_FIELD


class _FakeStore:
    def __init__(self) -> None:
        self.pending: dict[str, dict[str, Any]] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.cleared: dict[str, str] = {}

    def get_pending(self, service: str) -> dict[str, Any] | None:
        return self.pending.get(service)

    def clear_pending(self, service: str, *, error: str = "") -> None:
        self.pending.pop(service, None)
        self.cleared[service] = error

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

    html = svc.callback("gmail", code="auth-code", state="s-123", error="", settings=OAuthSettings(), scope=LOCAL_SCOPE)

    # The browser page is generic — no exception text.
    assert secret not in html
    # But the detail is still captured internally for debugging.
    assert store.outcomes["s-123"]["status"] == "error"
    assert secret in store.outcomes["s-123"]["error"]


FLAG = "flag-value-6150"
_RESERVED = re.compile(
    r"The saved connection gmail/\S+ can't be saved: the field name _sealed is reserved for the sealed mark\. "
    r"Rename the field and save again\."
)
_TOO_LARGE = re.compile(
    r"The saved connection gmail/\S+ is too large to save: sealed, its record file would be \d+ bytes, and the "
    rf"limit is {MAX_RECORD_BYTES}\."
)


@pytest.mark.parametrize(
    ("access_token", "extra_fields", "reason"),
    [("at-1", {SEALED_FIELD: FLAG}, _RESERVED), (FLAG + "x" * MAX_RECORD_BYTES, {}, _TOO_LARGE)],
    ids=["field-named-like-the-mark", "too-large-once-sealed"],
)
def test_a_save_the_sealed_vault_refuses_shows_its_reason_and_logs_no_error(
    monkeypatch, caplog, key_file, sharepoint_stores, access_token, extra_fields, reason
):
    """With a key file, SealedDataVault.save refuses a field named like the sealed mark and a
    record too large to read back once sealed. A new sign-in from the same form brings the same
    values, so the page gives the vault's reason, which names the connection and never a value,
    not "try again". It refuses the person's input, so nothing is logged as an error."""
    store = _FakeStore()
    store.pending["gmail"] = {
        "state": "s-123",
        "clientId": "cid",
        "clientSecret": "csecret",
        "redirectUri": "http://127.0.0.1/cb",
        "verifier": "v",
        "startedAt": "",
        "extraFields": extra_fields,
    }
    svc = OAuthService()
    monkeypatch.setattr(svc, "_store", lambda settings: store)
    monkeypatch.setattr(svc, "_exchange_code", lambda **kwargs: {"access_token": access_token, "expires_in": 3600})
    monkeypatch.setattr(google, "_USERINFO_FETCHERS", {"gmail": lambda token: {"email": "person@example.com"}})

    with caplog.at_level(logging.DEBUG):
        page = svc.callback(
            "gmail", code="auth-code", state="s-123", error="", settings=OAuthSettings(_env_file=None), scope=LOCAL_SCOPE
        )

    assert [r for r in caplog.records if r.name == google._log.name and r.levelno >= logging.ERROR] == []
    outcome = store.outcomes["s-123"]
    assert outcome["status"] == "error" and reason.fullmatch(outcome["error"]), outcome["error"]
    assert html_lib.escape(outcome["error"]) in page and "try again" not in page
    assert store.cleared == {"gmail": outcome["error"]} and store.pending == {}
    assert FLAG not in page and FLAG not in json.dumps(outcome) and FLAG not in caplog.text
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []
