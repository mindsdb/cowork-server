"""_fetch_userinfo_linear: workspace-aware identity for Linear connections.

ENG-2188: a Linear account can belong to several workspaces under the same
email (unlike Google, where an account is an email), so the workspace id is
folded into the returned `email` for dedup, and the workspace name - not the
connecting person's own name - is returned as `name`, the same convention
`_fetch_userinfo_supabase` uses for its own per-organization identity.
"""

from __future__ import annotations

import json

import pytest

from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import _fetch_userinfo_linear


class _FakeJsonResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture(autouse=True)
def _silence_diagnostic_log(monkeypatch):
    # TEMP (ENG-2188): _fetch_userinfo_linear logs the raw GraphQL response
    # for live-verification purposes; keep test output clean.
    monkeypatch.setattr(google_module._log, "warning", lambda *a, **k: None)


def _stub_response(payload: dict, monkeypatch):
    monkeypatch.setattr(google_module, "urlopen", lambda request, timeout=20: _FakeJsonResponse(payload))


def test_workspace_name_used_over_person_name(monkeypatch):
    _stub_response(
        {
            "data": {
                "viewer": {"email": "user@example.com", "name": "User Name"},
                "organization": {"id": "org-123", "name": "Acme Workspace"},
            }
        },
        monkeypatch,
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com:org-123", "name": "Acme Workspace"}


def test_second_workspace_gets_a_distinct_identity(monkeypatch):
    _stub_response(
        {
            "data": {
                "viewer": {"email": "user@example.com", "name": "User Name"},
                "organization": {"id": "org-456", "name": "Other Workspace"},
            }
        },
        monkeypatch,
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity["email"] == "user@example.com:org-456"
    assert identity["email"] != "user@example.com:org-123"


def test_falls_back_to_person_name_when_organization_missing(monkeypatch):
    _stub_response({"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}}, monkeypatch)

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}
