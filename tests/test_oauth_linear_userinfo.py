"""_fetch_userinfo_linear / _fetch_linear_workspace: workspace-aware identity
for Linear connections.

ENG-2188: a Linear account can belong to several workspaces under the same
email (unlike Google, where an account is an email), so the workspace id is
folded into the returned `email` for dedup, and the workspace name - not the
connecting person's own name - is returned as `name`, the same convention
`_fetch_userinfo_supabase` uses for its own per-organization identity.

The workspace lookup is a deliberately separate, best-effort request from the
required viewer query (see `_fetch_linear_workspace`'s docstring) - these
tests exercise that split directly by returning different payloads per query.
"""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import _fetch_linear_workspace, _fetch_userinfo_linear


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
def _silence_failure_log(monkeypatch):
    # _fetch_linear_workspace logs a warning when the workspace lookup fails;
    # keep test output clean for the failure-path tests below.
    monkeypatch.setattr(google_module._log, "warning", lambda *a, **k: None)


def _is_organization_query(request) -> bool:
    return "organization" in json.loads(request.data.decode("utf-8"))["query"]


def _stub_two_calls(monkeypatch, *, viewer=None, organization=None, organization_raises=None):
    """Route a fake urlopen by which query the request carries."""

    def _fake_urlopen(request, timeout=20):
        if _is_organization_query(request):
            if organization_raises is not None:
                raise organization_raises
            return _FakeJsonResponse(organization)
        return _FakeJsonResponse(viewer)

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)


def test_workspace_name_used_over_person_name(monkeypatch):
    _stub_two_calls(
        monkeypatch,
        viewer={"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}},
        organization={"data": {"organization": {"id": "org-123", "name": "Acme Workspace"}}},
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com:org-123", "name": "Acme Workspace"}


def test_second_workspace_gets_a_distinct_identity(monkeypatch):
    _stub_two_calls(
        monkeypatch,
        viewer={"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}},
        organization={"data": {"organization": {"id": "org-456", "name": "Other Workspace"}}},
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity["email"] == "user@example.com:org-456"
    assert identity["email"] != "user@example.com:org-123"


def test_falls_back_to_person_name_when_organization_missing(monkeypatch):
    _stub_two_calls(
        monkeypatch,
        viewer={"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}},
        organization={"data": {}},
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_organization_graphql_errors_degrade_to_bare_email_instead_of_failing(monkeypatch):
    """The core resilience fix: an unverified/wrong `organization` field
    returning HTTP 200 + an `errors` array (which `_json_request` would not
    otherwise raise on) must not break the viewer identity fetch that a
    Linear connection's core functionality depends on."""
    _stub_two_calls(
        monkeypatch,
        viewer={"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}},
        organization={"errors": [{"message": "Cannot query field \"organization\" on type \"Query\"."}]},
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_organization_transport_failure_degrades_to_bare_email_instead_of_failing(monkeypatch):
    _stub_two_calls(
        monkeypatch,
        viewer={"data": {"viewer": {"email": "user@example.com", "name": "User Name"}}},
        organization_raises=HTTPError(
            "https://api.linear.app/graphql", 500, "Internal Server Error", {}, io.BytesIO(b"{}")
        ),
    )

    identity = _fetch_userinfo_linear("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_fetch_linear_workspace_returns_empty_pair_on_any_failure(monkeypatch):
    monkeypatch.setattr(
        google_module,
        "urlopen",
        lambda request, timeout=20: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _fetch_linear_workspace("access-token") == ("", "")


def test_fetch_linear_workspace_returns_id_and_name_on_success(monkeypatch):
    monkeypatch.setattr(
        google_module,
        "urlopen",
        lambda request, timeout=20: _FakeJsonResponse({"data": {"organization": {"id": "org-1", "name": "Acme"}}}),
    )

    assert _fetch_linear_workspace("access-token") == ("org-1", "Acme")
