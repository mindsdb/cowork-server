"""_fetch_userinfo_posthog / _fetch_posthog_organization: organization-aware
identity for PostHog connections.

ENG-2240: a PostHog account can belong to several organizations under the
same email (unlike Google, where an account is an email), so the
organization id is folded into the returned `email` for dedup, and the
organization name - not the connecting person's own name - is returned as
`name`, the same convention `_fetch_userinfo_supabase`/`_fetch_userinfo_linear`
use for their own per-organization identity.

The organization lookup is a deliberately separate, best-effort request from
the required user query (see `_fetch_posthog_organization`'s docstring) -
these tests exercise that split directly by returning different payloads per
endpoint path.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from cowork.services.connectors.oauth import google as google_module
from cowork.services.connectors.oauth.google import _fetch_posthog_organization, _fetch_userinfo_posthog


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
    # _fetch_posthog_organization logs a warning when the org lookup fails;
    # keep test output clean for the failure-path tests below.
    monkeypatch.setattr(google_module._log, "warning", lambda *a, **k: None)


def _stub_us_calls(monkeypatch, *, user=None, organization=None, organization_raises=None):
    """Route a fake urlopen by endpoint path, both against the US host."""

    def _fake_urlopen(request, timeout=20):
        assert urlparse(request.full_url).hostname == "us.posthog.com"
        if request.full_url.endswith("/api/organizations/"):
            if organization_raises is not None:
                raise organization_raises
            return _FakeJsonResponse(organization)
        return _FakeJsonResponse(user)

    monkeypatch.setattr(google_module, "urlopen", _fake_urlopen)


def test_organization_name_used_over_person_name(monkeypatch):
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com", "first_name": "User", "last_name": "Name"},
        organization={"results": [{"id": "org-123", "name": "Acme Org"}]},
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity == {"email": "user@example.com:org-123", "name": "Acme Org"}


def test_second_organization_gets_a_distinct_identity(monkeypatch):
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com", "first_name": "User", "last_name": "Name"},
        organization={"results": [{"id": "org-456", "name": "Other Org"}]},
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity["email"] == "user@example.com:org-456"
    assert identity["email"] != "user@example.com:org-123"


def test_falls_back_to_person_name_when_no_organizations(monkeypatch):
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com", "first_name": "User", "last_name": "Name"},
        organization={"results": []},
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_falls_back_to_email_when_no_name_and_no_organization(monkeypatch):
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com"},
        organization={"results": []},
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity == {"email": "user@example.com", "name": "user@example.com"}


def test_organization_bad_shape_degrades_to_bare_email_instead_of_failing(monkeypatch):
    """The core resilience fix: an unexpected /api/organizations/ response
    shape must not break the user identity fetch a PostHog connection's core
    functionality depends on."""
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com", "first_name": "User", "last_name": "Name"},
        organization={"detail": "not the shape we expected"},
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_organization_transport_failure_degrades_to_bare_email_instead_of_failing(monkeypatch):
    _stub_us_calls(
        monkeypatch,
        user={"email": "user@example.com", "first_name": "User", "last_name": "Name"},
        organization_raises=HTTPError("https://us.posthog.com/api/organizations/", 500, "Internal Server Error", {}, None),
    )

    identity = _fetch_userinfo_posthog("access-token")

    assert identity == {"email": "user@example.com", "name": "User Name"}


def test_fetch_posthog_organization_returns_empty_pair_on_any_failure(monkeypatch):
    monkeypatch.setattr(
        google_module,
        "urlopen",
        lambda request, timeout=20: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _fetch_posthog_organization("access-token", api_host="https://us.posthog.com") == ("", "")


def test_fetch_posthog_organization_returns_id_and_name_on_success(monkeypatch):
    monkeypatch.setattr(
        google_module,
        "urlopen",
        lambda request, timeout=20: _FakeJsonResponse({"results": [{"id": "org-1", "name": "Acme"}]}),
    )

    assert _fetch_posthog_organization("access-token", api_host="https://us.posthog.com") == ("org-1", "Acme")


def test_fetch_posthog_organization_accepts_a_raw_list_response(monkeypatch):
    monkeypatch.setattr(
        google_module,
        "urlopen",
        lambda request, timeout=20: _FakeJsonResponse([{"id": "org-1", "name": "Acme"}]),
    )

    assert _fetch_posthog_organization("access-token", api_host="https://us.posthog.com") == ("org-1", "Acme")
