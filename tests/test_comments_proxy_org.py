"""Org tenancy: the comments proxy authenticates upstream as the caller.

Desktop attaches the user's stored MindsHub key. An org pod has no such
settings, so the shared resolver produced an empty key, the proxy sent no
Authorization at all, and the gateway answered its own 401 - which the review
sidebar renders as "Session expired". These pin the org branch: the caller's own
bearer, this deployment's own inference host, and a refusal rather than an
anonymous upstream request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cowork.common.settings.app_settings as app_settings
import cowork.services.comments_proxy as cp

ORG_BASE = "https://api.staging.mindshub.ai/v1"
JWT = "eyJhbGciOiJSUzI1NiJ9.jwt-payload.signature"


def _request(headers: dict[str, str] | None = None) -> SimpleNamespace:
    """Enough of a Request for the resolver: it reads headers and nothing else."""
    return SimpleNamespace(headers=headers or {})


def test_org_base_prefers_the_explicit_operator_url(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_MINDS_BASE_URL", ORG_BASE)
    assert cp._org_inference_base() == ORG_BASE


def test_org_base_derives_from_the_deployment_host(monkeypatch):
    # A per-PR namespace has its own inference AND its own auth database, so the
    # base must follow default_turn_minds_api_host rather than the ENV slug.
    monkeypatch.setattr(
        app_settings, "TurnQueueSettings", lambda: SimpleNamespace(minds_base_url="")
    )
    monkeypatch.setattr(
        app_settings, "default_turn_minds_api_host", lambda: "https://api-pr-42.dev.mindshub.ai"
    )
    assert cp._org_inference_base() == "https://api-pr-42.dev.mindshub.ai/v1"


def test_org_resolves_to_the_operator_base_and_the_callers_bearer(monkeypatch):
    monkeypatch.setattr(cp, "_org_mode", lambda: True)
    monkeypatch.setenv("COWORK_TURN_MINDS_BASE_URL", ORG_BASE)
    base, credential = cp.resolve_comments_upstream(
        _request({"Authorization": f"Bearer {JWT}"})
    )
    assert base == ORG_BASE
    # Bare token, no scheme: _forward_headers adds "Bearer " itself.
    assert credential == JWT


def test_org_never_reads_tenant_settable_user_settings(monkeypatch):
    # An org admin controls openai_base_url/minds_url. Reading either would let
    # them point a member's credential at a host of their choosing, which is
    # exactly what caller_bearer's contract forbids. Assert the org branch does
    # not CALL them, not merely that their values failed to reach the result.
    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("org mode must not read tenant-settable user settings")

    monkeypatch.setattr(cp, "_org_mode", lambda: True)
    monkeypatch.setenv("COWORK_TURN_MINDS_BASE_URL", ORG_BASE)
    monkeypatch.setattr(cp, "get_user_settings", _must_not_be_called)
    monkeypatch.setattr(cp, "provider_api_key", _must_not_be_called)

    base, credential = cp.resolve_comments_upstream(
        _request({"Authorization": f"Bearer {JWT}"})
    )
    assert base == ORG_BASE
    assert credential == JWT


def test_org_without_authorization_resolves_to_an_empty_credential(monkeypatch):
    monkeypatch.setattr(cp, "_org_mode", lambda: True)
    monkeypatch.setenv("COWORK_TURN_MINDS_BASE_URL", ORG_BASE)
    base, credential = cp.resolve_comments_upstream(_request())
    assert base == ORG_BASE
    assert credential == ""


def test_desktop_never_forwards_the_incoming_authorization(monkeypatch):
    # Electron's main process overwrites Authorization on every loopback request
    # with the sidecar's own token, so forwarding it would leak OUR credential.
    monkeypatch.setattr(cp, "_org_mode", lambda: False)
    monkeypatch.setattr(
        cp,
        "resolve_inference_endpoint",
        lambda settings=None: ("https://api.mindshub.ai/v1", "mdb_userkey"),
    )
    assert cp.resolve_comments_upstream(_request({"Authorization": f"Bearer {JWT}"})) == (
        "https://api.mindshub.ai/v1",
        "mdb_userkey",
    )
