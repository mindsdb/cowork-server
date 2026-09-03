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

import httpx
import pytest
from fastapi.testclient import TestClient

import cowork.api.v1.endpoints.comments as comments_router
import cowork.common.settings.app_settings as app_settings
import cowork.services.comments_proxy as cp
from cowork.server import app

client = TestClient(app)

ORG_BASE = "https://api.staging.mindshub.ai/v1"
JWT = "eyJhbGciOiJSUzI1NiJ9.jwt-payload.signature"
ARTIFACT = "artifact/721ce811-9a3e-4f2b-8c1d-2f7b6a5e4d3c"


def _request(headers: dict[str, str] | None = None) -> SimpleNamespace:
    """Enough of a Request for the resolver: it reads headers and nothing else."""
    return SimpleNamespace(headers=headers or {})


# --- the resolver ---------------------------------------------------------


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


# --- through the HTTP boundary --------------------------------------------


class _FakeClient:
    """Minimal stand-in for the shared httpx client; records the one call made."""

    def __init__(self, response=None, stream_upstream=None):
        self._response = response
        self._stream_upstream = stream_upstream
        self.calls = {}

    async def request(self, method, url, headers=None, content=None):
        self.calls["rest"] = {"method": method, "url": url, "headers": headers, "content": content}
        return self._response

    def build_request(self, method, url, headers=None, timeout=None):
        self.calls["stream"] = {"method": method, "url": url, "headers": headers}
        return ("req", url)

    async def send(self, req, stream=False):
        return self._stream_upstream


class _FakeUpstream:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    async def aiter_raw(self):
        yield b"event: thread.created\ndata: {\"id\":\"t1\"}\n\n"

    async def aclose(self):
        pass


@pytest.fixture
def org_mode(monkeypatch):
    """Turn on both org gates.

    There are two because the check is inlined per module (the repo has no shared
    helper for it): the endpoint's gate decides to proxy at all rather than serve
    the local journal, and the proxy's own gate decides which credential goes
    upstream. Patching only one leaves the request half in org mode.
    """
    monkeypatch.setattr(comments_router, "_org_mode", lambda: True)
    monkeypatch.setattr(cp, "_org_mode", lambda: True)
    monkeypatch.setenv("COWORK_TURN_MINDS_BASE_URL", ORG_BASE)


def test_org_rest_forwards_the_callers_bearer(org_mode, monkeypatch):
    fake = _FakeClient(response=httpx.Response(200, json={"threads": []}))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.get(
        f"/api/v1/artifact-comments/{ARTIFACT}/threads?status=all",
        headers={"Authorization": f"Bearer {JWT}"},
    )

    assert r.status_code == 200
    call = fake.calls["rest"]
    assert call["url"] == f"{ORG_BASE}/artifact-comments/{ARTIFACT}/threads?status=all"
    assert call["headers"]["Authorization"] == f"Bearer {JWT}"


def test_org_stream_forwards_the_callers_bearer(org_mode, monkeypatch):
    fake = _FakeClient(stream_upstream=_FakeUpstream())
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    with client.stream(
        "GET",
        f"/api/v1/artifact-comments/{ARTIFACT}/stream?since=x",
        headers={"Authorization": f"Bearer {JWT}"},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_raw())

    assert b"thread.created" in body
    call = fake.calls["stream"]
    assert call["url"] == f"{ORG_BASE}/artifact-comments/{ARTIFACT}/stream?since=x"
    assert call["headers"]["Authorization"] == f"Bearer {JWT}"
    # The DRF/406 workaround must survive the credential change.
    assert call["headers"]["Accept"] == "text/event-stream, */*"


def test_org_refuses_without_a_credential_and_never_calls_upstream(org_mode, monkeypatch):
    fake = _FakeClient(response=httpx.Response(200, json={}))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.get(f"/api/v1/artifact-comments/{ARTIFACT}/threads")

    assert r.status_code == 401
    # Sending it anyway would come back as the gateway's own 401 page, which the
    # renderer reads as an expired session.
    assert fake.calls == {}


def test_org_stream_refuses_without_a_credential(org_mode, monkeypatch):
    fake = _FakeClient(stream_upstream=_FakeUpstream())
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.get(f"/api/v1/artifact-comments/{ARTIFACT}/stream")

    assert r.status_code == 401
    assert fake.calls == {}


def test_org_sends_only_content_type_and_authorization(org_mode, monkeypatch):
    fake = _FakeClient(response=httpx.Response(200, json={"threads": []}))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    client.get(
        f"/api/v1/artifact-comments/{ARTIFACT}/threads",
        headers={
            "Authorization": f"Bearer {JWT}",
            "Cookie": "view_session=secret",
            "X-User-Id": "spoofed-user",
        },
    )

    # Built from scratch, so a client cannot smuggle a cookie or an identity
    # header into a service that trusts identity headers from its own ingress.
    assert set(fake.calls["rest"]["headers"]) == {"Content-Type", "Authorization"}
