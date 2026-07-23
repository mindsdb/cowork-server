"""cowork-server artifact-comments proxy: REST forward + SSE passthrough."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import cowork.services.comments_proxy as cp
from cowork.server import app

client = TestClient(app)
BASE = "https://api.staging.mindshub.ai/v1"
KEY = "mdb_testkey"

# Real resolver, captured before the autouse fixture below monkeypatches it away.
_resolve_inference_endpoint = cp.resolve_inference_endpoint


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    monkeypatch.setattr(cp, "resolve_inference_endpoint", lambda settings=None: (BASE, KEY))


class _FakeClient:
    """Minimal stand-in for the shared httpx client."""

    def __init__(self, response=None, stream_upstream=None):
        self._response = response
        self._stream_upstream = stream_upstream
        self.calls = {}

    async def request(self, method, url, headers=None, content=None):
        self.calls["rest"] = {"method": method, "url": url, "headers": headers, "content": content}
        return self._response

    def build_request(self, method, url, headers=None, timeout=None):
        self.calls["stream"] = {"method": method, "url": url, "headers": headers, "timeout": timeout}
        return ("req", url)

    async def send(self, req, stream=False):
        return self._stream_upstream


class _FakeUpstream:
    def __init__(self, status_code, chunks, headers=None):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})

    async def aiter_raw(self):
        for c in [b"event: thread.created\ndata: {\"id\":\"t1\",\"version\":1}\n\n",
                  b": keepalive\n\n"]:
            yield c

    async def aclose(self):
        pass


def test_rest_forwards_with_auth_and_returns_response(monkeypatch):
    resp = httpx.Response(200, json={"threads": []}, headers={"content-type": "application/json"})
    fake = _FakeClient(response=resp)
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.get("/api/v1/artifact-comments/alice/rep123/threads?status=open")
    assert r.status_code == 200
    assert r.json() == {"threads": []}
    call = fake.calls["rest"]
    assert call["method"] == "GET"
    assert call["url"] == f"{BASE}/artifact-comments/alice/rep123/threads?status=open"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_rest_forwards_post_body(monkeypatch):
    resp = httpx.Response(200, json={"id": "t1", "type": "thread.created"})
    fake = _FakeClient(response=resp)
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.post(
        "/api/v1/artifact-comments/alice/rep123/threads",
        json={"selector": "#c", "text": "hi"},
    )
    assert r.status_code == 200
    call = fake.calls["rest"]
    assert call["url"] == f"{BASE}/artifact-comments/alice/rep123/threads"
    assert b'"text"' in call["content"]


def test_stream_passthrough_is_event_stream(monkeypatch):
    fake = _FakeClient(stream_upstream=_FakeUpstream(200, None))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    with client.stream("GET", "/api/v1/artifact-comments/alice/rep123/stream?since=x") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        assert r.headers["cache-control"] == "no-store"
        body = b"".join(r.iter_raw())
    assert b"thread.created" in body
    call = fake.calls["stream"]
    assert call["url"] == f"{BASE}/artifact-comments/alice/rep123/stream?since=x"
    # Includes */* so the auth-gated ingress's DRF auth_request negotiates JSON (not 406).
    assert call["headers"]["Accept"] == "text/event-stream, */*"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
def test_rest_forwards_patch_and_delete(monkeypatch, method):
    resp = httpx.Response(200, json={"id": "t1", "type": "thread.updated"})
    fake = _FakeClient(response=resp)
    monkeypatch.setattr(cp, "get_proxy_client", lambda: fake)

    r = client.request(
        method,
        "/api/v1/artifact-comments/alice/rep123/threads/tid-1",
        content=b"{}" if method == "PATCH" else None,
    )
    assert r.status_code == 200
    call = fake.calls["rest"]
    assert call["method"] == method
    assert call["url"] == f"{BASE}/artifact-comments/alice/rep123/threads/tid-1"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_503_when_endpoint_unconfigured(monkeypatch):
    monkeypatch.setattr(cp, "resolve_inference_endpoint", lambda settings=None: ("", ""))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: _FakeClient())
    r = client.get("/api/v1/artifact-comments/alice/rep123/threads")
    assert r.status_code == 503


class _FakeSettings:
    def __init__(self, openai_base_url, minds_url="https://api.mindshub.ai"):
        self.openai_base_url = openai_base_url
        self.minds_url = minds_url
        self.minds_api_key = "mdb_prodkey"


@pytest.mark.parametrize(
    "openai_base_url,expected_base,expected_key",
    [
        # Prod MindsHub OpenAI-compatible endpoint.
        ("https://api.mindshub.ai/v1", "https://api.mindshub.ai/v1", "mdb_oaikey"),
        # Hyphenated dev/staging subdomain — must still be recognised as MindsHub.
        ("https://api.staging.mindshub.ai/v1", "https://api.staging.mindshub.ai/v1", "mdb_oaikey"),
        # Non-MindsHub / unset custom endpoint → fall back to prod minds_url.
        ("https://api.openai.com/v1", "https://api.mindshub.ai", "mdb_prodkey"),
        ("", "https://api.mindshub.ai", "mdb_prodkey"),
    ],
)
def test_resolve_inference_endpoint_host_matching(
    monkeypatch, openai_base_url, expected_base, expected_key
):
    monkeypatch.setattr(cp, "provider_api_key", lambda settings, provider: "mdb_oaikey")
    base, key = _resolve_inference_endpoint(_FakeSettings(openai_base_url))
    assert base == expected_base
    assert key == expected_key


# --- path-segment normalization (traversal guard) -------------------------


def test_upstream_url_builds_expected_path():
    url = cp._upstream_url(BASE, "alice", "rep123", "threads/t1/replies", "status=open")
    assert url == f"{BASE}/artifact-comments/alice/rep123/threads/t1/replies?status=open"


@pytest.mark.parametrize(
    "user_dir,report_id,subpath",
    [
        ("..", "rep123", "threads"),          # climb via user_dir
        ("alice", "..", "threads"),           # climb via report_id
        ("alice", "rep123", "../../v1/models"),  # climb via subpath
        ("alice", "rep123", "a/../../b"),     # embedded dot-segment
        ("alice", "rep123", "."),             # bare dot
        ("alice", "rep123", "a//b"),          # empty segment
    ],
)
def test_upstream_url_rejects_traversal(user_dir, report_id, subpath):
    with pytest.raises(ValueError):
        cp._upstream_url(BASE, user_dir, report_id, subpath, "")


def test_upstream_url_percent_encodes_segments():
    # A '?' inside a segment must not open a query string on the upstream.
    url = cp._upstream_url(BASE, "alice", "rep123", "th?x", "")
    assert url == f"{BASE}/artifact-comments/alice/rep123/th%3Fx"
