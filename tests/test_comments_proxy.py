"""cowork-server artifact-comments proxy: REST forward + SSE passthrough."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import cowork.services.comments_proxy as cp
from cowork.server import app

client = TestClient(app)
BASE = "https://api.dev.mindshub.ai/v1"
KEY = "mdb_testkey"


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
    assert call["headers"]["Accept"] == "text/event-stream"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_503_when_endpoint_unconfigured(monkeypatch):
    monkeypatch.setattr(cp, "resolve_inference_endpoint", lambda settings=None: ("", ""))
    monkeypatch.setattr(cp, "get_proxy_client", lambda: _FakeClient())
    r = client.get("/api/v1/artifact-comments/alice/rep123/threads")
    assert r.status_code == 503
