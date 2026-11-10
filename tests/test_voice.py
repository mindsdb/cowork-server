"""Voice calls (ENG-4138, prototype): the server starts a GPT-Live session through
MindsHub Inference as the caller, with a fixed prompt and client delegation."""
from __future__ import annotations

import json

import httpx
import pytest

import cowork.services.voice as voice
from cowork.server import create_app

BEARER = "Bearer user-jwt"


@pytest.fixture(autouse=True)
def _fresh_status_cache():
    voice._status_cache.clear()
    yield
    voice._status_cache.clear()


@pytest.fixture
def inference(monkeypatch):
    """MindsHub Inference: answers with `inference.answer`; records each request."""
    seen: list[httpx.Request] = []

    class Fake:
        answer = httpx.Response(201, json={"session": {"id": "live_1"}, "transport": {"type": "webrtc", "sdp": "answer"}})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return Fake.answer

    monkeypatch.setattr(voice, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(voice, "default_turn_minds_api_host", lambda: "https://api-pr-cowork-1.dev.mindshub.ai")
    Fake.seen = seen
    return Fake


async def _call(method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(
            method, f"/api/v1/voice{path}", headers={"X-MindsHub-Authorization": BEARER}, **kwargs
        )


async def test_starts_a_gpt_live_session_as_the_caller_with_the_fixed_session(inference):
    started = await _call("POST", "/session", json={"sdp": "offer-sdp"})

    assert started.status_code == 200, started.text
    assert started.json() == {"session": {"id": "live_1"}, "transport": {"type": "webrtc", "sdp": "answer"}}
    [request] = inference.seen
    assert str(request.url) == "https://api-pr-cowork-1.dev.mindshub.ai/v1/live/sessions"
    assert request.headers["Authorization"] == BEARER
    body = json.loads(request.content)
    assert body["session"] == {"model": "gpt-live-1", "instructions": voice.VOICE_PROMPT, "delegation": {"type": "client"}}
    assert body["transport"] == {"type": "webrtc", "sdp": "offer-sdp"}


@pytest.mark.parametrize(
    ("status", "error", "expected", "words"),
    [
        (404, {"code": "live_not_enabled", "message": "x"}, 409, "aren't available"),
        (429, {"code": "rate_limit_exceeded", "message": "x"}, 429, "Too many"),
        (401, {"code": "unauthorized", "message": "x"}, 401, "Couldn't start"),
        (502, {"code": "upstream_error", "message": "x"}, 424, "Couldn't start"),
    ],
)
async def test_a_refusal_says_why(inference, status, error, expected, words):
    inference.answer = httpx.Response(status, json={"error": error})

    refused = await _call("POST", "/session", json={"sdp": "offer-sdp"})

    assert refused.status_code == expected
    assert words in refused.json()["detail"]["message"]


async def test_status_asks_inference_once_whether_live_voice_is_on(inference):
    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request", "param": "transport.type"}})

    assert (await _call("GET", "/status")).json() == {"available": True}
    assert (await _call("GET", "/status")).json() == {"available": True}
    assert len(inference.seen) == 1
    assert json.loads(inference.seen[0].content) == {}


async def test_status_is_off_where_inference_does_not_serve_it(inference):
    inference.answer = httpx.Response(404, json={"error": {"code": "live_not_enabled"}})

    assert (await _call("GET", "/status")).json() == {"available": False}


async def test_a_caller_inference_refuses_is_not_cached_for_everyone(inference):
    inference.answer = httpx.Response(401, json={"error": {"code": "unauthorized"}})
    assert (await _call("GET", "/status")).json() == {"available": False}

    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request"}})
    assert (await _call("GET", "/status")).json() == {"available": True}
