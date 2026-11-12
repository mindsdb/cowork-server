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
    instructions = body["session"].pop("instructions")
    assert body["session"] == {
        "model": "gpt-live-1",
        "delegation": {"type": "client"},
        "audio": {"output": {"voice": "cinder"}},
    }
    # The fixed prompt, then the person's date and time (UTC with no zone sent).
    assert instructions.startswith(voice.VOICE_PROMPT + "\n\nIt is now ")
    assert "(UTC) where the user is" in instructions
    assert body["transport"] == {"type": "webrtc", "sdp": "offer-sdp"}


async def test_the_call_knows_the_persons_date_and_time_and_what_memory_holds_about_them(inference, monkeypatch):
    import cowork.api.v1.endpoints.voice as endpoint

    monkeypatch.setattr(endpoint, "_person_memory", lambda scope: {"profile": "Jorge, founder; lives in SF.", "rules": "Always use metric."})
    started = await _call("POST", "/session", json={"sdp": "offer-sdp", "timezone": "America/Los_Angeles"})

    assert started.status_code == 200, started.text
    instructions = json.loads(inference.seen[0].content)["session"]["instructions"]
    assert "(America/Los_Angeles) where the user is" in instructions
    assert "What you know about the user:\nJorge, founder; lives in SF." in instructions
    assert "The user's standing preferences:\nAlways use metric." in instructions


def test_local_now_reads_the_persons_zone_or_falls_back_to_utc():
    from datetime import datetime, timezone

    at = datetime(2026, 11, 12, 4, 5, tzinfo=timezone.utc)
    assert voice.local_now("America/Los_Angeles", at) == "Wednesday, November 11, 2026, 8:05 PM (America/Los_Angeles)"
    assert voice.local_now("Not/AZone", at) == "Thursday, November 12, 2026, 4:05 AM (UTC)"
    assert voice.local_now(None, at).endswith("(UTC)")
    assert voice.voice_context(tz_name=None, memory={"profile": "x" * 5000}, now=at).count("x") == voice.MAX_MEMORY_CHARS


@pytest.mark.parametrize(("asked", "sent"), [("willow", "willow"), ("cedar", "cedar"), ("not-a-voice", "cinder")])
async def test_the_session_speaks_in_the_chosen_voice_or_the_default(inference, asked, sent):
    started = await _call("POST", "/session", json={"sdp": "offer-sdp", "voice": asked})

    assert started.status_code == 200, started.text
    [request] = inference.seen
    assert json.loads(request.content)["session"]["audio"] == {"output": {"voice": sent}}


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


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_a_busy_or_failing_inference_is_not_cached_for_everyone(inference, status):
    inference.answer = httpx.Response(status, json={"error": {"code": "x"}})
    assert (await _call("GET", "/status")).json() == {"available": False}

    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request"}})
    assert (await _call("GET", "/status")).json() == {"available": True}


async def test_an_unreachable_inference_is_not_cached(monkeypatch, inference):
    def down(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(voice, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(down)))
    assert (await _call("GET", "/status")).json() == {"available": False}
    assert voice._status_cache == {}


async def test_a_session_answer_that_isnt_json_is_a_bad_answer(inference):
    inference.answer = httpx.Response(201, content=b"<html>oops</html>")

    answered = await _call("POST", "/session", json={"sdp": "offer-sdp"})

    assert answered.status_code == 424
    assert answered.json()["detail"]["code"] == "bad_answer"
