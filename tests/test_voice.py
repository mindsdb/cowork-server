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


@pytest.fixture(autouse=True)
def zdr(monkeypatch):
    """ZDR off unless a test sets ``state.on``; ``state.asked`` holds each bearer it was read with."""
    from types import SimpleNamespace

    from cowork.api.v1.endpoints import voice as voice_ep

    from cowork.services.mindshub_zdr import ZdrState

    state = SimpleNamespace(on=False, asked=[])

    async def _zdr_state(*, bearer_token, org_id, user_id):
        state.asked.append(bearer_token)
        if isinstance(state.on, Exception):
            raise state.on
        if isinstance(state.on, ZdrState):
            return state.on
        return ZdrState.ON if state.on else ZdrState.OFF

    monkeypatch.setattr(voice_ep, "zdr_state", _zdr_state)
    return state


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


async def test_a_busy_file_store_starts_the_call_without_memory(inference, monkeypatch):
    import cowork.api.v1.endpoints.voice as endpoint
    from cowork.db.units import FilesystemBusy

    async def busy(fn, *, scope, deadline=True):
        raise FilesystemBusy("no slot")

    monkeypatch.setattr(endpoint, "run_efs", busy)
    started = await _call("POST", "/session", json={"sdp": "offer-sdp"})

    assert started.status_code == 200, started.text
    instructions = json.loads(inference.seen[0].content)["session"]["instructions"]
    assert "What you know about the user" not in instructions


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
        # Inference words an empty wallet and too many open sessions for the person.
        (402, {"code": "wallet_empty", "message": "Your organization's balance is empty."}, 402, "balance is empty"),
        (402, {"code": "wallet_empty"}, 402, "balance can't cover"),
        (429, {"code": "too_many_live_sessions", "message": "You have 3 voice calls open."}, 429, "3 voice calls open"),
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

    assert (await _call("GET", "/status")).json()["available"] is True
    assert (await _call("GET", "/status")).json()["available"] is True
    assert len(inference.seen) == 1
    assert json.loads(inference.seen[0].content) == {}


async def test_status_is_off_where_inference_does_not_serve_it(inference):
    inference.answer = httpx.Response(404, json={"error": {"code": "live_not_enabled"}})

    assert (await _call("GET", "/status")).json()["available"] is False


async def test_an_off_is_asked_again_sooner_than_an_on(monkeypatch, inference):
    now = [1000.0]
    monkeypatch.setattr(voice.time, "monotonic", lambda: now[0])
    inference.answer = httpx.Response(404, json={"error": {"code": "live_not_enabled"}})
    assert (await _call("GET", "/status")).json()["available"] is False

    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request"}})
    now[0] += voice._STATUS_OFF_TTL_S - 1
    assert (await _call("GET", "/status")).json()["available"] is False
    now[0] += 1
    assert (await _call("GET", "/status")).json()["available"] is True

    inference.answer = httpx.Response(404, json={"error": {"code": "live_not_enabled"}})
    now[0] += voice._STATUS_OFF_TTL_S
    assert (await _call("GET", "/status")).json()["available"] is True
    assert len(inference.seen) == 2


async def test_a_caller_inference_refuses_is_not_cached_for_everyone(inference):
    inference.answer = httpx.Response(401, json={"error": {"code": "unauthorized"}})
    assert (await _call("GET", "/status")).json()["available"] is False

    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request"}})
    assert (await _call("GET", "/status")).json()["available"] is True


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_a_busy_or_failing_inference_is_not_cached_for_everyone(inference, status):
    inference.answer = httpx.Response(status, json={"error": {"code": "x"}})
    assert (await _call("GET", "/status")).json()["available"] is False

    inference.answer = httpx.Response(400, json={"error": {"code": "invalid_request"}})
    assert (await _call("GET", "/status")).json()["available"] is True


async def test_an_unreachable_inference_is_not_cached(monkeypatch, inference):
    def down(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(voice, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(down)))
    assert (await _call("GET", "/status")).json()["available"] is False
    assert voice._status_cache == {}


async def test_a_session_answer_that_isnt_json_is_a_bad_answer(inference):
    inference.answer = httpx.Response(201, content=b"<html>oops</html>")

    answered = await _call("POST", "/session", json={"sdp": "offer-sdp"})

    assert answered.status_code == 424
    assert answered.json()["detail"]["code"] == "bad_answer"


async def test_status_lists_the_voices_a_call_can_speak_in(inference):
    status = (await _call("GET", "/status")).json()

    assert status["defaultVoice"] == voice.DEFAULT_VOICE
    assert status["voices"][0] == {"name": "cinder", "style": voice.VOICE_STYLES["cinder"]}
    assert [v["name"] for v in status["voices"]] == list(voice.VOICE_STYLES)


async def test_under_zdr_voice_is_unavailable_and_inference_is_not_asked(inference, zdr):
    zdr.on = True
    response = await _call("GET", "/status")
    assert response.json()["available"] is False
    assert inference.seen == []
    assert zdr.asked == ["user-jwt"]


@pytest.mark.parametrize("on", [True, RuntimeError("auth down")])
async def test_under_zdr_a_session_is_refused_before_mindshub_hears_anything(inference, zdr, on):
    zdr.on = on
    response = await _call("POST", "/session", json={"sdp": "v=0"})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "zdr_restricted"
    assert inference.seen == []


@pytest.mark.parametrize(("on", "says_zdr"), [
    (True, True),
    (RuntimeError("auth down"), False),
    ("unknown", False),
])
@pytest.mark.parametrize("path", ["/session", "/operate"])
async def test_only_a_known_zdr_names_zdr_in_the_refusal(inference, zdr, on, says_zdr, path):
    from cowork.services.mindshub_zdr import ZdrState

    zdr.on = ZdrState.UNKNOWN if on == "unknown" else on
    body = {"sdp": "v=0"} if path == "/session" else {"messages": [{"role": "user", "content": "hi"}]}
    response = await _call("POST", path, json=body)
    assert response.status_code == 403
    message = response.json()["detail"]["message"]
    assert ("zero data retention" in message) is says_zdr
    assert inference.seen == []


@pytest.mark.parametrize(("on", "message"), [
    (True, "Dictation isn't available while zero data retention is on."),
    ("unknown", "Dictation is unavailable right now. Try again shortly."),
    (RuntimeError("auth down"), "Dictation is unavailable right now. Try again shortly."),
])
async def test_under_zdr_dictation_is_refused_in_its_own_words(inference, zdr, on, message):
    # Dictation's audio goes to the same non-ZDR voice model as a call.
    from cowork.services.mindshub_zdr import ZdrState

    zdr.on = ZdrState.UNKNOWN if on == "unknown" else on
    response = await _call("POST", "/session", json={"sdp": "v=0", "mode": "dictation"})
    assert response.status_code == 403
    assert response.json()["detail"] == {"code": "zdr_restricted", "message": message}
    assert inference.seen == []


async def test_dictation_only_listens_and_reads_no_memory(inference, monkeypatch):
    """ENG-4630: the composer's mic starts a session that only transcribes."""
    import cowork.api.v1.endpoints.voice as endpoint

    def no_memory(scope):
        raise AssertionError("dictation reads no memory")

    monkeypatch.setattr(endpoint, "_person_memory", no_memory)
    started = await _call("POST", "/session", json={"sdp": "offer-sdp", "mode": "dictation", "timezone": "UTC"})

    assert started.status_code == 200, started.text
    instructions = json.loads(inference.seen[0].content)["session"]["instructions"]
    assert instructions == voice.DICTATION_PROMPT
    assert "Never speak" in instructions


async def test_an_unknown_session_mode_is_refused(inference):
    refused = await _call("POST", "/session", json={"sdp": "offer-sdp", "mode": "karaoke"})

    assert refused.status_code == 422
    assert inference.seen == []
