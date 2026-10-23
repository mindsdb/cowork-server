import asyncio
import json

import httpx
import pytest

from cowork.handlers import jev_route

TARGET = jev_route.JevTarget(base_url="https://minds.example/v1/", api_key="turn-key")
MESSAGES = [
    {"role": "user", "content": "Build a dashboard of our 2025 sales."},
    {"role": "assistant", "content": "[ran tool: query_datasource]\nRevenue was $4.81M."},
    {"role": "user", "content": "what were those headline numbers again?"},
]


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; scripts one POST response or raises."""

    def __init__(self, *, response=None, exception=None, captured=None, delay=0.0):
        self._response = response
        self._exception = exception
        self._captured = captured if captured is not None else {}
        self._delay = delay

    def __call__(self, *_args, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, *, headers, json):
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["json"] = json
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exception is not None:
            raise self._exception
        return self._response


def _body(choice="answer_directly", p_direct=0.97):
    return {
        "model": "jev-1.13.0",
        "answers": {"route": {
            "type": "choice", "choice": choice, "confidence": 0.9,
            "probabilities": {"answer_directly": p_direct, "needs_agent": 1 - p_direct},
        }},
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


async def _decide(monkeypatch, client, *, min_p_direct=0.5, timeout_seconds=1.0):
    monkeypatch.setattr(jev_route.httpx, "AsyncClient", client)
    return await jev_route.decide(
        target=TARGET, messages=MESSAGES, model="jev",
        timeout_seconds=timeout_seconds, min_p_direct=min_p_direct,
    )


@pytest.mark.asyncio
async def test_decide_sends_the_route_question_with_the_latest_message_apart(monkeypatch):
    captured = {}
    result = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured))

    assert captured["url"] == "https://minds.example/v1/decisions"
    assert captured["json"] == {
        "model": "jev",
        "state": {
            "latest_user_message": "what were those headline numbers again?",
            "earlier_conversation": MESSAGES[:-1],
        },
        "questions": {"route": jev_route.ROUTE_QUESTION},
    }
    assert result == jev_route.JevRoute(direct=True, p_direct=0.97, ms=result.ms, model="jev-1.13.0")


@pytest.mark.asyncio
@pytest.mark.parametrize(("p_direct", "min_p_direct", "direct"), [
    (0.97, 0.5, True),
    (0.5, 0.5, True),
    (0.49, 0.5, False),
    (0.55, 0.8, False),
])
async def test_decide_routes_direct_only_at_or_above_the_threshold(monkeypatch, p_direct, min_p_direct, direct):
    """The probability decides, not the choice label: a direct answer needs
    Jev at least this sure, so the threshold can be tuned without Jev."""
    result = await _decide(
        monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body(p_direct=p_direct))),
        min_p_direct=min_p_direct,
    )

    assert result.direct is direct
    assert result.p_direct == p_direct
    assert result.error == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    {"choice": "maybe", "probabilities": {"answer_directly": 0.9}},
    {"choice": "answer_directly", "probabilities": {}},
    {"choice": "answer_directly"},
    {"choice": "answer_directly", "probabilities": {"answer_directly": True}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": "0.9"}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": 1.5}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": float("nan")}},
])
async def test_decide_rejects_a_malformed_answer(monkeypatch, answer):
    body = {"answers": {"route": {"type": "choice", **answer}}}
    result = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, body)))

    assert result.error == "malformed_response"
    assert result.direct is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("client", "error"), [
    (_FakeAsyncClient(response=_FakeResponse(402, {})), "http_402"),
    (_FakeAsyncClient(exception=httpx.ConnectError("boom")), "transport_error"),
    (_FakeAsyncClient(exception=RuntimeError("boom")), "RuntimeError"),
    (_FakeAsyncClient(response=_FakeResponse(200, {"unexpected": True})), "malformed_response"),
])
async def test_decide_returns_failures_and_never_raises(monkeypatch, client, error):
    result = await _decide(monkeypatch, client)

    assert result.error == error
    assert result.direct is False


@pytest.mark.asyncio
async def test_decide_enforces_a_hard_wall_clock_timeout(monkeypatch):
    """httpx's own timeout is per phase and measures inactivity, not total
    elapsed time, so a response that trickles data forever would never trip
    it. The outer asyncio.timeout is what actually has to catch this."""
    started = asyncio.get_event_loop().time()
    result = await _decide(
        monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), delay=0.05), timeout_seconds=0.01,
    )
    elapsed = asyncio.get_event_loop().time() - started

    assert result.error == "timeout"
    assert elapsed < 0.05  # returned at the deadline, not after the slow response finished


async def _decide_headers(monkeypatch, context):
    """Run the real decision under `context` (or none) and return the headers it sent."""
    from anton.core.llm.tracing import reset_trace_context, set_trace_context

    captured = {}
    token = set_trace_context(context)
    try:
        await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured))
    finally:
        reset_trace_context(token)
    return captured["headers"]


@pytest.mark.asyncio
async def test_decide_attributes_its_trace_to_the_turn_it_routes(monkeypatch):
    """Without these headers the gateway stamps the call origin:direct-api,
    indistinguishable from the user calling Jev directly."""
    from anton.core.llm.tracing import TraceContext

    headers = await _decide_headers(
        monkeypatch,
        TraceContext(
            session_id="conv-1",
            harness="anton",
            surface="web",
            tags=("cowork-gate",),
            metadata={"cowork_server_version": "1.2.3", "correlation_id": "corr-1"},
        ),
    )

    assert headers["Authorization"] == "Bearer turn-key"
    # The user's own turn, so not hidden as a probe.
    assert "X-Minds-Request-Kind" not in headers
    # Never a session: a refused decision would mark the user's conversation
    # failed in the Sessions view, though the turn recovers by delegating.
    assert "Langfuse-Session-Id" not in headers
    assert headers["Langfuse-Tags"].split(",") == ["anton", "surface:web", jev_route.JEV_ROUTE_TAG]
    assert json.loads(headers["Langfuse-Metadata"]) == {
        "cowork_server_version": "1.2.3",
        "correlation_id": "corr-1",
        "harness": "anton",
        "surface": "web",
        "conversation_id": "conv-1",
    }


@pytest.mark.asyncio
async def test_decide_never_sends_turn_id(monkeypatch):
    """harness + turn_id makes the gateway name the trace "{harness}:turn-N",
    which would count every decision as a user turn."""
    from anton.core.llm.tracing import TraceContext

    headers = await _decide_headers(
        monkeypatch,
        TraceContext(session_id="conv-1", harness="anton", turn_id=7, metadata={"turn_id": "7"}),
    )

    assert "turn_id" not in json.loads(headers["Langfuse-Metadata"])


@pytest.mark.asyncio
async def test_decide_outside_a_turn_sends_only_its_credential(monkeypatch):
    assert await _decide_headers(monkeypatch, None) == {"Authorization": "Bearer turn-key"}
