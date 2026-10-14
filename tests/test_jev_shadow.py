import asyncio

import httpx
import pytest

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.handlers import jev_shadow


def _settings(**overrides):
    fields = {
        "jev_shadow_enabled": True,
        "jev_shadow_model": "jev",
        "jev_shadow_timeout_seconds": 1.0,
        **overrides,
    }
    return TurnQueueSettings(**fields)


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


LLM_BLOCK = {"provider": "minds-cloud", "api_key": "turn-key", "base_url": "https://minds.example/v1"}


@pytest.mark.asyncio
async def test_probe_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", lambda **_: (_ for _ in ()).throw(AssertionError("must not call out")))
    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings(jev_shadow_enabled=False))
    assert result is None


@pytest.mark.asyncio
async def test_probe_without_minted_credential_is_noop(monkeypatch):
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", lambda **_: (_ for _ in ()).throw(AssertionError("must not call out")))
    result = await jev_shadow.probe(messages=[], llm_block=None, settings=_settings())
    assert result is None


@pytest.mark.asyncio
async def test_probe_success_extracts_choice_and_confidence(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {"route": {"type": "choice", "choice": "needs_agent", "confidence": 0.87, "probabilities": {}}},
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }
    captured = {}
    fake = _FakeAsyncClient(response=_FakeResponse(200, body), captured=captured)
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(
        messages=[{"role": "user", "content": "hi"}], llm_block=LLM_BLOCK, settings=_settings(),
    )

    assert result["jev_choice"] == "needs_agent"
    assert result["jev_confidence"] == 0.87
    assert result["jev_model"] == "jev-1.13.0"
    assert isinstance(result["jev_ms"], int)
    assert captured["url"] == "https://minds.example/v1/decisions"
    assert captured["headers"]["Authorization"] == "Bearer turn-key"
    assert captured["json"]["model"] == "jev"
    assert captured["json"]["state"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "not-a-valid-choice", "confidence": 0.5},
        {"choice": "needs_agent", "confidence": 9},
        {"choice": "needs_agent", "confidence": -0.1},
        {"choice": "needs_agent", "confidence": "0.9"},
        {"choice": "needs_agent", "confidence": True},
        {"choice": "needs_agent", "confidence": float("inf")},
        {"choice": "needs_agent", "confidence": float("nan")},
    ],
)
@pytest.mark.asyncio
async def test_probe_rejects_invalid_answer_values(monkeypatch, answer):
    body = {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", **answer}}}
    fake = _FakeAsyncClient(response=_FakeResponse(200, body))
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())

    assert result["jev_error"] == "malformed_response"
    assert "jev_choice" not in result
    assert "jev_confidence" not in result


@pytest.mark.asyncio
async def test_probe_missing_base_url_is_noop(monkeypatch):
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", lambda **_: (_ for _ in ()).throw(AssertionError("must not call out")))
    block = {"provider": "minds-cloud", "api_key": "turn-key"}
    result = await jev_shadow.probe(messages=[], llm_block=block, settings=_settings())
    assert result is None


@pytest.mark.asyncio
async def test_probe_missing_api_key_is_noop(monkeypatch):
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", lambda **_: (_ for _ in ()).throw(AssertionError("must not call out")))
    block = {"provider": "minds-cloud", "base_url": "https://minds.example/v1"}
    result = await jev_shadow.probe(messages=[], llm_block=block, settings=_settings())
    assert result is None


@pytest.mark.asyncio
async def test_probe_non_200_returns_error_not_raise(monkeypatch):
    fake = _FakeAsyncClient(response=_FakeResponse(429, {}))
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())

    assert result == {"jev_ms": result["jev_ms"], "jev_error": "http_429"}


@pytest.mark.asyncio
async def test_probe_swallows_transport_errors(monkeypatch):
    fake = _FakeAsyncClient(exception=httpx.ConnectError("boom"))
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())

    assert result["jev_error"] == "transport_error"
    assert isinstance(result["jev_ms"], int)


@pytest.mark.asyncio
async def test_probe_swallows_malformed_response(monkeypatch):
    fake = _FakeAsyncClient(response=_FakeResponse(200, {"answers": {}}))
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())

    assert result["jev_error"] == "malformed_response"


@pytest.mark.asyncio
async def test_probe_enforces_a_hard_wall_clock_timeout(monkeypatch):
    """httpx's own timeout kwarg only bounds per-phase inactivity, not total
    elapsed time, so a response that trickles data forever would never trip
    it. The outer asyncio.timeout is what actually has to catch this."""
    fake = _FakeAsyncClient(response=_FakeResponse(200, {"answers": {}}), delay=0.05)
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    started = asyncio.get_event_loop().time()
    result = await jev_shadow.probe(
        messages=[], llm_block=LLM_BLOCK, settings=_settings(jev_shadow_timeout_seconds=0.01),
    )
    elapsed = asyncio.get_event_loop().time() - started

    assert result["jev_error"] == "timeout"
    assert elapsed < 0.05  # returned at the deadline, not after the slow response finished


_SUCCESS_BODY = {
    "model": "jev-1.13.0",
    "answers": {"route": {"type": "choice", "choice": "needs_agent", "confidence": 0.9, "probabilities": {}}},
    "usage": {"input_tokens": 1, "output_tokens": 0},
}


async def _probe_headers(monkeypatch, context):
    """Run the real probe under `context` (or none) and return the headers it sent."""
    from anton.core.llm.tracing import reset_trace_context, set_trace_context

    captured = {}
    fake = _FakeAsyncClient(response=_FakeResponse(200, _SUCCESS_BODY), captured=captured)
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)
    token = set_trace_context(context)
    try:
        await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())
    finally:
        reset_trace_context(token)
    return captured["headers"]


@pytest.mark.asyncio
async def test_probe_attributes_its_trace_to_the_turn_it_shadows(monkeypatch):
    """Without these headers the gateway stamps the probe
    origin:direct-api, indistinguishable from the user calling Jev directly."""
    import json

    from anton.core.llm.tracing import TraceContext

    headers = await _probe_headers(
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
    # Hidden from the user's own Traces list: this is our experiment, not their call.
    assert headers["X-Minds-Request-Kind"] == "probe"
    # Never a session: the customer's Sessions view counts every row in a
    # session whatever its kind, so the probe would join (and could fail) their
    # conversation. The conversation id rides in the metadata instead.
    assert "Langfuse-Session-Id" not in headers
    # The probe's own tag, not the gate's: a probe trace must not read as a gate call.
    assert headers["Langfuse-Tags"].split(",") == ["anton", "surface:web", jev_shadow.JEV_SHADOW_TAG]
    metadata = json.loads(headers["Langfuse-Metadata"])
    # harness is what flips the gateway's origin tag to origin:harness.
    assert metadata == {
        "cowork_server_version": "1.2.3",
        "correlation_id": "corr-1",
        "harness": "anton",
        "surface": "web",
        "conversation_id": "conv-1",
    }


@pytest.mark.asyncio
async def test_probe_never_sends_turn_id(monkeypatch):
    """harness + turn_id makes the gateway name the trace "{harness}:turn-N",
    which would count every probe as a user turn."""
    import json

    from anton.core.llm.tracing import TraceContext

    headers = await _probe_headers(
        monkeypatch,
        TraceContext(session_id="conv-1", harness="anton", turn_id=7, metadata={"turn_id": "7"}),
    )

    assert "turn_id" not in json.loads(headers["Langfuse-Metadata"])


@pytest.mark.asyncio
async def test_probe_outside_a_turn_sends_only_its_credential_and_kind(monkeypatch):
    headers = await _probe_headers(monkeypatch, None)

    assert headers == {"Authorization": "Bearer turn-key", "X-Minds-Request-Kind": "probe"}
