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
