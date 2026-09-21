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

    def __init__(self, *, response=None, exception=None, captured=None):
        self._response = response
        self._exception = exception
        self._captured = captured if captured is not None else {}

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
    assert isinstance(result["jev_ms"], int)
    assert captured["url"] == "https://minds.example/v1/decisions"
    assert captured["headers"]["Authorization"] == "Bearer turn-key"
    assert captured["json"]["model"] == "jev"
    assert captured["json"]["state"] == [{"role": "user", "content": "hi"}]


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

    assert result["jev_error"] == "exception"
    assert isinstance(result["jev_ms"], int)


@pytest.mark.asyncio
async def test_probe_swallows_malformed_response(monkeypatch):
    fake = _FakeAsyncClient(response=_FakeResponse(200, {"answers": {}}))
    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", fake)

    result = await jev_shadow.probe(messages=[], llm_block=LLM_BLOCK, settings=_settings())

    assert result["jev_error"] == "exception"
