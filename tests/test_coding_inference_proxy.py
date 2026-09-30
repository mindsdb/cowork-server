from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from cowork.api.v1.endpoints.coding import (
    _inference_body,
    _inference_headers,
    _inference_url,
    _require_inference_client,
)
import json

import httpx

from cowork.coding import inference_proxy as inference_proxy_module
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.codex_config import LOCAL_PROXY_TOKEN
from cowork.coding.inference_proxy import (
    MAX_INFERENCE_BODY_BYTES,
    RATE_LIMIT_DEFAULT_WAIT_SECONDS,
    RATE_LIMIT_RETRIES,
    proxy_inference,
    read_inference_body,
    retry_after_seconds,
    terminal_rejection,
    upstream_error_message,
)


def _request(headers: list[tuple[bytes, bytes]], body: bytes = b"") -> Request:
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {"type": "http", "method": "POST", "path": "/", "headers": headers},
        receive=receive,
    )


def test_inference_url_normalizes_the_responses_api_base() -> None:
    assert _inference_url("https://api.mindshub.ai", "responses") == "https://api.mindshub.ai/v1/responses"
    assert _inference_url("https://api.mindshub.ai/v1/", "models") == "https://api.mindshub.ai/v1/models"
    assert _inference_url("https://api.mindshub.ai", "models", "after=model-1&limit=20") == (
        "https://api.mindshub.ai/v1/models?after=model-1&limit=20"
    )


def test_inference_headers_strip_codex_transport_headers() -> None:
    request = _request(
        [
            (b"accept", b"text/event-stream"),
            (b"content-type", b"application/json"),
            (b"x-codex-turn-metadata", b'{"turn_id":"turn-1"}'),
        ]
    )

    headers = _inference_headers(request, "secret-key")

    assert headers == {
        "Authorization": "Bearer secret-key",
        "content-type": "application/json",
    }


def test_inference_proxy_requires_the_process_local_codex_credential() -> None:
    _require_inference_client(_request([(b"authorization", f"Bearer {LOCAL_PROXY_TOKEN}".encode())]))

    with pytest.raises(HTTPException) as missing:
        _require_inference_client(_request([]))
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        _require_inference_client(_request([(b"authorization", b"Bearer mindshub-cowork-loopback")]))
    assert wrong.value.status_code == 401


def test_inference_body_strips_codex_client_metadata() -> None:
    body = b'{"model":"fable","client_metadata":{"turn_id":"turn-1"},"stream":true}'

    assert _inference_body(body) == b'{"model":"fable","stream":true}'


def test_inference_body_preserves_non_json_and_unrelated_payloads() -> None:
    assert _inference_body(b"") == b""
    assert _inference_body(b"not-json") == b"not-json"
    assert _inference_body(b'{"model":"fable"}') == b'{"model":"fable"}'


@pytest.mark.asyncio
async def test_inference_body_reader_accepts_a_bounded_payload() -> None:
    body = b'{"model":"fable"}'
    request = _request([(b"content-length", str(len(body)).encode())], body)

    assert await read_inference_body(request) == body


@pytest.mark.asyncio
async def test_inference_body_reader_rejects_declared_and_streamed_oversize_payloads() -> None:
    declared = _request([(b"content-length", str(MAX_INFERENCE_BODY_BYTES + 1).encode())])
    with pytest.raises(HTTPException) as declared_error:
        await read_inference_body(declared)
    assert declared_error.value.status_code == 413

    streamed = _request([], b"x" * (MAX_INFERENCE_BODY_BYTES + 1))
    with pytest.raises(HTTPException) as streamed_error:
        await read_inference_body(streamed)
    assert streamed_error.value.status_code == 413


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "model_authentication_failed"), (402, "insufficient_credits"), (403, "model_authentication_failed"), (404, "model_unavailable")],
)
def test_deterministic_upstream_rejections_become_a_terminal_400_with_a_stable_code(status: int, code: str) -> None:
    rejection = terminal_rejection(status, b'{"error": {"message": "Your wallet has no balance.", "type": "x"}}')

    assert rejection is not None
    returned_code, body = rejection
    assert returned_code == code
    assert json.loads(body) == {
        "error": {
            "message": "Your wallet has no balance.",
            "type": "invalid_request_error",
            "code": code,
            "upstream_status": status,
        }
    }


@pytest.mark.parametrize("status", [400, 429, 500, 502, 503])
def test_retryable_and_already_terminal_statuses_are_not_rewritten(status: int) -> None:
    assert terminal_rejection(status, b"{}") is None


def test_upstream_error_message_survives_plain_text_and_odd_json_shapes() -> None:
    assert upstream_error_message(b"Payment Required") == "Payment Required"
    assert upstream_error_message(b'{"error": "wallet empty"}') == "wallet empty"
    assert upstream_error_message(b'{"detail": "Not Found"}') == "Not Found"
    assert upstream_error_message(b"\xff\xfe") == "��"


@pytest.mark.asyncio
async def test_proxy_answers_an_upstream_402_with_one_terminal_400(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(
            402,
            json={"error": {"message": "Your wallet has no balance to cover the model 'gpt'.", "code": "insufficient_credits"}},
            headers={"x-request-id": "req-1"},
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inference_proxy_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    request = _request([(b"content-type", b"application/json")], b'{"model": "gpt", "input": "hi"}')

    response = await proxy_inference(request, "responses", EngineCredentials(minds_url="https://api.example", minds_api_key="mdb_key"))

    assert upstream_calls == 1
    assert response.status_code == 400
    assert response.headers["x-mindshub-error-code"] == "insufficient_credits"
    assert response.headers["x-mindshub-upstream-status"] == "402"
    assert response.headers["x-request-id"] == "req-1"
    assert json.loads(response.body)["error"]["code"] == "insufficient_credits"
    assert json.loads(response.body)["error"]["message"] == "Your wallet has no balance to cover the model 'gpt'."


@pytest.mark.asyncio
async def test_proxy_streams_successful_and_retryable_responses_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"busy", headers={"retry-after": "2"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inference_proxy_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    request = _request([(b"content-type", b"application/json")], b"{}")

    response = await proxy_inference(request, "responses", EngineCredentials(minds_url="https://api.example", minds_api_key="mdb_key"))

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert "x-mindshub-error-code" not in response.headers


def _mock_upstream(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> list[float]:
    """Serve ``responses`` in order (repeating the last) and record proxy sleeps instead of waiting."""
    calls = iter(responses)
    last = responses[-1]
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return next(calls, last)

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inference_proxy_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(inference_proxy_module.asyncio, "sleep", fake_sleep)
    return sleeps


def _rate_limited(retry_after: str | None = "1") -> httpx.Response:
    return httpx.Response(
        429,
        json={"error": {"message": "Rate limit exceeded for model 'gpt'. Please slow down and retry.", "code": "rate_limited"}},
        headers={"retry-after": retry_after} if retry_after is not None else {},
    )


async def _proxy(body: bytes = b"{}"):
    request = _request([(b"content-type", b"application/json")], body)
    return await proxy_inference(request, "responses", EngineCredentials(minds_url="https://api.example", minds_api_key="mdb_key"))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["included_allowance_exhausted", "free_air_daily_spend_fuse_exceeded"])
async def test_proxy_fails_a_reset_bound_429_terminally_without_waiting(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    upstream = httpx.Response(
        429,
        json={"error": {"message": "Your included allowance for 'gpt' is exhausted.", "code": code}},
        headers={"x-mindshub-reset-at": "2026-09-26T19:00:00Z", "x-should-retry": "false"},
    )
    sleeps = _mock_upstream(monkeypatch, [upstream, httpx.Response(200, content=b"ok")])

    response = await _proxy()

    assert sleeps == []
    assert response.status_code == 400
    assert response.headers["x-mindshub-error-code"] == code
    assert response.headers["x-mindshub-upstream-status"] == "429"
    assert response.headers["x-mindshub-reset-at"] == "2026-09-26T19:00:00Z"
    assert json.loads(response.body)["error"]["code"] == code


@pytest.mark.asyncio
async def test_proxy_waits_out_a_velocity_429_and_streams_the_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _mock_upstream(monkeypatch, [_rate_limited("3"), _rate_limited(None), httpx.Response(200, content=b"data: ok")])

    response = await _proxy()

    assert sleeps == [3.0, RATE_LIMIT_DEFAULT_WAIT_SECONDS]
    assert response.status_code == 200
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == b"data: ok"


@pytest.mark.asyncio
async def test_proxy_fails_a_persistent_velocity_429_as_terminal_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _mock_upstream(monkeypatch, [_rate_limited("1")])

    response = await _proxy()

    assert len(sleeps) == RATE_LIMIT_RETRIES
    assert response.status_code == 400
    assert response.headers["x-mindshub-error-code"] == "rate_limited"
    error = json.loads(response.body)["error"]
    assert error["code"] == "rate_limited"
    assert error["message"] == "Rate limit exceeded for model 'gpt'. Please slow down and retry."


@pytest.mark.asyncio
async def test_proxy_does_not_hold_codex_past_the_wait_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _mock_upstream(monkeypatch, [_rate_limited("600"), httpx.Response(200, content=b"ok")])

    response = await _proxy()

    assert sleeps == []
    assert response.status_code == 400
    assert response.headers["x-mindshub-error-code"] == "rate_limited"


@pytest.mark.asyncio
async def test_proxy_treats_an_unlabelled_429_as_a_velocity_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _mock_upstream(monkeypatch, [httpx.Response(429, content=b"Too Many Requests"), httpx.Response(200, content=b"ok")])

    response = await _proxy()

    assert sleeps == [RATE_LIMIT_DEFAULT_WAIT_SECONDS]
    assert response.status_code == 200


def test_retry_after_parses_delta_seconds_and_falls_back_on_anything_else() -> None:
    assert retry_after_seconds("4") == 4.0
    assert retry_after_seconds("0") == 0.0
    assert retry_after_seconds(None) == RATE_LIMIT_DEFAULT_WAIT_SECONDS
    assert retry_after_seconds("-1") == RATE_LIMIT_DEFAULT_WAIT_SECONDS
    assert retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") == RATE_LIMIT_DEFAULT_WAIT_SECONDS


class _ClosingStream(httpx.AsyncByteStream):
    """An upstream body that records when the proxy closes it."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_proxy_reuses_one_upstream_client_across_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    created = 0
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        nonlocal created
        created += 1
        return real_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"ok")), **kwargs)

    monkeypatch.setattr(inference_proxy_module.httpx, "AsyncClient", factory)

    for _ in range(3):
        response = await _proxy()
        assert b"".join([chunk async for chunk in response.body_iterator]) == b"ok"

    assert created == 1
    await inference_proxy_module.close_inference_client()


@pytest.mark.asyncio
async def test_proxy_closes_the_upstream_when_codex_stops_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _ClosingStream()
    _mock_upstream(monkeypatch, [httpx.Response(200, stream=stream)])

    response = await _proxy()
    iterator = response.body_iterator
    assert await anext(iterator) == b"data: one\n\n"
    # A disconnect closes the body iterator before it is exhausted.
    await iterator.aclose()

    assert stream.closed
