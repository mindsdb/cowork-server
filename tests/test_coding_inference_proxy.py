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
    proxy_inference,
    read_inference_body,
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
