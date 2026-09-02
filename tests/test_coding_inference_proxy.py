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
from cowork.coding.engines.codex_config import LOCAL_PROXY_TOKEN
from cowork.coding.inference_proxy import MAX_INFERENCE_BODY_BYTES, read_inference_body


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
