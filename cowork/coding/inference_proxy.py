from __future__ import annotations

import json

import httpx
from fastapi import HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

from cowork.coding.engines.base import EngineCredentials

INFERENCE_PATHS = {"models", "responses", "responses/compact"}
MAX_INFERENCE_BODY_BYTES = 16 * 1024 * 1024
MAX_REJECTION_BODY_BYTES = 64 * 1024

# Codex retries every non-2xx response except 400, so a deterministic upstream
# rejection (bad credential, empty wallet, unknown model) is otherwise repeated
# five times before the task fails with the raw status line. These are rewritten
# to a single terminal 400 whose body names the failure.
TERMINAL_UPSTREAM_CODES = {
    401: "model_authentication_failed",
    402: "insufficient_credits",
    403: "model_authentication_failed",
    404: "model_unavailable",
}


def inference_url(minds_url: str, path: str, query: str = "") -> str:
    base = minds_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    url = f"{base}/{path}"
    return f"{url}?{query}" if query else url


def inference_body(body: bytes) -> bytes:
    """Remove Codex transport metadata rejected by MindsHub Inference."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or "client_metadata" not in payload:
        return body
    payload.pop("client_metadata")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def upstream_error_message(body: bytes) -> str:
    """Extract the human-readable message from an upstream error body."""
    text = body[:MAX_REJECTION_BODY_BYTES].decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or text)
    if isinstance(error, str):
        return error
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"]
    return text


def terminal_rejection(status_code: int, body: bytes) -> tuple[str, bytes] | None:
    """Return the (code, body) of a non-retryable 400 for a deterministic upstream rejection."""
    code = TERMINAL_UPSTREAM_CODES.get(status_code)
    if code is None:
        return None
    payload = {
        "error": {
            "message": upstream_error_message(body) or f"MindsHub inference rejected the request ({status_code})",
            "type": "invalid_request_error",
            "code": code,
            "upstream_status": status_code,
        }
    }
    return code, json.dumps(payload, ensure_ascii=False).encode()


def inference_headers(request: Request, api_key: str) -> dict[str, str]:
    """Build the narrow upstream header set accepted by MindsHub Inference."""
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type := request.headers.get("content-type"):
        headers["content-type"] = content_type
    return headers


async def read_inference_body(request: Request) -> bytes:
    """Read a bounded request body without allowing an untrusted allocation."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length > MAX_INFERENCE_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Inference request is too large")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_INFERENCE_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Inference request is too large")
        body.extend(chunk)
    return bytes(body)


async def proxy_inference(request: Request, path: str, credentials: EngineCredentials) -> StreamingResponse:
    if path not in INFERENCE_PATHS:
        raise HTTPException(status_code=404, detail="inference route not found")
    if not credentials.minds_api_key:
        raise HTTPException(status_code=409, detail="MindsHub is not connected")

    body = inference_body(await read_inference_body(request))
    headers = inference_headers(request, credentials.minds_api_key)
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                inference_url(credentials.minds_url, path, request.url.query),
                headers=headers,
                content=body,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail="MindsHub inference is unavailable") from exc

    async def close_upstream() -> None:
        await upstream.aclose()
        await client.aclose()

    response_headers = {
        name: value
        for name in ("content-type", "retry-after", "x-mindshub-dropped-params", "x-request-id")
        if (value := upstream.headers.get(name))
    }
    if upstream.status_code in TERMINAL_UPSTREAM_CODES:
        try:
            raw = await _read_bounded(upstream, MAX_REJECTION_BODY_BYTES)
        finally:
            await close_upstream()
        code, body = terminal_rejection(upstream.status_code, raw)
        response_headers.pop("content-type", None)
        response_headers["x-mindshub-error-code"] = code
        response_headers["x-mindshub-upstream-status"] = str(upstream.status_code)
        return Response(content=body, status_code=400, media_type="application/json", headers=response_headers)
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )


async def _read_bounded(upstream: httpx.Response, limit: int) -> bytes:
    body = bytearray()
    async for chunk in upstream.aiter_bytes():
        body.extend(chunk[: max(0, limit - len(body))])
        if len(body) >= limit:
            break
    return bytes(body)
