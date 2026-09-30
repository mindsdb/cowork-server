from __future__ import annotations

import asyncio
import json

import anyio
import httpx
from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from cowork.coding.engines.base import EngineCredentials
from cowork.coding.inference_trace import trace_headers

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

# MindsHub answers 429 for three different denials. An exhausted allowance or a
# tripped free-serving fuse holds until a reset hours away, so retrying cannot
# succeed and they fail terminally like the codes above. A velocity limit clears
# within seconds, so the proxy waits out Retry-After itself: Codex's one
# immediate retry would land inside the same window and fail the turn.
TERMINAL_RATE_LIMIT_CODES = frozenset({"included_allowance_exhausted", "free_air_daily_spend_fuse_exceeded"})
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_DEFAULT_WAIT_SECONDS = 2.0
# Codex holds the request open while the proxy waits, so the total wait stays
# well under its stream idle timeout.
RATE_LIMIT_MAX_TOTAL_WAIT_SECONDS = 45.0

# One client per event loop keeps connections to MindsHub open across requests.
# A new client per request paid a TCP and TLS handshake on every model call of
# every turn. An httpx client is bound to the loop it first ran on, so a loop
# change (tests run one per test) gets a fresh client.
_client: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient] | None = None


def _inference_client() -> httpx.AsyncClient:
    global _client
    loop = asyncio.get_running_loop()
    if _client is None or _client[0] is not loop or _client[1].is_closed:
        _client = (loop, httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None)))
    return _client[1]


async def close_inference_client() -> None:
    """Close the shared client during shutdown. Idempotent."""
    global _client
    if _client is None:
        return
    loop, client = _client
    _client = None
    if loop is asyncio.get_running_loop():
        await client.aclose()


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


def upstream_error_code(body: bytes) -> str:
    """Extract the machine-readable ``error.code`` from an upstream error body."""
    try:
        payload = json.loads(body[:MAX_REJECTION_BODY_BYTES])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else ""


def terminal_rejection(status_code: int, body: bytes, code: str | None = None) -> tuple[str, bytes] | None:
    """Return the (code, body) of a non-retryable 400 for a deterministic upstream rejection."""
    code = code or TERMINAL_UPSTREAM_CODES.get(status_code)
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


def retry_after_seconds(value: str | None) -> float:
    """Parse a delta-seconds Retry-After, falling back to a short default."""
    try:
        seconds = float(value) if value else RATE_LIMIT_DEFAULT_WAIT_SECONDS
    except ValueError:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    return seconds if seconds >= 0 else RATE_LIMIT_DEFAULT_WAIT_SECONDS


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
    headers = {**inference_headers(request, credentials.minds_api_key), **trace_headers(request)}
    url = inference_url(credentials.minds_url, path, request.url.query)
    upstream = await _send_through_rate_limits(_inference_client(), request.method, url, headers, body)
    if isinstance(upstream, Response):
        return upstream

    if upstream.status_code in TERMINAL_UPSTREAM_CODES:
        try:
            raw = await _read_bounded(upstream, MAX_REJECTION_BODY_BYTES)
        finally:
            await upstream.aclose()
        return _terminal_response(upstream, raw)
    return _UpstreamStreamingResponse(upstream)


class _UpstreamStreamingResponse(StreamingResponse):
    """Stream an upstream body and always return its connection to the pool.

    The close runs when the response call exits, however it exits. A close in
    the body generator's ``finally`` never runs when Codex disconnects before
    the body starts, because Starlette then cancels the response before the
    generator is entered. Starlette also skips background tasks on a
    disconnect. Either way the pooled connection would stay checked out.
    """

    def __init__(self, upstream: httpx.Response) -> None:
        super().__init__(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
        )
        self._upstream = upstream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Shielded, so the cancellation that ended the stream can't also
            # interrupt the close.
            with anyio.CancelScope(shield=True):
                await self._upstream.aclose()


async def _send_through_rate_limits(
    client: httpx.AsyncClient, method: str, url: str, headers: dict[str, str], body: bytes
) -> httpx.Response | Response:
    """Send, waiting out velocity 429s; returns the streaming upstream or a terminal 400."""
    waited = 0.0
    attempt = 0
    while True:
        try:
            upstream = await client.send(client.build_request(method, url, headers=headers, content=body), stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="MindsHub inference is unavailable") from exc
        if upstream.status_code != 429:
            return upstream
        try:
            raw = await _read_bounded(upstream, MAX_REJECTION_BODY_BYTES)
        finally:
            await upstream.aclose()
        code = upstream_error_code(raw)
        if code in TERMINAL_RATE_LIMIT_CODES:
            return _terminal_response(upstream, raw, code)
        wait = retry_after_seconds(upstream.headers.get("retry-after"))
        if attempt == RATE_LIMIT_RETRIES or waited + wait > RATE_LIMIT_MAX_TOTAL_WAIT_SECONDS:
            return _terminal_response(upstream, raw, "rate_limited")
        attempt += 1
        waited += wait
        await asyncio.sleep(wait)


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name in ("content-type", "retry-after", "x-mindshub-dropped-params", "x-mindshub-reset-at", "x-request-id")
        if (value := upstream.headers.get(name))
    }


def _terminal_response(upstream: httpx.Response, raw: bytes, code: str | None = None) -> Response:
    rejection = terminal_rejection(upstream.status_code, raw, code)
    assert rejection is not None
    code, body = rejection
    headers = _response_headers(upstream)
    headers.pop("content-type", None)
    headers["x-mindshub-error-code"] = code
    headers["x-mindshub-upstream-status"] = str(upstream.status_code)
    return Response(content=body, status_code=400, media_type="application/json", headers=headers)


async def _read_bounded(upstream: httpx.Response, limit: int) -> bytes:
    body = bytearray()
    async for chunk in upstream.aiter_bytes():
        body.extend(chunk[: max(0, limit - len(body))])
        if len(body) >= limit:
            break
    return bytes(body)
