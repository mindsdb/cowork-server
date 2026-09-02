from __future__ import annotations

import json

import httpx
from fastapi import HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from cowork.coding.engines.base import EngineCredentials

INFERENCE_PATHS = {"models", "responses", "responses/compact"}
MAX_INFERENCE_BODY_BYTES = 16 * 1024 * 1024


def inference_response_status(status_code: int) -> int:
    """Stop the Codex transport retrying a deterministic credit rejection."""
    return 400 if status_code == 402 else status_code


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
    if upstream.status_code == 402:
        response_headers["x-mindshub-error-code"] = "insufficient_credits"
        response_headers["x-mindshub-upstream-status"] = "402"
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=inference_response_status(upstream.status_code),
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )
