from __future__ import annotations

import json

import httpx
from fastapi import HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from cowork.coding.engines.base import EngineCredentials

INFERENCE_PATHS = {"models", "responses", "responses/compact"}


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


async def proxy_inference(request: Request, path: str, credentials: EngineCredentials) -> StreamingResponse:
    if path not in INFERENCE_PATHS:
        raise HTTPException(status_code=404, detail="inference route not found")
    if not credentials.minds_api_key:
        raise HTTPException(status_code=409, detail="MindsHub is not connected")

    body = inference_body(await request.body())
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
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )
