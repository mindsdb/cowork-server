"""HTTP forwarder for fullstack-artifact previews.

Ported from `cowork/server/anton_api/preview_proxy.py`. Unlike the
desktop version (which ran a second uvicorn on a loopback port), this
implementation hooks straight into the main FastAPI app as a route:
the proxy URL is `<server-base>/api/v1/artifacts/proxy/{token}/...`.
That works behind a reverse proxy because no extra port is involved.

Each request looks up the artifact dir by token, re-reads `port` from
`metadata.json` on every call (so a backend restart on a fresh port is
picked up automatically), and streams the upstream response back. CORS
headers are injected on every response — artifact backends aren't
required to know about CORS, and the sandboxed iframe has an opaque
origin, so without these headers every fetch from artifact JS would be
blocked browser-side.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse

from cowork.common.http_client import get_proxy_client
from cowork.services.artifacts import get_preview_mount
from cowork.services.comments_layer import ACTIVATION_PARAM, inject_layer

logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 §6.1) — never forwarded either direction.
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Any CORS headers from the upstream backend are dropped so we can
# inject our own consistent set — duplicates would be treated as a
# CORS error by the browser.
_CORS_RESPONSE_HEADERS = {
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-allow-credentials",
    "access-control-expose-headers",
    "access-control-max-age",
}

# Headers the upstream `httpx` request must not carry — `Content-Length`
# is recomputed by httpx itself, and `Host` is rewritten below.
_UPSTREAM_BLOCKED_REQUEST_HEADERS = {"content-length", "host"}


def _read_backend_port(artifact_dir: Path) -> Optional[int]:
    try:
        meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    port = meta.get("port")
    if isinstance(port, int) and 0 < port < 65536:
        return port
    return None


def _cors_headers(req: Request) -> dict[str, str]:
    requested = req.headers.get("access-control-request-headers") or "*"
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": requested,
        "Access-Control-Max-Age": "600",
    }


def _strip_hop_headers(headers, *, drop_cors: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in headers.items():
        lk = k.lower()
        if lk in _HOP_HEADERS:
            continue
        if drop_cors and lk in _CORS_RESPONSE_HEADERS:
            continue
        out.append((k, v))
    return out


def _build_upstream_headers(req: Request, backend_port: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in req.headers.items():
        lk = k.lower()
        if lk in _HOP_HEADERS or lk in _UPSTREAM_BLOCKED_REQUEST_HEADERS:
            continue
        out.append((k, v))
    out.append(("host", f"127.0.0.1:{backend_port}"))
    return out


async def proxy_artifact_request(
    token: str, rel_path: str, request: Request
) -> Response:
    """Forward an iframe request to the artifact's backend.

    `rel_path` is the path *after* `/proxy/{token}/`, e.g. an empty
    string for the index page, or `static/foo.js` for a relative asset
    the artifact's HTML loaded. It's joined onto the backend's root.
    """
    cors = _cors_headers(request)

    # Short-circuit preflight so artifact backends don't need to
    # implement OPTIONS themselves.
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=cors)

    artifact_dir = get_preview_mount(token)
    if artifact_dir is None:
        return PlainTextResponse(
            "Preview mount has expired or is unknown",
            status_code=404,
            headers=cors,
        )
    backend_port = _read_backend_port(artifact_dir)
    if backend_port is None:
        return PlainTextResponse(
            "Artifact backend is not running yet",
            status_code=503,
            headers=cors,
        )

    client = get_proxy_client()

    body = await request.body()
    # `rel_path` from FastAPI's `{rel_path:path}` has no leading slash;
    # add one explicitly so the URL is well-formed when the iframe
    # requests the index (`rel_path == ""`).
    upstream_path = f"/{rel_path}" if rel_path else "/"
    url = f"http://127.0.0.1:{backend_port}{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    upstream_headers = _build_upstream_headers(request, backend_port)

    upstream_req = client.build_request(
        request.method,
        url,
        headers=upstream_headers,
        content=body,
    )
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as exc:
        # ECONNREFUSED while the backend is starting / dead — surface as
        # 502 so the iframe shows the upstream error rather than us.
        return PlainTextResponse(
            f"Proxy error: {exc}", status_code=502, headers=cors
        )

    # For the root HTML response, patch the `api-base` meta tag so that
    # absolute-path fetch calls (e.g. fetch('/api/top-cpu')) from the
    # artifact JS are routed through this proxy route, not to the cowork
    # server root.  The proxy lives at /api/v1/artifacts/proxy/{token}/…,
    # so a bare /api/… path misses it entirely and hits a 404.
    content_type = upstream.headers.get("content-type", "")
    if (
        upstream_path == "/"
        and upstream.status_code == 200
        and "text/html" in content_type
    ):
        raw = await upstream.aread()
        await upstream.aclose()
        proxy_prefix = f"/api/v1/artifacts/proxy/{token}"
        # Replace the conventional empty api-base value that Anton generates.
        patched = raw.replace(
            b'name="api-base" content=""',
            f'name="api-base" content="{proxy_prefix}"'.encode(),
        )
        # Inject the on-artifact comment marker layer when the renderer opts in
        # (same activation flag as the static serve path). Fullstack previews
        # flow through this proxy rather than serve_artifact_file, so this is the
        # equivalent injection point for them.
        if ACTIVATION_PARAM in request.query_params:
            try:
                patched = inject_layer(patched.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                pass
        resp_headers = dict(_strip_hop_headers(upstream.headers, drop_cors=True))
        resp_headers.update(cors)
        # Drop Content-Length — the patched body may be larger than the
        # original; let the ASGI layer set the correct value.
        resp_headers.pop("content-length", None)
        resp_headers.pop("Content-Length", None)
        return Response(
            content=patched,
            status_code=upstream.status_code,
            headers=resp_headers,
        )

    resp_headers = _strip_hop_headers(upstream.headers, drop_cors=True)
    resp_headers.extend(cors.items())

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=dict(resp_headers),
    )
