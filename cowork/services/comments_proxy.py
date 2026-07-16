"""Proxy artifact-comment REST + SSE from the renderer to the inference backend.

The renderer holds no bearer token, so it calls these cowork-server routes and the
server attaches the user's MindsHub credential (the same Minds API key publish uses;
auth's /v1/authenticate/ maps an mdb_ key to X-User-Id = the Keycloak sub). Targets
inference's auth-gated `/v1/artifact-comments/*` prefix (cowork ≠ browser viewer, no
vet_). SSE is streamed straight through (httpx stream -> StreamingResponse).
"""

from __future__ import annotations

import logging
from urllib.parse import quote, urlparse

import httpx
from fastapi import Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import SecretStr

from cowork.common.http_client import get_proxy_client
from cowork.common.settings.user_settings import Provider, get_user_settings, provider_api_key

logger = logging.getLogger(__name__)

_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}
# Response headers httpx/ASGI must recompute or that don't apply across the hop.
_HOP_HEADERS = {
    "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
    "content-length", "content-encoding",
}


def _secret_str(val: SecretStr | str | None) -> str:
    if val is None:
        return ""
    return val.get_secret_value() if isinstance(val, SecretStr) else str(val)


def resolve_inference_endpoint(settings=None) -> tuple[str, str]:
    """(base_url, api_key) for the active provider's env — mirrors publish's resolver.

    A custom OpenAI-compatible MindsHub endpoint (dev/staging) wins over the default
    minds_url (prod); the base already includes `/v1`.
    """
    settings = settings or get_user_settings()
    oai = settings.openai_base_url or ""
    host = (urlparse(oai).hostname or "").lower()
    # Any MindsHub host counts — dev/staging use hyphenated subdomains
    # (e.g. api-gitlab.dev.mindshub.ai), not just the prod `api.` prefix.
    if host == "mindshub.ai" or host.endswith(".mindshub.ai"):
        return oai.rstrip("/"), _secret_str(provider_api_key(settings, Provider.OPENAI_COMPATIBLE))
    return (settings.minds_url or "").rstrip("/"), _secret_str(settings.minds_api_key)


# Segments that would traverse out of the /artifact-comments/ prefix on the
# upstream (httpx sends dot-segments verbatim; nginx normalizes them away).
_BAD_SEGMENTS = {"", ".", ".."}


def _clean_segments(user_dir: str, report_id: str, subpath: str) -> list[str]:
    """Path segments for the upstream URL, or raise ValueError on traversal.

    user_dir/report_id are single router segments (the [^/]+ converter already
    forbids '/'); subpath is a {path} param and may hold several '/'-joined
    segments. Reject any empty or dot-segment, and any residual slash/backslash
    (an encoded %2F/%5C that slipped through), so a caller can't climb above the
    prefix or smuggle extra path structure. Each survivor is percent-encoded so
    stray '?'/'#'/'%' can't rewrite the URL either.
    """
    segments = [user_dir, report_id, *(subpath.split("/") if subpath else [])]
    for seg in segments:
        if seg in _BAD_SEGMENTS or "/" in seg or "\\" in seg:
            raise ValueError(f"invalid path segment: {seg!r}")
    return segments


def _upstream_url(base: str, user_dir: str, report_id: str, subpath: str, query: str) -> str:
    path = "/".join(quote(seg, safe="") for seg in _clean_segments(user_dir, report_id, subpath))
    url = f"{base}/artifact-comments/{path}"
    if query:
        url = f"{url}?{query}"
    return url


def _forward_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def forward_comments_rest(
    request: Request, user_dir: str, report_id: str, subpath: str
) -> Response:
    base, api_key = resolve_inference_endpoint()
    if not base:
        return PlainTextResponse("inference endpoint not configured", status_code=503)
    client = get_proxy_client()
    body = await request.body()
    try:
        url = _upstream_url(base, user_dir, report_id, subpath, request.url.query)
    except ValueError:
        return PlainTextResponse("invalid path", status_code=400)
    try:
        r = await client.request(
            request.method, url, headers=_forward_headers(api_key), content=body
        )
    except httpx.RequestError as exc:
        logger.warning("comments proxy REST upstream error: %s", exc)
        return PlainTextResponse("upstream connection error", status_code=502)
    out_headers = {k: v for k, v in r.headers.items() if k.lower() not in _HOP_HEADERS}
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=out_headers,
        media_type=r.headers.get("content-type"),
    )


async def forward_comments_stream(
    request: Request, user_dir: str, report_id: str
) -> Response:
    base, api_key = resolve_inference_endpoint()
    if not base:
        return PlainTextResponse("inference endpoint not configured", status_code=503)
    client = get_proxy_client()
    try:
        url = _upstream_url(base, user_dir, report_id, "stream", request.url.query)
    except ValueError:
        return PlainTextResponse("invalid path", status_code=400)
    headers = _forward_headers(api_key)
    # NOT a bare "text/event-stream": the auth-gated ingress runs an nginx
    # auth_request subrequest to auth (DRF), forwarding this Accept. DRF only
    # renders application/json|text/html, so a bare event-stream Accept makes
    # auth return 406 -> nginx turns any non-2xx/401/403 auth status into a 500,
    # and the request never reaches inference. Append */* so DRF negotiates JSON
    # for the auth check; inference's StreamingResponse sets the SSE media type
    # regardless of Accept.
    headers["Accept"] = "text/event-stream, */*"
    # read=None: the SSE connection is long-lived; a read timeout would sever it.
    upstream_req = client.build_request(
        "GET", url, headers=headers, timeout=httpx.Timeout(30.0, connect=5.0, read=None)
    )
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as exc:
        logger.warning("comments proxy stream upstream error: %s", exc)
        return PlainTextResponse("upstream connection error", status_code=502)

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=dict(_SSE_HEADERS),
        media_type="text/event-stream",
    )
