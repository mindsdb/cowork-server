"""Proxy artifact-comment REST + SSE from the renderer to the inference backend.

Targets inference's auth-gated `/v1/artifact-comments/*` prefix (cowork ≠ browser
viewer, so no vet_ token). SSE is streamed straight through (httpx stream ->
StreamingResponse).

Which credential goes upstream depends on tenancy, and the two answers have
nothing in common. On desktop the renderer holds no bearer, so the server attaches
the user's stored MindsHub credential (the same Minds API key publish uses; auth's
/v1/authenticate/ maps an mdb_ key to X-User-Id = the Keycloak sub). In an org
deployment those settings do not exist, and the renderer DOES hold a bearer — the
one the ingress just validated — so the server forwards that instead. See
resolve_comments_upstream.
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
from cowork.principal import caller_bearer

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


def _org_mode() -> bool:
    """True when this deployment is multi-tenant.

    A local, lazily-imported copy of the same check comments.py makes. Importing
    it from there instead would point a service at the endpoint module that
    imports it, and the settings read is behind an lru_cache anyway.
    """
    from cowork.common.settings.app_settings import get_app_settings

    return get_app_settings().tenancy_mode == "org"


def _org_inference_base() -> str:
    """This deployment's OWN inference base — never a tenant-settable one.

    Mirrors the org model catalog (services/providers.py, fetch_org_minds_models),
    which resolves the same URL for the same reason: `openai_base_url` and
    `minds_url` are settable by an org admin, and sending a member's credential
    to a host they chose is what caller_bearer's contract forbids. The turn
    producer derives the same host through minds_chat_base_url; the two agree on
    every org host, and this spelling leaves out that helper's `mdb.ai` ->
    `/api/v1` rule, which only ever applied to chat.

    Constructed per call on purpose. pydantic-settings reads the environment when
    instantiated and the tests monkeypatch it per test, so hoisting this into a
    module-level constant would freeze whichever value import time happened to see.
    """
    from cowork.common.settings.app_settings import (
        TurnQueueSettings,
        default_turn_minds_api_host,
    )

    return TurnQueueSettings().minds_base_url or f"{default_turn_minds_api_host()}/v1"


def resolve_comments_upstream(request: Request) -> tuple[str, str]:
    """(base_url, credential) for this request, chosen by tenancy.

    Desktop keeps the user's stored MindsHub key and their own endpoint. An org
    pod has neither — no user settings exist for it — so the shared resolver
    yielded an empty key and the gateway answered 401. The caller's own bearer is
    the credential that fits: the ingress just validated it against the very auth
    endpoint inference's ingress uses, and forwarding it makes the upstream
    identity equal to the caller's by construction, with no key to mint, cache or
    revoke.

    The credential is bare (no `Bearer ` prefix), matching what _forward_headers
    expects. An empty one is the caller's problem, not ours: callers must refuse
    rather than send an anonymous upstream request.
    """
    if not _org_mode():
        return resolve_inference_endpoint()
    return _org_inference_base(), caller_bearer(request)


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


def _upstream_or_refusal(request: Request) -> tuple[str, str] | Response:
    """The upstream to call, or the response to answer with instead.

    Both forwarders need the same two refusals, and the org one has to happen
    BEFORE the request goes out: an upstream call carrying no Authorization comes
    back as the gateway's own 401 page, which the renderer shows as "Session
    expired" — a wrong and unactionable message for a server-side gap.

    Desktop keeps its old shape, empty key included. There the credential is a
    user setting that may legitimately be unset, and the request has always gone
    out anyway.
    """
    base, credential = resolve_comments_upstream(request)
    if not base:
        return PlainTextResponse("inference endpoint not configured", status_code=503)
    if _org_mode() and not credential:
        logger.warning("comments proxy: org request carries no caller credential")
        return PlainTextResponse("missing caller credential", status_code=401)
    return base, credential


async def forward_comments_rest(
    request: Request, user_dir: str, report_id: str, subpath: str
) -> Response:
    upstream = _upstream_or_refusal(request)
    if isinstance(upstream, Response):
        return upstream
    base, api_key = upstream
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
    upstream = _upstream_or_refusal(request)
    if isinstance(upstream, Response):
        return upstream
    base, api_key = upstream
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
