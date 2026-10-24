"""The draft-preview host.

A small Starlette app served on a port of its own (cowork/draft_host/listener.py),
so ``TrustedHeaderMiddleware``, the bearer layer, CORS and every ``X-User-*``
header of the API app are not on this code path at all. The only route is
``GET/HEAD /_t/{token}/{rel_path}``, and only for a ``Host`` in
COWORK_DRAFT_HOSTS. User, organization, project and artifact come from the
signed token alone, and the file is found by the same code the authenticated
``/drafts`` route uses, so the review grant is checked again on every request.

The boot-time route walker does not see this app, so
tests/test_draft_host_app.py pins its surface instead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from http import HTTPStatus
from uuid import UUID

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.routing import Match, Route

from cowork.api.v1.artifact_preview import HTML_SANDBOX_CSP, wants_comment_layer
from cowork.api.v1.artifact_scope import review_artifact_for_request
from cowork.api.v1.endpoints.artifact_workspace import (
    _DRAFT_RESPONSE_HEADERS,
    _draft_file_parts,
    _draft_file_response,
    _draft_selectors,
)
from cowork.common.request_logging import RequestLoggingMiddleware
from cowork.common.settings.app_settings import default_draft_env_label, get_app_settings
from cowork.db.scoped import ScopedSession, scope_for_identity
from cowork.db.session import get_open_session
from cowork.draft_host.token import DraftTokenError, verify_draft_token

logger = logging.getLogger(__name__)

EXPIRED_MESSAGE = "This preview link has expired. Open the preview again."

_EXPIRED_PAGE = (
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    "<title>Preview link expired</title></head>"
    '<body style="font-family: system-ui, sans-serif; margin: 2rem; color: #333;">'
    f"<p>{EXPIRED_MESSAGE}</p></body></html>\n"
)
# Replaced or dropped on every response: nothing on this host sets a cookie,
# and the one CSP is ours alone.
_REMOVED_RESPONSE_HEADERS = (
    "set-cookie",
    "clear-site-data",
    "service-worker-allowed",
    "content-security-policy",
    "content-security-policy-report-only",
    "access-control-allow-credentials",
    "x-artifact-grant",
)
# The route template the request log names; the path itself carries the token.
DRAFT_ROUTE = "/_t/{token}/{rel_path:path}"


def draft_csp(web_origin: str) -> str:
    """The draft host's one Content-Security-Policy.

    The sandbox is the preview iframe's, without allow-same-origin: the host is
    shared by every user, so a draft must not get its origin. The Cloudflare
    worker in front of the host sets the same value at the edge.
    """
    ancestors = web_origin.rstrip("/") or "'none'"
    return f"{HTML_SANDBOX_CSP}; connect-src 'self'; frame-ancestors {ancestors}"


def draft_response_headers(web_origin: str) -> dict[str, str]:
    """Headers every draft-host response carries, whatever its status."""
    return {
        **_DRAFT_RESPONSE_HEADERS,
        "Content-Security-Policy": draft_csp(web_origin),
        "Referrer-Policy": "no-referrer",
        "Origin-Agent-Cluster": "?1",
        # The frame's origin is opaque (no allow-same-origin), so its own module
        # scripts and fonts are cross-origin reads. No credentials: the token in
        # the path is the only pass, and whoever holds it may read the files.
        "Access-Control-Allow-Origin": "*",
    }


def request_host(scope) -> str:
    """The request's Host header, lower-cased, without port or trailing dot.

    Only Host: the draft Ingress pins it with upstream-vhost, while
    X-Forwarded-Host is whatever the client sent.
    """
    for name, value in scope.get("headers") or ():
        if name == b"host":
            host = value.decode("latin-1").strip().lower()
            break
    else:
        return ""
    if host.startswith("["):
        end = host.find("]")
        host = host[: end + 1] if end != -1 else host
    elif ":" in host:
        host = host.rsplit(":", 1)[0]
    return host.rstrip(".")


class _DraftHeadersMiddleware:
    """Stamp the draft host's headers on every response, 404 and 405 included.

    Pure ASGI so a streamed file is not buffered. The Cloudflare worker sets
    the same headers at the edge, but the ingress can be reached without it.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        fixed = draft_response_headers(get_app_settings().web_origin)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=list(message.get("headers") or []))
                for name in _REMOVED_RESPONSE_HEADERS:
                    del headers[name]
                for name, value in fixed.items():
                    headers[name] = value
                message = {**message, "headers": headers.raw}
            await send(message)

        await self.app(scope, receive, send_with_headers)


@dataclass(frozen=True)
class _DraftGrant:
    user_id: str
    org_id: str
    project_ref: str
    artifact_id: str


def _refuse(request: Request, reason: str) -> None:
    """Name the refusal for the request's one log line (request_logging.py).

    The reason only: the token and the path that carries it never reach a log.
    """
    request.state._request_log_reason = reason


def _grant_from_request(request: Request) -> _DraftGrant | None:
    """The identity the token carries, or None when it counts as absent.

    A token is honoured only on a request that carries ``Sec-Fetch-Site``.
    Link unfurlers and mail link scanners fetch a pasted link without Fetch
    Metadata and must not get the draft; every current browser sends the
    header.
    """
    if "sec-fetch-site" not in request.headers:
        _refuse(request, "draft_no_fetch_metadata")
        return None
    settings = get_app_settings()
    try:
        claims = verify_draft_token(
            request.path_params["token"],
            env=default_draft_env_label(),
            keys=settings.draft_link_keyring(),
        )
        return _DraftGrant(
            user_id=str(UUID(claims["sub"])),
            org_id=str(UUID(claims["org"])),
            project_ref=str(UUID(claims["proj"])),
            artifact_id=UUID(claims["art"]).hex,
        )
    except DraftTokenError as exc:
        _refuse(request, f"draft_token_{exc.reason}")
    except (ValueError, TypeError, AttributeError):
        _refuse(request, "draft_token_identity")
    return None


def _unauthorized(request: Request) -> Response:
    if request.headers.get("sec-fetch-mode") == "navigate":
        return HTMLResponse(_EXPIRED_PAGE, status_code=401)
    return PlainTextResponse("Unauthorized", status_code=401)


def _draft_response(request: Request, grant: _DraftGrant) -> Response:
    """Resolve and open the file as the token's user. Runs in a worker thread."""
    scope = scope_for_identity(org_id=grant.org_id, user_id=grant.user_id)
    project_selector, artifact_selector = _draft_selectors(grant.project_ref, grant.artifact_id)
    raw_session = get_open_session()
    try:
        source, folder, metadata, _is_own = review_artifact_for_request(
            ScopedSession(raw_session, scope), project_selector, artifact_selector
        )
    finally:
        raw_session.close()
    parts = _draft_file_parts(request.path_params["rel_path"], metadata)
    return _draft_file_response(source, folder, parts, comments=wants_comment_layer(request))


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


async def _serve_draft(request: Request) -> Response:
    settings = get_app_settings()
    if not settings.draft_links_enabled():
        _refuse(request, "draft_links_disabled")
        return PlainTextResponse("Not Found", status_code=404)
    if request_host(request.scope) not in settings.draft_host_names():
        # Fail closed: the draft Ingress pins Host, so any other Host did not
        # arrive the way a draft request does.
        _refuse(request, "draft_host_unknown")
        return PlainTextResponse("Not Found", status_code=404)
    if "download" in request.query_params:
        # Downloads stay on the authenticated API; this host only previews.
        return PlainTextResponse("Bad Request", status_code=400)
    grant = _grant_from_request(request)
    if grant is None:
        return _unauthorized(request)
    try:
        return await run_in_threadpool(_draft_response, request, grant)
    except HTTPException as exc:
        return PlainTextResponse(_status_phrase(exc.status_code), status_code=exc.status_code)
    except Exception:
        logger.exception("Draft file failed", extra={"event": "draft_host_failed"})
        return PlainTextResponse("Internal Server Error", status_code=500)


class _NamedRoute(Route):
    """A route that puts itself on the scope, as FastAPI's APIRoute does.

    RequestLoggingMiddleware names the request by ``scope["route"].path``: the
    template, never the path that carries the token. A plain Starlette route
    leaves it unset.
    """

    def matches(self, scope):
        match, child_scope = super().matches(scope)
        if match != Match.NONE:
            child_scope["route"] = self
        return match, child_scope


class _DraftApp(Starlette):
    def build_middleware_stack(self):
        # Outside ServerErrorMiddleware, as RequestLoggingApp does for the API,
        # so a generated 500 carries x-request-id and its log line too.
        return RequestLoggingMiddleware(super().build_middleware_stack())


def create_draft_app() -> Starlette:
    """The draft host: ``GET/HEAD /_t/{token}/{rel_path}``, 404 for anything else."""
    app = _DraftApp(
        routes=[_NamedRoute(DRAFT_ROUTE, _serve_draft, methods=["GET", "HEAD"])],
        middleware=[Middleware(_DraftHeadersMiddleware)],
    )
    # No slash redirects: the router would answer ``/_t/<token>`` with a 307
    # to the client's Host before the route runs, skipping its Host check.
    app.router.redirect_slashes = False
    return app
