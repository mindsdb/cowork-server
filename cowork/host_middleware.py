"""Host header check for a local server running without bearer-token auth.

A DNS-rebound page reaches the server under the attacker's own hostname, and
the browser sends that name as Host. With no token in front of the API, the
Host is the only part of such a request that gives it away, so only names a
page cannot be served from are accepted: localhost, IP literals, and the
names an operator lists in COWORK_ALLOWED_HOSTS.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Collection, Iterable
from urllib.parse import urlparse

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

_TRUSTED_HOSTS = frozenset({"localhost"})


def _hostname(value: str) -> str:
    try:
        return (urlparse(f"//{value}").hostname or "").casefold()
    except ValueError:
        return ""


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


class TrustedHostMiddleware:
    """Refuse requests whose Host a rebinding page could have sent.

    Registered in create_app() only in local tenancy with COWORK_REQUIRE_AUTH
    off; with a token, or behind the org ingress, that auth is the boundary.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: Iterable[str] = (),
        exempt_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self._allowed_hosts = frozenset(_hostname(host) for host in allowed_hosts) - {""}
        # Signed channel webhooks, the same set the bearer layer exempts.
        # create_app keeps filling it while mounting routes, before any request.
        self._exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] not in ("http", "websocket")
            or scope["path"] in self._exempt_paths
            or self._trusted(Headers(scope=scope).get("host", ""))
        ):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
            return
        await JSONResponse({"detail": "untrusted host"}, status_code=403)(scope, receive, send)

    def _trusted(self, host: str) -> bool:
        # Browsers always send Host, so a request without one is not a page's.
        if not host:
            return True
        hostname = _hostname(host)
        return (
            hostname in _TRUSTED_HOSTS
            or hostname in self._allowed_hosts
            or _is_ip_literal(hostname)
        )
