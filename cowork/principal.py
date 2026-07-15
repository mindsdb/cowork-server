"""Request principal for org (multi-tenant) deployments.

When COWORK_TENANCY_MODE=org the server runs behind the MindsHub auth
gateway, which validates the caller's JWT or API key and injects trusted
identity headers:

    X-User-Id          Keycloak user UUID          (required)
    X-Organization-Id  active organization UUID    (required)
    X-User-Email       user email                  (optional)
    X-User-Roles       comma-separated role names  (optional)

TrustedHeaderMiddleware turns those headers into a Principal on
``request.state.principal`` and rejects requests that lack the required
pair — identity is never derived from anything a client can set directly,
only from what the gateway injected after verification.

In local mode (the desktop sidecar, the default) the middleware is not
registered and ``request.state.principal`` is absent; ``get_principal``
returns None so shared code can branch on "no tenant context".
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Keep in sync with auth-service and mindshub_inference (minds/common/constants.py).
HEADER_USER_ID = "X-User-Id"
HEADER_ORG_ID = "X-Organization-Id"
HEADER_USER_EMAIL = "X-User-Email"
HEADER_USER_ROLES = "X-User-Roles"

# Always reachable without identity; channel webhooks are added by create_app().
_EXEMPT_PATHS = frozenset({"/api/v1/health", "/api/v1/health/"})


@dataclass(frozen=True)
class Principal:
    """Verified identity of the caller for the duration of one request."""

    user_id: str
    org_id: str
    email: str = ""
    roles: frozenset[str] = field(default_factory=frozenset)


class TrustedHeaderMiddleware(BaseHTTPMiddleware):
    """Build a Principal from gateway-injected identity headers.

    Registered in create_app() only when COWORK_TENANCY_MODE=org.
    Requests without valid identity headers are rejected with 401.
    """

    def __init__(self, app: ASGIApp, exempt_paths: Collection[str] = ()) -> None:
        super().__init__(app)
        self._exempt_paths = exempt_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # CORS preflight never carries identity headers.
        if request.method == "OPTIONS":
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS or request.url.path in self._exempt_paths:
            return await call_next(request)

        # Both ids are Keycloak UUIDs — validate format, normalize case.
        try:
            user_id = str(UUID(request.headers.get(HEADER_USER_ID, "").strip()))
            org_id = str(UUID(request.headers.get(HEADER_ORG_ID, "").strip()))
        except ValueError:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        roles = frozenset(
            role.strip()
            for role in request.headers.get(HEADER_USER_ROLES, "").split(",")
            if role.strip()
        )
        request.state.principal = Principal(
            user_id=user_id,
            org_id=org_id,
            email=request.headers.get(HEADER_USER_EMAIL, "").strip(),
            roles=roles,
        )
        return await call_next(request)


def get_principal(request: Request) -> Principal | None:
    """FastAPI dependency: the request's Principal, or None in local mode."""
    return getattr(request.state, "principal", None)
