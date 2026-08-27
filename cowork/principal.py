"""Request principal for org (multi-tenant) deployments.

When COWORK_TENANCY_MODE=org the server runs behind the MindsHub auth
gateway, which validates the caller's JWT or API key and injects trusted
identity headers:

    X-User-Id          Keycloak user UUID          (required)
    X-Organization-Id  active organization UUID    (required)
    X-User-Email       user email                  (optional)
    X-User-Roles       comma-separated role names  (optional)

TrustedHeaderMiddleware turns those headers into a Principal on
``request.state.principal``. A request without a valid pair is rejected
with 401, and is only logged and let through where an operator has asked
for that by setting COWORK_IDENTITY_ENFORCE=audit, the mode the org
cutover rolled out behind. Identity is never derived from anything a
client can set directly, only from what the gateway injected after
verification. That last sentence holds only while the gateway is the only
route to the pod, which is a network question this file cannot answer:
enforce mode refuses a caller carrying no identity, not one carrying a
well-formed forged pair.

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
    With enforce=True requests without valid identity are rejected with 401;
    with enforce=False (audit rollout) they are logged and allowed through.
    """

    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: Collection[str] = (),
        enforce: bool = True,
    ) -> None:
        super().__init__(app)
        self._exempt_paths = exempt_paths
        self._enforce = enforce

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
            if self._enforce:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            logger.warning(
                "identity: no principal on %s %s (audit mode)",
                request.method,
                request.url.path,
            )
            return await call_next(request)

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


def caller_bearer(request: Request | None) -> str:
    """The caller's own bearer token, for a read this server makes on their behalf.

    Every outbound MindsHub call that acts as the caller sends this and never the
    stored provider key or ``minds_url``: both of those are tenant-settable, so
    using either would let an org admin point a member's credential at a host
    they chose. Empty string when there is no bearer, which callers turn into
    their fail-closed answer rather than an anonymous request.

    This is not an identity source. Nothing here decides who the caller is: in
    org mode that is the gateway's injected headers, and this token is opaque to
    us. It exists only to be forwarded to the service that can verify it.
    """
    if request is None:
        return ""
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


# Keycloak org role that marks an organization admin (see auth's
# role_context/authenticate: X-User-Roles carries realm + org roles).
ORG_MANAGE_ROLE = "manage-organization"


def can_manage_org(principal: Principal | None) -> bool:
    """True when the caller may change org-level configuration. No principal
    (local mode never gets here; org audit mode) → False, fail closed."""
    return principal is not None and ORG_MANAGE_ROLE in principal.roles


def identity_trace_metadata(
    principal: Principal | None, base: dict[str, str] | None
) -> dict[str, str] | None:
    """Merge the principal's identity into trace metadata.

    Server-derived identity always wins over client-supplied keys, so a
    caller can't spoof attribution. No principal → base returned unchanged.
    """
    if principal is None:
        return base
    merged = dict(base or {})
    merged["user_id"] = principal.user_id
    merged["organization_id"] = principal.org_id
    return merged
