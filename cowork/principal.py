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

import binascii
import json
import logging
from base64 import urlsafe_b64decode
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

# Keep in sync with auth-service and mindshub_inference (minds/common/constants.py).
HEADER_USER_ID = "X-User-Id"
HEADER_ORG_ID = "X-Organization-Id"
HEADER_USER_EMAIL = "X-User-Email"
HEADER_USER_ROLES = "X-User-Roles"
HEADER_EXPECTED_ORG_ID = "X-Cowork-Expected-Organization-Id"
HEADER_ORG_RELOAD = "X-Cowork-Organization-Reload"

OrganizationBoundaryMode = Literal["audit", "enforce"]

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
        organization_boundary_mode: OrganizationBoundaryMode = "enforce",
    ) -> None:
        super().__init__(app)
        self._exempt_paths = exempt_paths
        self._enforce = enforce
        self._organization_boundary_mode = organization_boundary_mode

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

        boundary_response = self._organization_boundary_response(request, org_id)
        if boundary_response is not None:
            return boundary_response
        return await call_next(request)

    def _organization_boundary_response(
        self, request: Request, trusted_org_id: str
    ) -> JSONResponse | None:
        """Fence stale canonical-web documents against the trusted principal.

        The ingress has already authenticated the credential and supplied
        ``trusted_org_id``. The bearer shape is used only to identify Keycloak
        browser sessions; it never supplies identity. API keys and opaque
        service credentials remain compatible with existing callers.
        """
        if not _is_browser_jwt_request(request):
            return None

        supplied = request.headers.get(HEADER_EXPECTED_ORG_ID)
        status_code: int | None = None
        reason: str | None = None
        if supplied is None or not supplied.strip():
            status_code = 426
            reason = "missing expected organization"
        else:
            try:
                expected_org_id = str(UUID(supplied.strip()))
            except ValueError:
                status_code = 409
                reason = "malformed expected organization"
            else:
                if expected_org_id != trusted_org_id:
                    status_code = 409
                    reason = "organization mismatch"

        if status_code is None or reason is None:
            return None

        logger.warning(
            "organization boundary: %s on %s %s (%s mode)",
            reason,
            request.method,
            request.url.path,
            self._organization_boundary_mode,
        )
        if self._organization_boundary_mode == "audit":
            return None
        return JSONResponse(
            {
                "code": "organization_reload_required",
                "detail": "Reload Cowork to continue in the active organization.",
            },
            status_code=status_code,
            headers={
                "Cache-Control": "no-store",
                HEADER_ORG_RELOAD: "required",
            },
        )


def _has_jose_header(segment: str) -> bool:
    """True when a token's first segment decodes to a JOSE header object.

    Read the segment rather than measuring it. A compact header such as
    ``{"alg":"RS256"}`` encodes to exactly 20 characters, so a length threshold
    classifies a real signed token as an opaque credential and silently skips
    the boundary for it.
    """
    padding = "=" * (-len(segment) % 4)
    try:
        header = json.loads(urlsafe_b64decode(segment + padding))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return False
    return isinstance(header, dict) and isinstance(header.get("alg"), str) and bool(header["alg"])


def _is_browser_jwt_request(request: Request) -> bool:
    """True only for a bearer shaped like a browser Keycloak JWT.

    Authentication remains the ingress gateway's job; this only decides which
    callers take part in the browser reload protocol. MindsDB API keys carry
    the ``mdb_`` prefix, and every other opaque service credential fails the
    JOSE header check, so both keep their existing behavior.
    """
    authorization = request.headers.get("Authorization", "").strip()
    if not authorization.lower().startswith("bearer "):
        return False
    credential = authorization[7:].strip()
    if not credential or credential.lower().startswith("mdb_"):
        return False
    parts = credential.split(".")
    return len(parts) == 3 and all(parts) and _has_jose_header(parts[0])


def get_principal(request: Request) -> Principal | None:
    """FastAPI dependency: the request's Principal, or None in local mode."""
    return getattr(request.state, "principal", None)


# Where the desktop shell puts the caller's MindsHub credential, because it
# cannot use Authorization. See `hub_credential`.
HEADER_HUB_CREDENTIAL = "X-MindsHub-Authorization"


def hub_credential(request: Request | None) -> str:
    """The caller's MindsHub credential, for a read this server makes as them.

    Prefers ``X-MindsHub-Authorization`` over ``Authorization``, and the reason is
    the desktop shell. Electron's main process overwrites ``Authorization`` on
    every request to the loopback server with the server's own token, as a plain
    assignment in its ``onBeforeSendHeaders`` hook, so the caller's Keycloak JWT
    can never arrive under that name there. The web shell has no such hook and
    its ``Authorization`` is the JWT, which the fallback covers.

    **A credential, not an identity.** Nothing here decides who the caller is:
    this value is opaque to us and only the service it is forwarded to can
    verify it. That is why accepting it from a client-set header is safe, while
    accepting an identity from one would not be. A caller can present their own
    token or a useless one; neither buys them anything they did not already have.

    Deliberately separate from ``caller_bearer``. Folding the header preference
    into that one would let any client steer the credential on the org model
    catalog fetch too, which reads ``Authorization`` because in org mode the
    gateway put it there.

    **The fallback is org mode only, and that is a containment rule rather than a
    tidiness one.** In org mode the ingress put the caller's own JWT in
    ``Authorization``, so reading it is correct. On a desktop install nothing
    does: the Electron main process assigns THIS server's bearer to that header
    on every loopback request, so falling back there would forward our own
    credential to auth. It would be refused, and it would still have left the
    machine, which is exactly what the main process's own "scoped to our loopback
    origin so it can't leak elsewhere" is supposed to prevent. Empty is the right
    answer instead: the caller has no MindsHub credential, and the surfaces this
    feeds are all fail-closed.
    """
    if request is None:
        return ""
    explicit = request.headers.get(HEADER_HUB_CREDENTIAL, "")
    if explicit.lower().startswith("bearer "):
        return explicit[7:].strip()
    if get_app_settings().tenancy_mode != "org":
        return ""
    return caller_bearer(request)


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
