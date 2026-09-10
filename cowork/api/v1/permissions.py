"""The permission front door: every cowork-server route declares who may
call it here (ENG-2094).

Ported from mindshub_inference's front door (ENG-1564, minds/api/v1/permissions.py)
— same shape, starting with only the primitives every route can already use.
ENG-2094's AC1 wants each route's declaration to be one of three things: a
capability, an FGA relation, or an explicit public marker carrying its own
one-line reason (``OpenByDesign``, below). Bare identity — ``Authenticated``/
``AuthenticatedInOrgMode`` — is none of those three; it is what a route gets
before its real capability check exists, not a resting place. A capability
check against auth's now-live entitlements (``GET /v1/entitlements/me/``,
ENG-2088) is what those routes actually need to move to.

``require(permission)`` turns a ``Permission`` class into a FastAPI dependency
a route declares directly, e.g. ``principal: Principal = Depends(require(Authenticated))``,
instead of a route registering into a separate path-keyed table. Pass the
class itself, not an instance: neither check takes any config, and a class is
already a stable, hashable object (the same one on every import), so
``require`` can be plain ``lru_cache``d and every ``require(Authenticated)``
call anywhere returns the identical dependency function — that's what lets
FastAPI's own per-request dependency cache collapse repeats.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Protocol

from fastapi import Depends, HTTPException, Request, status

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, can_manage_org, get_principal


class Permission(Protocol):
    """Something a route can pass to ``require(...)``."""

    async def check(self, request: Request) -> object:
        """Raise ``HTTPException`` to deny. Return the resolved identity to
        allow — ``OpenByDesign`` returns ``None`` since there is nothing to
        resolve."""
        ...


class OpenByDesign:
    """No identity required — and that has to be true on its own, not
    because some other layer in front of the route happens to check it.

    Each use site needs a one-line reason that stands without reference to
    ``TrustedHeaderMiddleware`` (cowork/principal.py) or any other layer: a
    health probe infra must reach with zero auth, a webhook verified by its
    own signature, a route whose real credential is a token in the URL
    rather than a principal. "the middleware already checks this" is not a
    valid reason. That was this class's original criterion, and it
    misclassified routes that merely lacked their own check yet: the
    middleware only guarantees identity when ``identity_enforce ==
    "enforce"``, so justifying ``OpenByDesign`` that way describes a fact
    about a *different component's* current configuration, not anything
    true about the route itself. If the route would be a problem the moment
    the middleware were removed or misconfigured, it is not
    ``OpenByDesign`` — it needs ``Authenticated``/``AuthenticatedInOrgMode``
    or a capability check instead.

    Declaring this — rather than leaving a route with no permission
    dependency at all — is what lets the CI route walker (added once every
    route carries a declaration) tell "open on purpose" apart from "open
    because someone forgot the line"; both look identical at runtime
    without it.
    """

    async def check(self, request: Request) -> None:
        return None


class Authenticated:
    """A verified ``Principal`` is required; no further check.

    401s when the caller has no principal — no identity headers in org
    enforce mode, or audit/local mode where none was ever built. Use this
    for a route that is inherently a multi-tenant concept (e.g. the
    organization-switch capability) rather than one shared with desktop's
    single-user local mode, which never has a principal to check.

    Takes ``principal`` as a declared ``Depends(get_principal)`` rather than
    calling that function directly on ``request``: a subclass overriding
    ``check`` for its own reason (see ``NoStoreAuthenticated``) still gets
    ``principal`` resolved through FastAPI's own dependency graph, which is
    what lets a test's ``app.dependency_overrides[get_principal]`` reach it —
    a bare function call bypasses that entirely.
    """

    async def check(self, request: Request, principal: Principal | None = Depends(get_principal)) -> Principal:
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        return principal


class AuthenticatedInOrgMode(Authenticated):
    """``Authenticated`` in org mode; a no-op in local mode.

    Reads ``get_app_settings().tenancy_mode`` directly rather than depending
    on ``get_tenant_scope``: several tests build a minimal app around one
    router and override ``get_tenant_scope`` to ``None`` for concerns that
    have nothing to do with tenancy (e.g. ``tests/test_coding_service.py``),
    which would crash this check on ``None.org_mode`` if it shared that
    dependency.

    Enforces on ``tenancy_mode`` alone, not ``identity_enforce``: every real
    deployment (dev/staging/prod) pins ``identity_enforce=enforce``, so the
    org-mode audit rollout where this would be stricter than the middleware
    is not a state any of them run in today. Revisit if that changes.

    A floor, not a finish line: ENG-2094's AC1 wants a capability, an FGA
    relation, or a public marker — bare identity is none of those. Now that
    ENG-2088's entitlements are live (``GET /v1/entitlements/me/``), a route
    declared with this should eventually move to a real capability check;
    this is what a route gets before that check exists.
    """

    async def check(
        self, request: Request, principal: Principal | None = Depends(get_principal)
    ) -> Principal | None:
        if get_app_settings().tenancy_mode != "org":
            return None
        return await super().check(request, principal=principal)


class AuthenticatedOrgAdmin(AuthenticatedInOrgMode):
    """``AuthenticatedInOrgMode``, plus org-admin standing in org mode.

    Configuring a shared resource on behalf of the whole org (channels,
    org-scoped settings) is admin-owned; a no-op in local mode, same split as
    ``AuthenticatedInOrgMode``.
    """

    async def check(
        self, request: Request, principal: Principal | None = Depends(get_principal)
    ) -> Principal | None:
        principal = await super().check(request, principal=principal)
        if get_app_settings().tenancy_mode == "org" and not can_manage_org(principal):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="requires an org admin")
        return principal


@cache
def require(permission_cls: type[Permission]):
    """Turn a ``Permission`` class into a FastAPI dependency.

    Stamps ``permission_cls`` onto the returned dependency so a route's
    declaration is visible by inspecting ``route.dependant.dependencies``
    directly — the CI walker needs a way to tell "this route called
    ``require(...)``" apart from an ordinary data dependency like
    ``Depends(get_principal)``, and this is the one place that can mark that
    without a second, hand-maintained table that could drift from what's
    actually wired.

    ``dependency``'s own ``__signature__`` is replaced with ``check``'s
    (minus ``self``): FastAPI decides what to resolve for a dependency by
    inspecting *that callable's* signature, not the ``Permission`` class
    behind it. Copying it now — even though ``OpenByDesign``/``Authenticated``
    only ever need ``request`` — is what lets a later ``Permission`` (a typed
    body model, a nested ``Depends(...)``) be resolved by FastAPI without
    this function changing.
    """
    permission = permission_cls()
    check_params = [p for name, p in inspect.signature(permission_cls.check).parameters.items() if name != "self"]

    async def dependency(**resolved):
        return await permission.check(**resolved)

    dependency.__signature__ = inspect.Signature(check_params)
    dependency.permission_cls = permission_cls
    return dependency
