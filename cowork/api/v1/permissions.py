"""The permission front door: every cowork-server route declares who may
call it here (ENG-2094).

Ported from mindshub_inference's front door (ENG-1564, minds/api/v1/permissions.py).

**Each class names the credential the route actually takes.** That is the one
rule the vocabulary is built on, and it is why there are seven classes rather
than "open" and "closed":

===========================  =====================================================
``OpenByDesign``             no credential at all. Pinned route by route in
                             tests/test_open_by_design_pin.py.
``LoopbackOnly``             the caller's network position (``require_local``)
``DesktopOnly``              refused in org mode (``require_local_tenancy``)
``LoopbackDesktopOnly``      both of the above
``PlatformSignature``        the calling platform's HMAC over the body
``Authenticated``            a verified ``Principal``
``AuthenticatedInOrgMode``   a verified ``Principal``, in org mode only
``AuthenticatedOrgAdmin``    ... and org-admin standing
===========================  =====================================================

An earlier draft had one open marker covering all five non-identity cases, and
152 of 284 method-routes carried it. Nothing that does this for real collapses
them: Istio splits ``ipBlocks`` from ``requestPrincipals`` from ``principals``,
Envoy splits an empty ``requires`` from ``allow_missing``, OpenAPI gives mTLS
its own scheme type, and inference's own port has ``HeaderIdentity`` and
``SignedTokenIdentity`` beside its open marker. The name is the thing a reader
checks the code against, so a name that says "open" for a route guarded by
loopback cannot be checked at all.

**Inheritance is safe for every class except ``OpenByDesign``.** A route added
to a router that declares one of the restrictive classes inherits a check that
refuses. A route added to a router that declares ``OpenByDesign`` inherits a
marker that permits, and the walker cannot tell that apart from a deliberate
declaration — which is the hazard ASP.NET Core publishes a warning about for
controller-level ``[AllowAnonymous]``. The pin is what covers it.

ENG-2094's AC1 wants each route's declaration to be a capability, an FGA
relation, or an explicit public marker with a reason. Bare identity is none of
those three; it is what a route gets before its real capability check exists,
not a resting place. A capability check against auth's now-live entitlements
(``GET /v1/entitlements/me/``, ENG-2088) is what those 132 routes need next.

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

from cowork.api.v1.endpoints.guards import require_local, require_local_tenancy
from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, can_manage_org, get_principal


class Permission(Protocol):
    """Something a route can pass to ``require(...)``."""

    async def check(self, request: Request) -> object:
        """Raise ``HTTPException`` to deny. Return the resolved identity to
        allow, or ``None`` where the credential is not an identity."""
        ...


class OpenByDesign:
    """No identity required — and that has to be true on its own, not
    because some other layer in front of the route happens to check it.

    The odd name in this file, on purpose. Every other class here is named
    after the credential the route takes: ``LoopbackOnly`` is a network
    position, ``PlatformSignature`` is an HMAC, ``Authenticated`` is a
    ``Principal``. The absence of a credential has no name, so this one is
    named after the assertion instead — that the route is open because
    someone decided it, not because a line went missing. That is also the
    whole job: this class checks nothing and refuses nobody, it exists only
    to be distinguishable from a route nobody declared.

    Keep the name in step with auth, which exported it first (ENG-1563) and
    pins its own open set under it. ENG-2094's premise is one vocabulary
    across the four services, and a fourth service renaming it unilaterally
    is how that stops being true.

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
    dependency at all — is what lets the route walker tell "open on purpose"
    apart from "open because someone forgot the line"; both look identical at
    runtime without it.

    If the answer is "loopback", "desktop-only" or "the platform signed it",
    the class for that is ``LoopbackOnly`` / ``DesktopOnly`` /
    ``PlatformSignature``, not this. This one means no credential at all, and
    every route that carries it is pinned with its reason in
    tests/test_open_by_design_pin.py — declare it here and the build stays red
    until the reason is written down.

    Never declare it on a router. Route-level dependencies are ADDED to
    router-level ones rather than replacing them, so a router carrying this
    hands the marker to every route added later, and the author never chose
    it.
    """

    async def check(self, request: Request) -> None:
        return None


class LoopbackOnly:
    """The credential is the caller's network position, not an identity.

    ``require_local`` (endpoints/guards.py) is the check: a loopback peer, a
    loopback ``Host`` (DNS rebinding), and an ``Origin`` in
    ``allowed_origins`` (CSRF). For a route only the desktop sidecar, its
    renderer, or the Electron main process ever calls.

    Takes it as a declared ``Depends(require_local)`` rather than calling it,
    for the same reason ``Authenticated`` declares ``get_principal``: a bare
    call inside ``check`` is invisible to FastAPI, so a test's
    ``app.dependency_overrides[require_local]`` would stop reaching it.

    Named rather than folded into ``OpenByDesign``: "no principal" and "a
    credential that is not a principal" are different declarations, and every
    system that does this for real keeps them apart — Istio splits
    ``ipBlocks`` from ``requestPrincipals``, Envoy splits an empty ``requires``
    from ``allow_missing``, and mindshub_inference's own front door (ENG-1564)
    has ``HeaderIdentity`` and ``SignedTokenIdentity`` beside its open marker.
    Collapsing them loses the one fact a reader needs: what would have to go
    wrong for this route to be reachable.
    """

    async def check(self, request: Request, _local: None = Depends(require_local)) -> None:
        return None


class DesktopOnly:
    """Refused outright in org mode; desktop is the only place it runs.

    ``require_local_tenancy`` 403s the route whenever
    ``tenancy_mode == "org"``, so there is no multi-tenant identity concept to
    check — not because the route is public, but because the deployment it
    would be public in cannot reach it.
    """

    async def check(self, request: Request, _desktop: None = Depends(require_local_tenancy)) -> None:
        return None


class LoopbackDesktopOnly:
    """``LoopbackOnly`` and ``DesktopOnly`` together, in that order.

    Loopback first, matching the order these two guards were declared in
    before they became one declaration: a non-loopback caller is told "local
    requests only" whatever the tenancy mode, and only a loopback caller
    reaches the org-mode refusal.

    Not a subclass of either — ``require()`` copies one class's ``check``
    signature, so composing by inheritance would mean re-declaring both
    sub-dependencies here anyway, and a flat signature is what a reader can
    check the order against.

    Safe to declare on a router, which is the point. A route added to a router
    carrying this inherits a check that REFUSES; a route added to one carrying
    ``OpenByDesign`` inherits a marker that permits. Inheriting a restrictive
    default is deny-by-default; inheriting a permissive one is the hazard
    ASP.NET Core publishes a warning about, where a controller-level
    ``[AllowAnonymous]`` silently beats an action-level ``[Authorize]``.
    """

    async def check(
        self,
        request: Request,
        _local: None = Depends(require_local),
        _desktop: None = Depends(require_local_tenancy),
    ) -> None:
        return None


class PlatformSignature:
    """The credential is the calling platform's own signature over the body.

    Slack, Discord, Telegram and WhatsApp sign their webhooks with a secret
    this deployment stored when the channel was configured, and
    ``bridge.verify_signature`` (cowork/channels/webhooks.py) rejects an
    unsigned or mis-signed request with 401. There is no Cowork principal in
    the flow by construction: the caller is a platform, not a person.

    A marker, not a check — the verification is in the handler, because it
    needs the raw body and the org the payload routes to, neither of which a
    dependency has. What the name buys over ``OpenByDesign`` is the reader
    knowing a credential exists at all, which is the distinction Envoy draws
    between an empty ``requires`` and ``allow_missing``, and the one
    mindshub_inference draws with ``SignedTokenIdentity``.
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

    Enforces on ``tenancy_mode`` alone and deliberately does NOT read
    ``identity_enforce``. That flag is ``TrustedHeaderMiddleware``'s rollout
    lever, and an authorization layer that softens because the identity layer
    below it is in audit mode has no lever of its own. Every system that ships
    both keeps them separate: Istio's ``PeerAuthentication: PERMISSIVE`` does
    not soften an ``AuthorizationPolicy``, and Envoy's ext_authz
    ``failure_mode_allow`` is a different switch from the RBAC filter's
    ``shadow_rules``.

    The consequence is real and worth stating: in org mode with
    ``identity_enforce=audit`` this is STRICTER than the middleware, so the
    audit rollback lever does not cover anything the front door refuses. No
    deployment runs that state (values-dev, values-staging and values-prod all
    pin ``enforce``), and ``create_app()`` warns at boot if one ever does. If a
    real cutover needs a shadow phase, the answer is a shadow mode on this
    layer — evaluate, record the denial it would have made, allow — not a
    second layer quietly reading the first one's flag.

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

    Reads the forwarded roles, and that is the thing to fix next.
    ``can_manage_org`` asks whether ``"manage-organization"`` is in
    ``principal.roles``, which ``TrustedHeaderMiddleware`` parsed out of the
    ``X-User-Roles`` header. ENG-2094's AC3 names that header as the source a
    declared capability must never be checked against; auth's resolved answer
    (``org_role`` and ``permissions[]`` from ``GET /v1/entitlements/me/``) is.
    So this is a floor in the same sense ``AuthenticatedInOrgMode`` is, not a
    capability check. It is written this way because it replaces
    ``channels.py``'s own ``_require_org_admin``, which asked the same
    question of the same header on the same six routes — moving that into the
    vocabulary changes nothing about who is admitted, and swapping the source
    is its own slice.
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

    ``eval_str=True`` resolves those annotations against the class's OWN
    defining module. Without it the copied parameters carry bare strings
    under ``from __future__ import annotations``, and FastAPI then resolves
    them against ``dependency.__globals__`` — this module — so a
    ``Permission`` subclass living in an endpoint module gets a silent
    ``ForwardRef`` for any name this module does not import: the route
    registers, valid requests answer 422, and ``/openapi.json`` raises. An
    undefined name is now a ``NameError`` at import, which is the failure
    worth having.

    ``__qualname__`` is rewritten because it is the only handle tooling has
    on a closure: ``scripts/dump_routes.py`` reads it to build auth's
    endpoint-authorization map, and every declaration would otherwise
    read ``require.<locals>.dependency``.
    """
    permission = permission_cls()
    check_params = [
        p
        for name, p in inspect.signature(permission_cls.check, eval_str=True).parameters.items()
        if name != "self"
    ]

    async def dependency(**resolved):
        return await permission.check(**resolved)

    dependency.__signature__ = inspect.Signature(check_params)
    dependency.__name__ = dependency.__qualname__ = f"require({permission_cls.__name__})"
    dependency.permission_cls = permission_cls
    return dependency
