"""ENG-2094: the set of routes that answer with no credential at all, pinned.

Ported from auth's ``tests/unit/test_permission_declarations.py``
(``_OPEN_BY_DESIGN_REASONS``, ENG-1563), which is the one mechanism of the
three sibling front doors cowork-server did not have. The declaration in the
route file says a route is open; nothing until now said the SET of open routes
was reviewed. A pin closes that: route 13 cannot arrive without a diff that
makes someone write down why, in the same commit, in a file whose only job is
to be read.

It also settles what a walker cannot see. FastAPI ADDS a router-level
``dependencies=[...]`` to each route's own, so a route added to a router that
declares ``OpenByDesign`` inherits the marker and the walker reports no gap —
it is declared, just not by its author. Set equality catches that where
``undeclared_routes`` structurally cannot.

Assert in BOTH directions on purpose: an entry with no route is a stale pin
that has stopped protecting anything, and a route with no entry is the case
this exists for.

Only ``OpenByDesign`` is pinned. The other non-identity declarations
(``LoopbackOnly``, ``DesktopOnly``, ``LoopbackDesktopOnly``,
``PlatformSignature``) each carry a check that refuses, so a route inheriting
one of those from a router inherits a closed door. ``OpenByDesign`` is the
only declaration where inheriting it is the hazard.
"""
from __future__ import annotations

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.api.v1.permissions import OpenByDesign
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.server import create_app

#: (path, methods) -> why this answers with no credential of any kind.
#:
#: The reason has to stand on its own. "TrustedHeaderMiddleware checks it" is
#: not a reason: it describes another component's current configuration, not
#: anything true about the route. If the route would be a problem the moment
#: that middleware were removed, it does not belong here.
OPEN_BY_DESIGN_REASONS: dict[tuple[str, tuple[str, ...]], str] = {
    ("/api/v1/health/", ("GET",)): (
        "the pre-auth readiness probe every client and the kubelet polls before "
        "anything can authenticate"
    ),
    ("/api/v1/health/live", ("GET",)): (
        "the kubelet's liveness probe, which has no identity headers to send "
        "and nowhere to get them; refusing it kills the pod"
    ),
    ("/api/v1/connectors/specs/{connector_id}", ("GET",)): (
        "a static connector-registry lookup; the same answer for every caller, "
        "no tenant data and no secret"
    ),
    ("/api/v1/connectors/specs/match", ("POST",)): (
        "a stateless token match against the same static registry"
    ),
    ("/api/v1/connectors/oauth/{service}/callback", ("GET",)): (
        "the OAuth provider's own redirect target; the browser arrives with "
        "code/state and no principal exists by construction"
    ),
    ("/api/v1/responses/", ("OPTIONS",)): (
        "a hardcoded CORS-preflight body; an OPTIONS request never carries "
        "identity headers, so requiring any would refuse every preflight"
    ),
    (
        "/api/v1/artifacts/proxy/{token}/{rel_path:path}",
        ("DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"),
    ): (
        "the credential is the mount token in the URL; _PREVIEW_MOUNTS is only "
        "written by /preview-mount, which is DesktopOnly, so in org mode the "
        "dict is empty and every token 404s"
    ),
    ("/api/v1/settings/logout", ("POST",)): (
        "a no-op in org mode; the desktop half is loopback-gated by "
        "require_local_in_desktop_mode"
    ),
    ("/api/v1/settings/install-status", ("GET",)): "a hardcoded stub; there is nothing to expose",
    ("/api/v1/integrations", ("GET",)): "a hardcoded stub returning an empty list",
    ("/api/v1/integrations/{service}/oauth/start", ("POST",)): (
        "a hardcoded stub that ignores its input and answers 'not yet available'"
    ),
    ("/api/v1/scratchpad/cancel", ("POST",)): "a hardcoded stub",
    ("/api/v1/browse/status", ("GET",)): "a hardcoded stub reporting the feature is off",
}


@pytest.fixture(scope="module")
def open_routes() -> set[tuple[str, tuple[str, ...]]]:
    app = create_app()
    found = set()
    for route in app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if {cls for cls in declared_permissions(route)} != {OpenByDesign}:
            continue
        path, methods = route_key(route)
        found.add((path, tuple(m for m in methods if m != "HEAD")))
    return found


def test_no_route_is_open_without_a_written_reason(open_routes):
    unpinned = sorted(open_routes - set(OPEN_BY_DESIGN_REASONS))
    assert unpinned == [], (
        "these routes answer with no credential and no recorded reason. Add an "
        f"entry to OPEN_BY_DESIGN_REASONS saying why, or declare a Permission "
        f"that refuses: {unpinned}"
    )


def test_no_reason_outlives_its_route(open_routes):
    stale = sorted(set(OPEN_BY_DESIGN_REASONS) - open_routes)
    assert stale == [], (
        "these pins name a route that is no longer open (or no longer exists). "
        f"Delete the entry so the pin keeps meaning something: {stale}"
    )


def test_every_reason_stands_without_naming_a_layer_in_front(open_routes):
    """The criterion, enforced rather than described.

    OpenByDesign's docstring rules out "some other layer checks it" as a
    reason. Left to review, that is the reason people reach for: an earlier
    draft of this PR classified 45 routes open on exactly that basis.
    """
    forbidden = ("middleware", "TrustedHeader", "identity_enforce", "exempt")
    offenders = {
        key: reason
        for key, reason in OPEN_BY_DESIGN_REASONS.items()
        if any(word.casefold() in reason.casefold() for word in forbidden)
    }
    assert offenders == {}, (
        "a reason that points at another layer describes that layer's current "
        f"configuration, not this route: {offenders}"
    )


def test_the_open_set_is_small_enough_to_read(open_routes):
    """A number someone notices moving.

    mindshub_services pins its own route count for the same reason. The bound
    is deliberately loose — it is a tripwire for a bulk reclassification, not
    a budget to spend down.
    """
    assert len(open_routes) <= 20, (
        f"{len(open_routes)} routes now answer with no credential. If that is "
        "right, raise the bound in the same commit that explains why."
    )
