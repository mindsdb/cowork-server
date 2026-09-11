"""ENG-2094: enumerate routes whose declared Permission is missing or
self-contradictory.

``require()`` (permissions.py) stamps ``permission_cls`` onto the dependency
it returns specifically so this can tell "this route called require(...)"
apart from an ordinary data dependency like ``Depends(get_principal)``.

Two walks, because a declaration fails in two ways. ``undeclared_routes``
finds the route nobody declared. ``contradictory_routes`` finds the route
declared twice in opposite directions, which reads as covered and is not.

Used both at boot (``create_app()`` refuses to start on either) and in CI
(``tests/test_route_walker.py``).

Covers ``APIRoute`` (HTTP) and ``APIWebSocketRoute`` (websocket) — both carry
a ``.dependant.dependencies`` built the same way, router-level
``dependencies=[...]`` included. Does NOT cover a ``Mount``-ed sub-app: it
delegates to an entirely different ASGI app with no uniform way to inspect
its own auth. Nothing in this codebase uses either today (FastAPI's own
``/docs``/``/openapi.json`` are plain ``starlette.routing.Route``, not
``APIRoute``, so they're excluded here without special-casing), so this is a
ceiling to know about if one is ever added, not a gap that bites today.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

RouteKey = tuple[str, tuple[str, ...]]

_CHECKED_ROUTE_TYPES = (APIRoute, APIWebSocketRoute)


def undeclared_routes(app: FastAPI) -> list[APIRoute | APIWebSocketRoute]:
    """Every route on ``app`` with no ``require(...)``-declared Permission.

    Checks ``route.dependant.dependencies`` rather than the endpoint
    function's own signature: a router-level ``dependencies=[...]`` (the
    common case in this codebase) lands there too, never in the function's
    parameters.
    """
    return [
        route
        for route in app.routes
        if isinstance(route, _CHECKED_ROUTE_TYPES) and not declared_permissions(route)
    ]


def declared_permissions(route: APIRoute | APIWebSocketRoute) -> list[type]:
    """Every ``Permission`` class declared on ``route``, router-level first.

    A route can carry more than one: FastAPI ADDS a route-level
    ``dependencies=[...]`` to its router's rather than substituting for it, so
    both run.
    """
    return [
        cls
        for dep in route.dependant.dependencies
        if (cls := getattr(dep.call, "permission_cls", None)) is not None
    ]


def contradictory_routes(app: FastAPI) -> list[APIRoute | APIWebSocketRoute]:
    """Every route declaring ``OpenByDesign`` next to a Permission that denies.

    A route-level ``dependencies=[Depends(require(OpenByDesign))]`` does not
    override the router's ``AuthenticatedInOrgMode``; both dependencies
    resolve and the stricter one still refuses. So the pair always means the
    author believed they were opening a route that stayed closed, and
    ``undeclared_routes`` cannot see it — the route is declared, twice.
    Whichever answer is wanted, saying both is never it.
    """
    from cowork.api.v1.permissions import OpenByDesign

    return [
        route
        for route in app.routes
        if isinstance(route, _CHECKED_ROUTE_TYPES)
        and OpenByDesign in (declared := declared_permissions(route))
        and len(set(declared)) > 1
    ]


def route_key(route: APIRoute | APIWebSocketRoute) -> RouteKey:
    methods = getattr(route, "methods", None) or ("WEBSOCKET",)
    return (route.path, tuple(sorted(methods)))
