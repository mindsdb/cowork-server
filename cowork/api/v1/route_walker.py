"""ENG-2094: enumerate routes with no declared Permission.

``require()`` (permissions.py) stamps ``permission_cls`` onto the dependency
it returns specifically so this can tell "this route called require(...)"
apart from an ordinary data dependency like ``Depends(get_principal)``.

Used both at boot (``create_app()`` refuses to start with a gap) and in CI
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
        if isinstance(route, _CHECKED_ROUTE_TYPES)
        and not any(
            getattr(dep.call, "permission_cls", None) is not None
            for dep in route.dependant.dependencies
        )
    ]


def route_key(route: APIRoute | APIWebSocketRoute) -> RouteKey:
    methods = getattr(route, "methods", None) or ("WEBSOCKET",)
    return (route.path, tuple(sorted(methods)))
