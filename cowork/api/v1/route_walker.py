"""ENG-2094: enumerate routes with no declared Permission.

``require()`` (permissions.py) stamps ``permission_cls`` onto the dependency
it returns specifically so this can tell "this route called require(...)"
apart from an ordinary data dependency like ``Depends(get_principal)``.

Used both at boot (``create_app()`` refuses to start with a gap) and in CI
(``tests/test_route_walker.py``).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.routing import APIRoute

RouteKey = tuple[str, tuple[str, ...]]


def undeclared_routes(app: FastAPI) -> list[APIRoute]:
    """Every ``APIRoute`` on ``app`` with no ``require(...)``-declared Permission.

    Checks ``route.dependant.dependencies`` rather than the endpoint
    function's own signature: a router-level ``dependencies=[...]`` (the
    common case in this codebase) lands there too, never in the function's
    parameters.
    """
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute)
        and not any(
            getattr(dep.call, "permission_cls", None) is not None
            for dep in route.dependant.dependencies
        )
    ]


def route_key(route: APIRoute) -> RouteKey:
    return (route.path, tuple(sorted(route.methods)))
