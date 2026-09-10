"""ENG-2094: every route declares a Permission via require(...).

This is the CI walker the permission front door was built for (see
require()'s docstring in cowork/api/v1/permissions.py) — create_app() runs
the same check at boot and refuses to start on a gap.
"""
from fastapi import Depends, FastAPI

from cowork.api.v1.permissions import OpenByDesign, require
from cowork.api.v1.route_walker import undeclared_routes, route_key
from cowork.server import create_app


def test_every_route_declares_a_permission():
    gaps = [route_key(route) for route in undeclared_routes(create_app())]
    assert not gaps, f"routes with no declared Permission: {sorted(gaps)}"


def test_undeclared_routes_flags_a_route_with_no_permission():
    """undeclared_routes itself has no test proving it catches a real gap —
    everything else only asserts the current, fully-wired app has zero. A
    regression here (e.g. a typo'd attribute name) would otherwise go
    unnoticed until a real route slipped through undeclared."""
    app = FastAPI()

    @app.get("/no-permission")
    def _unguarded():
        return {"ok": True}

    @app.get("/declared", dependencies=[Depends(require(OpenByDesign))])
    def _guarded():
        return {"ok": True}

    gaps = {route_key(route) for route in undeclared_routes(app)}

    assert ("/no-permission", ("GET",)) in gaps
    assert ("/declared", ("GET",)) not in gaps


def test_undeclared_routes_covers_websocket_routes_too():
    app = FastAPI()

    @app.websocket("/no-permission-ws")
    async def _unguarded_ws(websocket):
        await websocket.accept()

    @app.websocket("/declared-ws", dependencies=[Depends(require(OpenByDesign))])
    async def _guarded_ws(websocket):
        await websocket.accept()

    gaps = {route_key(route) for route in undeclared_routes(app)}

    assert ("/no-permission-ws", ("WEBSOCKET",)) in gaps
    assert ("/declared-ws", ("WEBSOCKET",)) not in gaps
