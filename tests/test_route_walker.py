"""ENG-2094: every route declares a Permission via require(...), exactly once.

This is the CI walker the permission front door was built for (see
require()'s docstring in cowork/api/v1/permissions.py) — create_app() runs
the same checks at boot and refuses to start on either failure.
"""
import pytest
from fastapi import Depends, FastAPI

from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
from cowork.api.v1.route_walker import contradictory_routes, route_key, undeclared_routes
from cowork.server import create_app


def test_the_app_boots_with_every_route_declared():
    """create_app() itself is the assertion.

    It raises RuntimeError naming the offending routes before it can return,
    so there is no app object to walk when something is wrong — building one
    is the whole test. Walking the result too would only re-check what
    create_app() already refused to let past.
    """
    assert create_app() is not None


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


def test_contradictory_routes_flags_open_declared_next_to_a_closed_one():
    """A route-level dependency is ADDED to its router's, never substituted.

    So OpenByDesign on a route whose router declares AuthenticatedInOrgMode
    leaves the route closed while reading as open — the state
    `OPTIONS /api/v1/responses/` shipped in, answering 401 in org mode under a
    comment saying it carried "its own OpenByDesign instead".
    """
    app = FastAPI()
    router_deps = [Depends(require(AuthenticatedInOrgMode))]

    @app.get("/contradicted", dependencies=router_deps + [Depends(require(OpenByDesign))])
    def _contradicted():
        return {"ok": True}

    @app.get("/open", dependencies=[Depends(require(OpenByDesign))])
    def _open():
        return {"ok": True}

    @app.get("/closed", dependencies=router_deps)
    def _closed():
        return {"ok": True}

    flagged = {route_key(route) for route in contradictory_routes(app)}

    assert ("/contradicted", ("GET",)) in flagged
    assert ("/open", ("GET",)) not in flagged
    assert ("/closed", ("GET",)) not in flagged


def test_the_gap_message_names_a_websocket_route_instead_of_raising():
    """The boot message used route.methods, which APIWebSocketRoute lacks.

    An undeclared websocket route answered with `AttributeError:
    'APIWebSocketRoute' object has no attribute 'methods'` over the top of the
    message that was supposed to name it — losing the diagnostic for exactly
    the route type the walker was extended to cover.
    """
    app = FastAPI()

    @app.websocket("/undeclared-ws")
    async def _undeclared_ws(websocket):
        await websocket.accept()

    gaps = undeclared_routes(app)
    assert gaps

    # The formatting create_app() does with the walk's output.
    rendered = ", ".join(f"{list(methods)} {path}" for path, methods in map(route_key, gaps))
    assert "/undeclared-ws" in rendered
    assert "WEBSOCKET" in rendered


def test_create_app_refuses_to_boot_on_a_gap(monkeypatch):
    """AC2 is a boot property, not just a CI one. Prove the raise fires."""
    from cowork.api.v1 import route_walker

    sentinel = FastAPI()

    @sentinel.get("/smuggled")
    def _smuggled():
        return {"ok": True}

    monkeypatch.setattr(
        "cowork.server.undeclared_routes", lambda app: list(sentinel.routes)
    )
    with pytest.raises(RuntimeError, match="no declared Permission"):
        create_app()
    assert route_walker.undeclared_routes  # the real one is untouched
