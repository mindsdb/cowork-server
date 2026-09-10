"""ENG-2094: every route declares a Permission via require(...).

This is the CI walker the permission front door was built for (see
require()'s docstring in cowork/api/v1/permissions.py) — create_app() runs
the same check at boot and refuses to start on a gap.
"""
from cowork.api.v1.route_walker import route_key, undeclared_routes
from cowork.server import create_app


def test_every_route_declares_a_permission():
    gaps = [route_key(route) for route in undeclared_routes(create_app())]
    assert not gaps, f"routes with no declared Permission: {sorted(gaps)}"
