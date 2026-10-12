"""The request session's database comes from settings, never from the request.

FastAPI reads a dependency's parameters from the request. A parameter on
get_session, which every session route depends on, would let the request pick
the database URI that get_engine opens.
"""
from fastapi.dependencies.utils import get_flat_dependant, get_flat_params
from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.server import create_app


def test_no_route_reads_a_database_uri_from_the_request():
    app = create_app()

    # Path, query, header, cookie and body parameters.
    routes_reading_db_uri = sorted(
        route.path
        for route in app.routes
        if isinstance(route, (APIRoute, APIWebSocketRoute))
        and any(
            param.name == "db_uri"
            for param in [*get_flat_params(route.dependant), *get_flat_dependant(route.dependant).body_params]
        )
    )

    assert routes_reading_db_uri == []
