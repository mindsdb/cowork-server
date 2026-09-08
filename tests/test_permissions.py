"""Behaviour of the ENG-2094 permission front door's two starting primitives.

Exercises ``require(Open)``/``require(Authenticated)`` on a tiny scratch app
rather than ``create_app()``, so this doesn't depend on the settings
singleton or ``TrustedHeaderMiddleware`` — a route setting
``request.state.principal`` directly is enough to stand in for it.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.permissions import Authenticated, Open, require
from cowork.principal import Principal

PRINCIPAL = Principal(user_id="u1", org_id="o1")


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/open", dependencies=[Depends(require(Open))])
    def open_route():
        return {"ok": True}

    @app.get("/authenticated")
    def authenticated_route(principal: Principal = Depends(require(Authenticated))):
        return {"user_id": principal.user_id}

    return app


def _client() -> TestClient:
    return TestClient(_app())


def test_open_route_allows_no_principal():
    assert _client().get("/open").status_code == 200


def test_authenticated_route_401s_with_no_principal():
    resp = _client().get("/authenticated")
    assert resp.status_code == 401


def test_authenticated_route_allows_a_verified_principal():
    # No TrustedHeaderMiddleware on this scratch app, so stand in for it by
    # overriding the dependency directly rather than setting request.state.
    client = _client()
    client.app.dependency_overrides[require(Authenticated)] = lambda: PRINCIPAL
    resp = client.get("/authenticated")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u1"}


def test_require_stamps_permission_cls_for_the_ci_walker():
    assert require(Open).permission_cls is Open
    assert require(Authenticated).permission_cls is Authenticated


def test_require_is_cached_so_every_call_site_shares_one_dependency():
    assert require(Open) is require(Open)
    assert require(Authenticated) is require(Authenticated)
