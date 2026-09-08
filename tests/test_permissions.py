"""Behaviour of the ENG-2094 permission front door's primitives.

Exercises ``require(Open)``/``require(Authenticated)``/``require(AuthenticatedInOrgMode)``
on a tiny scratch app rather than ``create_app()``, so this doesn't depend on
the settings singleton or ``TrustedHeaderMiddleware`` — a route setting
``request.state.principal`` directly is enough to stand in for it.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.permissions import Authenticated, AuthenticatedInOrgMode, Open, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import Principal, get_principal

PRINCIPAL = Principal(user_id="u1", org_id="o1")
ORG_SCOPE = TenantScope(org_mode=True, org_id="o1", user_id="u1")
LOCAL_SCOPE = TenantScope(org_mode=False)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/open", dependencies=[Depends(require(Open))])
    def open_route():
        return {"ok": True}

    @app.get("/authenticated")
    def authenticated_route(principal: Principal = Depends(require(Authenticated))):
        return {"user_id": principal.user_id}

    @app.get("/authenticated-in-org-mode")
    def authenticated_in_org_mode_route(
        principal: Principal | None = Depends(require(AuthenticatedInOrgMode)),
    ):
        return {"user_id": principal.user_id if principal else None}

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


def test_authenticated_in_org_mode_401s_in_org_mode_with_no_principal():
    client = _client()
    client.app.dependency_overrides[get_tenant_scope] = lambda: ORG_SCOPE
    resp = client.get("/authenticated-in-org-mode")
    assert resp.status_code == 401


def test_authenticated_in_org_mode_allows_a_verified_principal_in_org_mode():
    client = _client()
    client.app.dependency_overrides[get_tenant_scope] = lambda: ORG_SCOPE
    client.app.dependency_overrides[get_principal] = lambda: PRINCIPAL
    resp = client.get("/authenticated-in-org-mode")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u1"}


def test_authenticated_in_org_mode_is_a_no_op_in_local_mode():
    client = _client()
    client.app.dependency_overrides[get_tenant_scope] = lambda: LOCAL_SCOPE
    # No principal override either — local mode never has one.
    resp = client.get("/authenticated-in-org-mode")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": None}


def test_require_stamps_permission_cls_for_the_ci_walker():
    assert require(Open).permission_cls is Open
    assert require(Authenticated).permission_cls is Authenticated
    assert require(AuthenticatedInOrgMode).permission_cls is AuthenticatedInOrgMode


def test_require_is_cached_so_every_call_site_shares_one_dependency():
    assert require(Open) is require(Open)
    assert require(Authenticated) is require(Authenticated)
    assert require(AuthenticatedInOrgMode) is require(AuthenticatedInOrgMode)
