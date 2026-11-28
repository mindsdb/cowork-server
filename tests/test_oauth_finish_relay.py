"""The web finish of a built-in connector sign-in: relayed to auth on the
member's own credential, with auth's refusal code passed through.

Requests go through the real router; auth is answered at the transport.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import oauth
from cowork.api.v1.permissions import AuthenticatedInOrgMode
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, get_tenant_scope
from cowork.principal import Principal, get_principal
from cowork.services.connectors.oauth import auth_proxy

MEMBER = TenantScope(org_mode=True, org_id="org-a", user_id="member-a")
BEARER = "Bearer sentinel-member-credential"
COMPLETION = "A" * 43


@pytest.fixture()
def auth(monkeypatch):
    """auth answered from ``answers[(method, path)]``; every request it receives is in ``sent``."""
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    get_app_settings.cache_clear()
    state: dict = {"answers": {}, "sent": []}

    def handle(request: httpx.Request) -> httpx.Response:
        state["sent"].append(request)
        status_code, body = state["answers"].get((request.method, request.url.path), (404, {"detail": "Not found."}))
        return httpx.Response(status_code, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        auth_proxy.httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    yield state
    get_app_settings.cache_clear()


def _client(scope: TenantScope = MEMBER) -> TestClient:
    app = FastAPI()
    app.include_router(oauth.router, prefix="/connectors/oauth")
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=scope.user_id, org_id=scope.org_id)
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    return TestClient(app, headers={"Authorization": BEARER})


def test_the_route_is_the_signed_in_members_own():
    [route] = [r for r in api_router.routes if route_key(r) == ("/api/v1/connectors/oauth/finish", ("POST",))]
    assert declared_permissions(route) == [AuthenticatedInOrgMode]


def test_finish_relays_on_the_members_credential(auth):
    answer = {"status": "success", "service": "linear", "name": "a-example-com"}
    auth["answers"][("POST", "/v1/oauth/finish")] = (200, answer)

    response = _client().post("/connectors/oauth/finish", json={"completion": COMPLETION})

    assert response.status_code == 200
    assert response.json() == answer
    [request] = auth["sent"]
    assert json.loads(request.content) == {"completion": COMPLETION}
    assert request.headers["authorization"] == BEARER


@pytest.mark.parametrize(("status_code", "code"), [(403, "login_not_yours"), (409, "wrong_organization"),
                                                   (410, "expired"), (400, "failed")])
def test_auths_refusal_code_passes_through(auth, status_code, code):
    auth["answers"][("POST", "/v1/oauth/finish")] = (status_code, {"error": code, "detail": "No."})

    response = _client().post("/connectors/oauth/finish", json={"completion": COMPLETION})

    assert response.status_code == status_code
    assert response.json() == {"detail": {"code": code, "message": "No."}}


def test_outside_org_mode_nothing_is_relayed(auth):
    response = _client(LOCAL_SCOPE).post("/connectors/oauth/finish", json={"completion": COMPLETION})

    assert response.status_code == 404
    assert auth["sent"] == []
