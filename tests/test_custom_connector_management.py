"""Admins can rename, unfeature, re-form and delete a custom connector, in their own scope only."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.api.v1.endpoints.connectors.custom import delete_custom_connector, update_custom_connector
from cowork.api.v1.endpoints.connectors.custom import router as custom_router
from cowork.api.v1.permissions import AuthenticatedOrgAdmin
from cowork.api.v1.route_walker import declared_permissions
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, get_scoped_session
from cowork.models.custom_connector import CustomConnector
from cowork.principal import ORG_MANAGE_ROLE, Principal, get_principal
from cowork.schemas.connectors import CustomConnectorUpdate
from cowork.server import create_app
from cowork.services.connectors.catalog import ConnectorCatalog
from tests._fakes import FakeRequest

ORG_A = TenantScope(org_mode=True, org_id="org-a", user_id="admin-a")
ORG_B = TenantScope(org_mode=True, org_id="org-b", user_id="admin-b")

FORM = {
    "form_id": "httpbin-connector",
    "title": "Connect httpbin",
    "fields": [{"name": "token", "label": "Bearer token", "type": "password", "secret": True}],
}


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        session.add(CustomConnector(connector_id="httpbin", label="httpbin", description="", spec=FORM))
        session.add(CustomConnector(connector_id="httpbin", label="httpbin", description="", spec=FORM, org_id="org-a"))
        session.commit()
    return eng


def _scoped(engine, scope=LOCAL_SCOPE) -> ScopedSession:
    return ScopedSession(Session(engine), scope)


def test_rename_and_recategorize(engine):
    result = update_custom_connector(
        "httpbin", CustomConnectorUpdate(label="HTTP Bin", category="developer"), _scoped(engine),
    )

    assert result.label == "HTTP Bin" and result.category == "developer" and result.custom is True


def test_unfeatured_connector_leaves_featured_but_stays_listed(engine):
    update_custom_connector("httpbin", CustomConnectorUpdate(featured=False), _scoped(engine))

    listed = {c.id: c for c in ConnectorCatalog(_scoped(engine)).list_connectors()}
    assert listed["httpbin"].featured is False


def test_a_null_field_is_left_unchanged(engine):
    result = update_custom_connector(
        "httpbin", CustomConnectorUpdate.model_validate({"featured": None, "label": None}), _scoped(engine),
    )

    assert result.featured is True and result.label == "httpbin"


def test_replace_the_form(engine):
    new_form = {**FORM, "fields": [{"name": "api_key", "label": "API key", "type": "password", "secret": True}]}

    update_custom_connector("httpbin", CustomConnectorUpdate(spec=new_form), _scoped(engine))

    spec = ConnectorCatalog(_scoped(engine)).get_connector("httpbin")
    assert [f.name for f in spec.form.fields] == ["api_key"]


def test_an_invalid_form_is_a_422(engine):
    with pytest.raises(HTTPException) as exc:
        update_custom_connector(
            "httpbin", CustomConnectorUpdate(spec={"form_id": "x", "fields": [{"name": "token"}]}), _scoped(engine),
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_delete_removes_the_definition_but_keeps_connections(engine, tmp_path, monkeypatch):
    vault_dir = Path(tmp_path) / "vault"
    LocalDataVault(vault_dir).save("httpbin", "httpbin-1a2b3c4d", {"token": "t"}, secure_keys=["token"])
    monkeypatch.setattr(
        "cowork.services.connectors.connections.ConnectorSettings",
        lambda: type("S", (), {"vault_dir": str(vault_dir)})(),
    )

    async def run_db(fn, *, scope):
        with Session(engine) as session:
            return fn(ScopedSession(session, scope))

    monkeypatch.setattr(connections_endpoints, "run_db", run_db)
    [before] = await connections_endpoints.list_connections(LOCAL_SCOPE, FakeRequest())

    delete_custom_connector("httpbin", _scoped(engine))

    assert ConnectorCatalog(_scoped(engine)).get_connector("httpbin") is None
    [after] = await connections_endpoints.list_connections(LOCAL_SCOPE, FakeRequest())
    assert before.custom is True
    assert (after.engine, after.name) == ("httpbin", "httpbin-1a2b3c4d")
    assert after.custom is False


def test_another_orgs_connector_is_a_404(engine):
    with pytest.raises(HTTPException) as exc:
        update_custom_connector("httpbin", CustomConnectorUpdate(featured=False), _scoped(engine, ORG_B))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        delete_custom_connector("httpbin", _scoped(engine, ORG_B))
    assert exc.value.status_code == 404


def test_an_org_admin_changes_only_their_own_orgs_row(engine):
    update_custom_connector("httpbin", CustomConnectorUpdate(label="Org copy"), _scoped(engine, ORG_A))

    assert ConnectorCatalog(_scoped(engine, ORG_A)).get_connector("httpbin").label == "Org copy"
    assert ConnectorCatalog(_scoped(engine)).get_connector("httpbin").label == "httpbin"


def _org_client(engine, principal) -> TestClient:
    app = FastAPI()
    app.include_router(custom_router, prefix="/custom")
    app.dependency_overrides[get_principal] = lambda: principal
    app.dependency_overrides[get_scoped_session] = lambda: _scoped(engine, ORG_A)
    return TestClient(app)


def test_a_member_who_is_not_an_admin_gets_a_403_in_org_mode(engine, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        member = Principal(user_id="member-a", org_id="org-a")
        admin = Principal(user_id="admin-a", org_id="org-a", roles=frozenset({ORG_MANAGE_ROLE}))

        assert _org_client(engine, member).patch("/custom/httpbin", json={"featured": False}).status_code == 403
        assert _org_client(engine, member).delete("/custom/httpbin").status_code == 403
        assert _org_client(engine, admin).patch("/custom/httpbin", json={"featured": False}).status_code == 200
    finally:
        get_app_settings.cache_clear()


def test_both_routes_require_an_org_admin():
    # AuthenticatedOrgAdmin's own 403 for a non-admin is covered in test_permissions.py.
    routes = [
        r for r in create_app().routes
        if getattr(r, "path", "").startswith("/api/v1/connectors/custom/")
    ]

    assert {tuple(sorted(r.methods)) for r in routes} == {("PATCH",), ("DELETE",)}
    for route in routes:
        assert AuthenticatedOrgAdmin in declared_permissions(route)
