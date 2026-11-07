"""A custom connector's owner can change and delete it, in their own scope only.

Locally the one user owns every definition. In org mode only the member who
built a definition sees it, and may change only its label, description and
category."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.api.v1.endpoints.connectors.custom import delete_custom_connector, update_custom_connector
from cowork.api.v1.endpoints.connectors.custom import router as custom_router
from cowork.api.v1.permissions import AuthenticatedInOrgMode, AuthenticatedOrgAdmin
from cowork.api.v1.route_walker import declared_permissions
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, get_scoped_session
from cowork.models.custom_connector import CustomConnector
from cowork.principal import ORG_MANAGE_ROLE, Principal, get_principal
from cowork.schemas.connectors import CustomConnectorUpdate
from cowork.server import create_app
from cowork.services.connectors.catalog import ConnectorCatalog
from tests._fakes import FakeRequest

ORG_A = TenantScope(org_mode=True, org_id="org-a", user_id="builder-a")
ORG_A_OTHER = TenantScope(org_mode=True, org_id="org-a", user_id="member-a2")
ORG_B = TenantScope(org_mode=True, org_id="org-b", user_id="member-b")

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
        session.add(CustomConnector(
            connector_id="httpbin", label="httpbin", description="", spec=FORM,
            org_id="org-a", created_by=ORG_A.user_id,
        ))
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


def test_a_null_category_or_description_clears_it(engine):
    update_custom_connector(
        "httpbin", CustomConnectorUpdate(category="crm", description="Bearer test API"), _scoped(engine),
    )

    result = update_custom_connector(
        "httpbin", CustomConnectorUpdate.model_validate({"category": None, "description": None}), _scoped(engine),
    )

    assert result.category == "other"
    assert result.description == ""
    assert result.label == "httpbin" and result.featured is True


def test_a_new_form_applies_its_connector_block(engine):
    new_form = {**FORM, "connector": {"label": "httpbin v2", "usage_notes": "Call /bearer."}}

    result = update_custom_connector(
        "httpbin", CustomConnectorUpdate(spec=new_form, category="developer"), _scoped(engine),
    )

    assert result.label == "httpbin v2"
    assert result.category == "developer"
    assert ConnectorCatalog(_scoped(engine)).get_connectors()["httpbin"]["usage_notes"] == "Call /bearer."


def test_a_field_sent_on_its_own_wins_over_the_connector_block(engine):
    new_form = {**FORM, "connector": {"label": "From the form", "category": "crm"}}

    sent = update_custom_connector(
        "httpbin", CustomConnectorUpdate(spec=new_form, label="Sent"), _scoped(engine),
    )
    kept = update_custom_connector(
        "httpbin", CustomConnectorUpdate.model_validate({"spec": new_form, "label": None, "category": None}),
        _scoped(engine),
    )

    assert sent.label == "Sent"
    assert kept.label == "Sent"
    assert kept.category == "other"


def test_replace_the_form(engine):
    new_form = {**FORM, "fields": [{"name": "api_key", "label": "API key", "type": "password", "secret": True}]}

    update_custom_connector("httpbin", CustomConnectorUpdate(spec=new_form), _scoped(engine))

    spec = ConnectorCatalog(_scoped(engine)).get_connector("httpbin")
    assert [f.name for f in spec.form.fields] == ["api_key"]


def test_a_new_form_color_moves_the_directory_tile(engine):
    update_custom_connector("httpbin", CustomConnectorUpdate(spec={**FORM, "logo_color": "#33aa77"}), _scoped(engine))

    listed = {c.id: c for c in ConnectorCatalog(_scoped(engine)).list_connectors()}
    assert listed["httpbin"].logo_color == "#33aa77"


def test_a_new_form_without_a_color_keeps_the_tile_color(engine):
    update_custom_connector("httpbin", CustomConnectorUpdate(spec={**FORM, "logo_color": "#33aa77"}), _scoped(engine))
    update_custom_connector("httpbin", CustomConnectorUpdate(spec=FORM), _scoped(engine))

    listed = {c.id: c for c in ConnectorCatalog(_scoped(engine)).list_connectors()}
    assert listed["httpbin"].logo_color == "#33aa77"


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
    monkeypatch.setenv("COWORK_VAULT_DIR", str(vault_dir))

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


@pytest.mark.parametrize("scope", [ORG_B, ORG_A_OTHER], ids=["another org", "another member"])
def test_someone_elses_connector_is_a_404(engine, scope):
    with pytest.raises(HTTPException) as exc:
        update_custom_connector("httpbin", CustomConnectorUpdate(label="Mine now"), _scoped(engine, scope))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        delete_custom_connector("httpbin", _scoped(engine, scope))
    assert exc.value.status_code == 404
    assert ConnectorCatalog(_scoped(engine, ORG_A)).get_connector("httpbin").label == "httpbin"


def test_the_builder_renames_only_their_own_row(engine):
    update_custom_connector("httpbin", CustomConnectorUpdate(label="Org copy"), _scoped(engine, ORG_A))

    assert ConnectorCatalog(_scoped(engine, ORG_A)).get_connector("httpbin").label == "Org copy"
    assert ConnectorCatalog(_scoped(engine)).get_connector("httpbin").label == "httpbin"


@pytest.mark.parametrize("body", [
    {"spec": {**FORM, "connector": {"usage_notes": "Send the team token in X-Team-Key."}}},
    {"featured": False},
    {"featured": None},
    {"label": "Renamed", "spec": FORM},
], ids=["form", "featured", "featured null", "label with form"])
def test_org_mode_changes_nothing_but_label_description_and_category(engine, body):
    with pytest.raises(HTTPException) as exc:
        update_custom_connector("httpbin", CustomConnectorUpdate.model_validate(body), _scoped(engine, ORG_A))
    assert exc.value.status_code == 422

    with Session(engine) as session:
        [row] = session.exec(select(CustomConnector).where(CustomConnector.org_id == "org-a")).all()
    assert (row.label, row.featured, row.spec, row.usage_notes) == ("httpbin", True, FORM, None)


def _org_client(engine, principal, scope) -> TestClient:
    app = FastAPI()
    app.include_router(custom_router, prefix="/custom")
    app.dependency_overrides[get_principal] = lambda: principal
    app.dependency_overrides[get_scoped_session] = lambda: _scoped(engine, scope)
    return TestClient(app)


def test_in_org_mode_the_builder_manages_their_connector_and_no_one_else(engine, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        builder = Principal(user_id=ORG_A.user_id, org_id="org-a")
        admin = Principal(user_id=ORG_A_OTHER.user_id, org_id="org-a", roles=frozenset({ORG_MANAGE_ROLE}))

        assert _org_client(engine, admin, ORG_A_OTHER).patch("/custom/httpbin", json={"label": "x"}).status_code == 404
        assert _org_client(engine, admin, ORG_A_OTHER).delete("/custom/httpbin").status_code == 404
        renamed = _org_client(engine, builder, ORG_A).patch("/custom/httpbin", json={"label": "Acme"})
        assert renamed.status_code == 200 and renamed.json()["label"] == "Acme"
        assert _org_client(engine, builder, ORG_A).delete("/custom/httpbin").status_code == 204
    finally:
        get_app_settings.cache_clear()


def test_both_routes_require_an_authenticated_member_not_an_admin():
    routes = [
        r for r in create_app().routes
        if getattr(r, "path", "").startswith("/api/v1/connectors/custom/")
    ]

    assert {tuple(sorted(r.methods)) for r in routes} == {("PATCH",), ("DELETE",)}
    for route in routes:
        assert AuthenticatedInOrgMode in declared_permissions(route)
        assert AuthenticatedOrgAdmin not in declared_permissions(route)
