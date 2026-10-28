"""Custom connectors are served beside the registry, and only to their own scope.

Runs the catalog against a real SQLite database through ScopedSession, the
same tenancy layer requests use, so the org boundary is the real one.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import MatchRequest
from cowork.services.connectors import catalog as catalog_module
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.custom_connectors import CustomConnectorService
from tests._fakes import FakeRequest

ORG_A = TenantScope(org_mode=True, org_id="org-a", user_id="user-a")
ORG_B = TenantScope(org_mode=True, org_id="org-b", user_id="user-b")
PERSON_A = TenantScope(
    org_id="partition-a", user_id="person-a", personal=True, personal_resources=ALL_PERSONAL_RESOURCES,
)
PERSON_B = TenantScope(
    org_id="partition-b", user_id="person-b", personal=True, personal_resources=ALL_PERSONAL_RESOURCES,
)
PERSON_SHARED = TenantScope(
    org_id="partition-a", user_id="person-a", personal=True,
    personal_resources=ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS},
)

HTTPBIN_FORM = {
    "form_id": "httpbin-connector",
    "title": "Connect httpbin",
    "fields": [{"name": "token", "label": "Bearer token", "type": "password", "secret": True}],
}


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


def _add(engine, *, connector_id="httpbin", org_id=None, featured=True, label="httpbin"):
    with Session(engine) as session:
        session.add(CustomConnector(
            connector_id=connector_id, label=label, description="Bearer test API",
            category="developer", logo_color="#3a7", spec=HTTPBIN_FORM,
            featured=featured, org_id=org_id,
        ))
        session.commit()


def _add_with_notes(engine, notes, *, org_id=None):
    """Store the httpbin definition with ``notes`` as its usage notes."""
    _add(engine, org_id=org_id)
    with Session(engine) as session:
        row = session.exec(select(CustomConnector)).one()
        row.usage_notes = notes
        session.add(row)
        session.commit()


def _catalog(engine, scope) -> ConnectorCatalog:
    return ConnectorCatalog(ScopedSession(Session(engine), scope))


class TestCatalog:
    def test_a_local_custom_connector_is_listed_flagged_custom_and_featured(self, engine):
        _add(engine)
        listed = {c.id: c for c in _catalog(engine, LOCAL_SCOPE).list_connectors()}

        assert listed["httpbin"].custom is True
        assert listed["httpbin"].featured is True
        assert listed["postgres"].custom is False

    def test_lookup_returns_the_stored_form(self, engine):
        _add(engine)
        spec = _catalog(engine, LOCAL_SCOPE).get_connector("httpbin")

        assert spec is not None and spec.custom is True
        assert spec.form.fields[0].name == "token"

    def test_match_finds_a_custom_connector_by_id(self, engine):
        _add(engine)
        result = _catalog(engine, LOCAL_SCOPE).match_connector("httpbin")

        assert [c.id for c in result.candidates] == ["httpbin"]

    def test_one_org_never_sees_another_orgs_connector(self, engine):
        _add(engine, org_id="org-a")

        assert _catalog(engine, ORG_A).get_connector("httpbin") is not None
        assert _catalog(engine, ORG_B).get_connector("httpbin") is None

    def test_a_local_install_never_sees_an_orgs_connector(self, engine):
        _add(engine, org_id="org-a")

        assert _catalog(engine, LOCAL_SCOPE).get_connector("httpbin") is None

    def test_an_org_never_sees_a_local_connector(self, engine):
        _add(engine)

        assert _catalog(engine, ORG_A).get_connector("httpbin") is None

    def test_a_personal_scope_sees_and_updates_its_own_saved_connector(self, engine):
        with Session(engine) as session:
            service = CustomConnectorService(ScopedSession(session, PERSON_A))
            service.upsert_from_form("httpbin", HTTPBIN_FORM)
            service.upsert_from_form("httpbin", {**HTTPBIN_FORM, "title": "Connect httpbin again"})

        assert _catalog(engine, PERSON_A).get_connector("httpbin").label == "Connect httpbin again"
        assert _catalog(engine, PERSON_B).get_connector("httpbin") is None
        assert _catalog(engine, LOCAL_SCOPE).get_connector("httpbin") is None

    def test_a_scope_shared_by_several_people_sees_no_custom_connector(self, engine):
        _add(engine)
        _add(engine, connector_id="private", org_id=PERSON_A.org_id)

        catalog = _catalog(engine, PERSON_SHARED)
        assert catalog.get_connector("httpbin") is None
        assert catalog.get_connector("private") is None
        assert not any(c.custom for c in catalog.list_connectors())

    def test_the_operator_view_sees_no_custom_connector(self, engine, personal_deployment):
        _add(engine)

        catalog = _catalog(engine, LOCAL_SCOPE)
        assert catalog.get_connector("httpbin") is None
        assert catalog.usage_notes_for(["httpbin"]) == {}

    @pytest.mark.parametrize("scope, row_org, own", [
        (LOCAL_SCOPE, None, True),
        (PERSON_A, "partition-a", True),
        (ORG_A, "org-a", False),
    ], ids=["single-user install", "personal connections", "org mode"])
    def test_custom_usage_notes_reach_an_agent_only_in_ones_own_scope(self, engine, scope, row_org, own):
        _add_with_notes(engine, "Call /bearer.", org_id=row_org)

        catalog = _catalog(engine, scope)

        assert catalog.usage_notes_for(["httpbin"]) == ({"httpbin": "Call /bearer."} if own else {})
        assert catalog.get_connectors()["httpbin"]["usage_notes"] == "Call /bearer."

    def test_custom_usage_notes_are_scrubbed_before_an_agent_reads_them(self, engine):
        _add_with_notes(engine, "Old key sk-proj-abcdefghijklmnopqrstuvwx, now rotated.")

        notes = _catalog(engine, LOCAL_SCOPE).usage_notes_for(["httpbin"])

        assert "sk-proj-abcdefghijklmnopqrstuvwx" not in notes["httpbin"]
        assert notes["httpbin"].endswith(", now rotated.")

    def test_a_builtin_id_wins_over_a_custom_row(self, engine, monkeypatch):
        warnings = []
        monkeypatch.setattr(catalog_module.logger, "warning", lambda msg, *args: warnings.append(msg % args))
        _add(engine, connector_id="postgres", label="Shadow Postgres")

        catalog = _catalog(engine, LOCAL_SCOPE)
        spec = catalog.get_connector("postgres")
        listed = {c.id: c for c in catalog.list_connectors()}

        assert spec.custom is False
        assert spec.label != "Shadow Postgres"
        assert listed["postgres"].custom is False
        assert any("'postgres'" in w for w in warnings)


@pytest.fixture()
def endpoint_db(engine, monkeypatch):
    """Route the specs endpoints' unit of work to the test database."""

    async def run_db(fn, *, scope):
        with Session(engine) as session:
            return fn(ScopedSession(session, scope))

    monkeypatch.setattr(specs_endpoints, "run_db", run_db)
    return engine


class TestSpecsEndpoints:
    @pytest.mark.asyncio
    async def test_local_list_includes_the_custom_connector(self, endpoint_db):
        _add(endpoint_db)
        result = await specs_endpoints.list_connector_specs(LOCAL_SCOPE, FakeRequest())

        assert any(c.id == "httpbin" and c.custom for c in result)

    @pytest.mark.asyncio
    async def test_lookup_serves_a_custom_connector(self, endpoint_db):
        _add(endpoint_db)
        spec = await specs_endpoints.get_connector_spec("httpbin", LOCAL_SCOPE)

        assert spec.custom is True

    @pytest.mark.asyncio
    async def test_lookup_of_another_orgs_connector_is_a_404(self, endpoint_db):
        _add(endpoint_db, org_id="org-a")

        with pytest.raises(HTTPException) as exc:
            await specs_endpoints.get_connector_spec("httpbin", ORG_B)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_match_includes_custom_connectors(self, endpoint_db):
        _add(endpoint_db)
        result = await specs_endpoints.match_connector_spec(MatchRequest(query="httpbin"), LOCAL_SCOPE)

        assert result.candidates[0].id == "httpbin"
