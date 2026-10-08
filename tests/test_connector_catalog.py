"""Custom connectors are served beside the registry, and only to their own scope.

Runs the catalog against a real SQLite database through ScopedSession, the
same tenancy layer requests use, so the org boundary is the real one.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import MatchRequest
from cowork.services.connectors import catalog as catalog_module
from cowork.services.connectors.catalog import ConnectorCatalog
from tests._fakes import FakeRequest

ORG_A = TenantScope(org_mode=True, org_id="org-a", user_id="user-a")
ORG_B = TenantScope(org_mode=True, org_id="org-b", user_id="user-b")

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

    def test_a_builtin_id_wins_over_a_custom_row(self, engine, monkeypatch):
        warnings = []
        monkeypatch.setattr(catalog_module.logger, "warning", lambda msg, *args: warnings.append(msg % args))
        _add(engine, connector_id="postgres", label="Shadow Postgres")

        spec = _catalog(engine, LOCAL_SCOPE).get_connector("postgres")

        assert spec.custom is False
        assert spec.label != "Shadow Postgres"
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
