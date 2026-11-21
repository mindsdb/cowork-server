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
from tests._connector_test_request import with_test_request
from tests._fakes import FakeRequest

ORG_A = TenantScope(org_mode=True, org_id="org-a", user_id="user-a")
ORG_B = TenantScope(org_mode=True, org_id="org-b", user_id="user-b")
ORG_A_OTHER_MEMBER = TenantScope(org_mode=True, org_id="org-a", user_id="user-a2")
ORG_A_NO_USER = TenantScope(org_mode=True, org_id="org-a")
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

HTTPBIN_FORM = with_test_request({
    "form_id": "httpbin-connector",
    "title": "Connect httpbin",
    "fields": [{"name": "token", "label": "Bearer token", "type": "password", "secret": True}],
})


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


def _add(engine, *, connector_id="httpbin", org_id=None, featured=True, label="httpbin", created_by=None):
    with Session(engine) as session:
        session.add(CustomConnector(
            connector_id=connector_id, label=label, description="Bearer test API",
            category="developer", logo_color="#3a7", spec=HTTPBIN_FORM,
            featured=featured, org_id=org_id, created_by=created_by,
        ))
        session.commit()


def _add_with_notes(engine, notes, *, org_id=None, created_by=None):
    """Store the httpbin definition with ``notes`` as its usage notes."""
    _add(engine, org_id=org_id, created_by=created_by)
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
        _add(engine, org_id="org-a", created_by=ORG_A.user_id)

        assert _catalog(engine, ORG_A).get_connector("httpbin") is not None
        assert _catalog(engine, ORG_B).get_connector("httpbin") is None

    def test_another_member_of_the_org_never_sees_a_members_connector(self, engine):
        _add(engine, org_id="org-a", created_by=ORG_A.user_id)

        assert _catalog(engine, ORG_A_OTHER_MEMBER).get_connector("httpbin") is None
        assert _catalog(engine, ORG_A_NO_USER).get_connector("httpbin") is None
        assert _catalog(engine, ORG_A_OTHER_MEMBER).match_connector("httpbin", 5).candidates == []

    def test_an_org_row_with_no_builder_is_seen_by_no_member(self, engine):
        _add(engine, org_id="org-a")

        assert _catalog(engine, ORG_A).get_connector("httpbin") is None
        # A scope with no user must not match the row's missing builder.
        assert _catalog(engine, ORG_A_NO_USER).get_connector("httpbin") is None

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
        _add_with_notes(engine, "Call /bearer.", org_id=row_org, created_by=scope.user_id)

        catalog = _catalog(engine, scope)

        assert catalog.usage_notes_for(["httpbin"]) == ({"httpbin": "Call /bearer."} if own else {})
        assert catalog.get_connectors()["httpbin"]["usage_notes"] == ("Call /bearer." if own else None)

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
    async def test_only_the_builder_looks_up_or_matches_an_org_connector(self, endpoint_db):
        _add(endpoint_db, org_id="org-a", created_by=ORG_A.user_id)

        assert (await specs_endpoints.get_connector_spec("httpbin", ORG_A)).custom is True
        assert (await specs_endpoints.match_connector_spec(MatchRequest(query="httpbin"), ORG_A)).candidates
        with pytest.raises(HTTPException) as exc:
            await specs_endpoints.get_connector_spec("httpbin", ORG_A_OTHER_MEMBER)
        assert exc.value.status_code == 404
        matched = await specs_endpoints.match_connector_spec(MatchRequest(query="httpbin"), ORG_A_OTHER_MEMBER)
        assert matched.candidates == []

    @pytest.mark.asyncio
    async def test_match_includes_custom_connectors(self, endpoint_db):
        _add(endpoint_db)
        result = await specs_endpoints.match_connector_spec(MatchRequest(query="httpbin"), LOCAL_SCOPE)

        assert result.candidates[0].id == "httpbin"


PROBED_FORM = with_test_request({
    "form_id": "supply-connector",
    "title": "Connect supply",
    "fields": [
        {"name": "base_url", "label": "Base URL", "type": "text"},
        {"name": "client_secret", "label": "Client secret", "type": "password", "secret": True,
         "description": "The OAuth client secret."},
        {"name": "resource_endpoint", "label": "Resource endpoint", "type": "text",
         "description": "For the connection check only, not where data is queried."},
    ],
})


def _add_probed(engine, *, notes=None, org_id=None, created_by=None, form=PROBED_FORM):
    """Store a supply definition whose form describes some of its fields."""
    with Session(engine) as session:
        session.add(CustomConnector(
            connector_id="supply", label="supply", description="Supply API", category="developer",
            spec=form, usage_notes=notes, org_id=org_id, created_by=created_by,
        ))
        session.commit()


class TestTurnNotes:
    def test_a_described_field_is_named_by_the_end_of_its_variable(self, engine):
        _add_probed(engine)

        text = _catalog(engine, LOCAL_SCOPE).turn_notes_for(["supply"])["supply"]

        assert "`__RESOURCE_ENDPOINT`: For the connection check only, not where data is queried." in text
        assert "`__CLIENT_SECRET`: The OAuth client secret." in text
        assert "__BASE_URL" not in text

    def test_field_lines_come_after_the_notes_which_stay_whole(self, engine):
        notes = "POST the query to {{base_url}}/integration/V1/query. " * 30
        _add_probed(engine, notes=notes)

        text = _catalog(engine, LOCAL_SCOPE).turn_notes_for(["supply"])["supply"]

        assert text.startswith(notes)
        assert text.index("__RESOURCE_ENDPOINT") > len(notes)

    def test_a_connector_with_no_notes_and_no_descriptions_gets_none(self, engine):
        _add(engine, connector_id="supply")

        assert _catalog(engine, LOCAL_SCOPE).turn_notes_for(["supply"]) == {}

    def test_a_long_description_is_cut(self, engine):
        form = {**PROBED_FORM, "fields": [{"name": "region", "label": "Region", "type": "text",
                                           "description": "x" * 500}]}
        _add_probed(engine, form=form)

        line = _catalog(engine, LOCAL_SCOPE).turn_notes_for(["supply"])["supply"].splitlines()[-1]

        assert line == "- `__REGION`: " + "x" * 200

    def test_a_secret_looking_description_is_scrubbed(self, engine):
        form = {**PROBED_FORM, "fields": [{"name": "key", "label": "Key", "type": "password",
                                           "description": "Was sk-proj-abcdefghijklmnopqrstuvwx, rotated."}]}
        _add_probed(engine, form=form)

        text = _catalog(engine, LOCAL_SCOPE).turn_notes_for(["supply"])["supply"]

        assert "sk-proj-abcdefghijklmnopqrstuvwx" not in text
        assert text.endswith(", rotated.")

    @pytest.mark.parametrize("scope, row_org, own", [
        (PERSON_A, "partition-a", True),
        (PERSON_SHARED, "partition-a", False),
        (ORG_A, "org-a", False),
    ], ids=["personal connections", "shared connections", "org mode"])
    def test_field_lines_reach_an_agent_only_in_ones_own_scope(self, engine, scope, row_org, own):
        _add_probed(engine, org_id=row_org, created_by=scope.user_id)

        notes = _catalog(engine, scope).turn_notes_for(["supply"])

        assert ("supply" in notes) is own

    def test_the_probe_still_reads_only_the_stored_notes(self, engine):
        _add_probed(engine)

        assert _catalog(engine, LOCAL_SCOPE).usage_notes_for(["supply"]) == {}

    def test_a_built_in_connector_keeps_its_own_notes(self, engine):
        catalog = _catalog(engine, LOCAL_SCOPE)

        assert catalog.turn_notes_for(["postgres"]) == catalog.usage_notes_for(["postgres"])
