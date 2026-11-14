"""A member's own custom templates in the org-mode directory, and building offered there."""

from __future__ import annotations

import pytest
from anton.core.tools.connector_draft import parse_connector_draft
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.custom_connector import CustomConnector
from cowork.services.connectors.custom_connectors import CustomConnectorService
from tests._fakes import FakeRequest, inline_run_db

BUILDER = TenantScope(org_mode=True, org_id="org-a", user_id="builder-a")
SAME_ORG = TenantScope(org_mode=True, org_id="org-a", user_id="member-a2")
OTHER_ORG = TenantScope(org_mode=True, org_id="org-b", user_id="builder-a")

DRAFT = {
    "connector_id": "acme_crm",
    "label": "Acme CRM",
    "fields": [{"name": "api_key", "label": "API key"}],
    "auth": {"type": "header", "headers": {"Authorization": "Bearer {api_key}"}},
    "check": {"path": "/v1/me"},
}


@pytest.fixture()
def engine():
    """The builder's one template, plus a stored row that takes a built-in id."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    CustomConnectorService(ScopedSession(Session(eng), BUILDER)).save_member_draft(parse_connector_draft(DRAFT))
    with Session(eng) as session:
        session.add(CustomConnector(
            connector_id="postgres", label="Shadow", spec={"fields": []}, org_id="org-a", created_by="builder-a",
        ))
        session.commit()
    return eng


@pytest.fixture()
def catalogue(monkeypatch):
    async def fake_proxy_catalogue(request, settings):
        return {"items": [{"id": "gmail"}]}

    monkeypatch.setattr(specs_endpoints.auth_proxy, "proxy_catalogue", fake_proxy_catalogue)


@pytest.fixture()
def drafts_on(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", "true")


async def _list(engine, monkeypatch, scope, **kwargs):
    monkeypatch.setattr(specs_endpoints, "run_db", inline_run_db(ScopedSession(Session(engine), scope)))
    return await specs_endpoints.list_connector_specs(scope, FakeRequest(), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_unavailable", [False, True])
async def test_the_builder_sees_their_template_listed_as_available(engine, monkeypatch, catalogue, drafts_on, include_unavailable):
    result = await _list(engine, monkeypatch, BUILDER, include_unavailable=include_unavailable)

    by_id = {c.id: c for c in result}
    assert by_id["acme_crm"].custom is True and by_id["acme_crm"].cloud_available is True
    assert by_id["acme_crm"].label == "Acme CRM"
    assert "gmail" in by_id
    labels = [c.label.lower() for c in result]
    assert labels == sorted(labels)


@pytest.mark.asyncio
async def test_a_stored_row_with_a_built_in_id_is_never_listed_as_custom(engine, monkeypatch, catalogue, drafts_on):
    result = await _list(engine, monkeypatch, BUILDER, include_unavailable=True)

    postgres = [c for c in result if c.id == "postgres"]
    assert len(postgres) == 1 and postgres[0].custom is False and postgres[0].label != "Shadow"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [SAME_ORG, OTHER_ORG], ids=["same-org-member", "other-org"])
async def test_no_one_else_sees_the_template(engine, monkeypatch, catalogue, drafts_on, scope):
    result = await _list(engine, monkeypatch, scope)

    assert [c.id for c in result] == ["gmail"]


@pytest.mark.asyncio
async def test_with_the_setting_off_no_template_is_listed_and_the_database_is_not_asked(engine, monkeypatch, catalogue):
    async def refuse(fn, *, scope):
        raise AssertionError("the database was asked")

    monkeypatch.setattr(specs_endpoints, "run_db", refuse)

    result = await specs_endpoints.list_connector_specs(BUILDER, FakeRequest())

    assert [c.id for c in result] == ["gmail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("setting", "can_build"), [("true", True), ("false", False)])
async def test_building_is_offered_in_org_mode_only_while_the_setting_is_on(monkeypatch, setting, can_build):
    monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", setting)

    result = await specs_endpoints.custom_connector_availability(BUILDER)

    assert result.can_build is can_build
