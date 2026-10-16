"""A handcrafted connector that passes its connection test becomes a custom connector.

The probe stream runs with a stand-in probe and real storage: a SQLite
database for definitions and a temp vault for credentials.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoints
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.handlers import probe as probe_handler
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import SubmitFormRequest
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.harnesses.anton_harness import tools
from cowork.services.connectors.custom_connectors import CustomConnectorService, UnstorableFormError, check_handcrafted_form
from cowork.services.connectors.probe import ProbeOutcome
from cowork.services.connectors.submissions import store

ORG = TenantScope(org_mode=True, org_id="org-a", user_id="user-a")

HTTPBIN_FORM = {
    "form_id": "fm_aaaaaaaaaa",
    "engine": "httpbin",
    "title": "Connect httpbin",
    "subtitle": "Bearer token test API",
    "how_to": "Any token works.",
    "logo_color": "#3a7",
    "connector": {"label": "httpbin", "category": "developer", "usage_notes": "Call /bearer."},
    "fields": [{"name": "token", "label": "Bearer token", "type": "password", "secret": True}],
}


@pytest.fixture()
def engine(monkeypatch):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)

    async def run_db(fn, *, scope):
        with Session(eng) as session:
            return fn(ScopedSession(session, scope))

    monkeypatch.setattr(probe_handler, "run_db", run_db)
    return eng


def _rows(engine) -> list[CustomConnector]:
    """Every stored definition, whatever its scope."""
    with Session(engine) as session:
        return list(session.exec(select(CustomConnector)).all())


class _Probe:
    verdict = "success"
    runs = 0

    def __init__(self, **_kwargs) -> None:
        pass

    async def run(self):
        type(self).runs += 1
        yield "verdict", ProbeOutcome(status=type(self).verdict, summary="ok", error="rejected")


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    _Probe.runs = 0
    vault = LocalDataVault(Path(tmp_path) / "vault")
    monkeypatch.setattr(probe_handler, "vault_for_scope", lambda scope: vault)
    monkeypatch.setattr(probe_handler, "CredentialProbe", _Probe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr("anton.workspace.Workspace", lambda path: SimpleNamespace(path=path))
    return vault


async def _probe(form_spec: dict, values: dict, *, scope=LOCAL_SCOPE, method=None) -> str:
    submission_id = store.stage(
        form_id=form_spec["form_id"], connector_id="httpbin", conversation_id=None,
        values=values, form_spec=form_spec, checked_against_stored_spec=False,
    )
    events = [e async for e in probe_handler.ProbeHandler(scope=scope).run(submission_id, "httpbin", method, "", None)]
    return "".join(events)


class TestService:
    def test_a_form_becomes_a_featured_definition_with_a_stable_form_id(self, engine):
        with Session(engine) as session:
            row = CustomConnectorService(ScopedSession(session, LOCAL_SCOPE)).upsert_from_form("httpbin", HTTPBIN_FORM)

            assert row.label == "httpbin"
            assert row.category == "developer"
            assert row.usage_notes == "Call /bearer."
            assert row.featured is True
            assert row.spec["form_id"] == "httpbin-connector"
            assert row.spec["how_to"] == "Any token works."
            assert "engine" not in row.spec and "connector" not in row.spec

    def test_saving_again_updates_the_definition_and_keeps_it_unfeatured(self, engine):
        with Session(engine) as session:
            service = CustomConnectorService(ScopedSession(session, LOCAL_SCOPE))
            first = service.upsert_from_form("httpbin", HTTPBIN_FORM)
            first.featured = False
            session.add(first)
            session.commit()

            again = service.upsert_from_form("httpbin", {**HTTPBIN_FORM, "subtitle": "Updated"})

            assert again.id == first.id
            assert again.description == "Updated"
            assert again.featured is False
        assert len(_rows(engine)) == 1

    def test_an_invalid_form_is_refused(self, engine):
        with Session(engine) as session:
            with pytest.raises(UnstorableFormError):
                CustomConnectorService(ScopedSession(session, LOCAL_SCOPE)).upsert_from_form(
                    "httpbin", {"form_id": "x", "fields": [{"name": "token"}]},
                )


class TestProbeStream:
    @pytest.mark.asyncio
    async def test_a_passing_probe_saves_the_connection_and_the_definition(self, engine, vault):
        _Probe.verdict = "success"

        stream = await _probe(HTTPBIN_FORM, {"token": "any-token"})

        assert _Probe.runs == 1
        [connection] = vault.list_connections()
        assert f"Saved as `{connection['name']}`" in stream
        [row] = _rows(engine)
        assert row.connector_id == "httpbin"
        with Session(engine) as session:
            spec = ConnectorCatalog(ScopedSession(session, LOCAL_SCOPE)).get_connector("httpbin")
        assert spec.custom is True and spec.featured is True

    @pytest.mark.asyncio
    async def test_a_failing_probe_saves_nothing(self, engine, vault):
        _Probe.verdict = "failure"

        await _probe(HTTPBIN_FORM, {"token": "wrong"})

        assert _Probe.runs == 1
        assert vault.list_connections() == []
        assert _rows(engine) == []

    @pytest.mark.asyncio
    async def test_in_org_mode_the_connection_is_saved_but_no_definition(self, engine, vault):
        _Probe.verdict = "success"

        await _probe(HTTPBIN_FORM, {"token": "any-token"}, scope=ORG)

        assert len(vault.list_connections()) == 1
        assert _rows(engine) == []

    @pytest.mark.asyncio
    async def test_a_second_connect_updates_the_definition(self, engine, vault):
        _Probe.verdict = "success"

        await _probe(HTTPBIN_FORM, {"token": "a"})
        await _probe({**HTTPBIN_FORM, "subtitle": "Second"}, {"token": "b"})

        [row] = _rows(engine)
        assert row.description == "Second"

    @pytest.mark.asyncio
    async def test_a_handcrafted_oauth_grant_is_saved_without_probing(self, engine, vault):
        form = {
            "form_id": "fm_cccccccccc", "engine": "httpbin", "title": "Connect httpbin",
            "methods": [{
                "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
                "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
            }],
        }

        await _probe(form, {"access_token": "t", "refresh_token": "r"}, method="oauth")

        assert _Probe.runs == 0
        assert len(vault.list_connections()) == 1
        assert [r.connector_id for r in _rows(engine)] == ["httpbin"]


class TestProbeIsNotSkippedByAModelFlag:
    """Only a real OAuth grant on the selected method skips the live test."""

    @pytest.mark.asyncio
    async def test_a_form_level_oauth_flag_is_still_probed(self, engine, vault):
        _Probe.verdict = "failure"
        form = {**HTTPBIN_FORM, "form_id": "fm_eeeeeeeeee", "submit_action": "oauth_launch"}

        await _probe(form, {"token": "typed-key"})

        assert _Probe.runs == 1
        assert vault.list_connections() == []
        assert _rows(engine) == []

    @pytest.mark.asyncio
    async def test_an_oauth_method_without_a_token_is_still_probed(self, engine, vault):
        _Probe.verdict = "failure"
        form = {
            "form_id": "fm_ffffffffff", "engine": "httpbin", "title": "Connect httpbin",
            "methods": [{
                "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
                "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
                "fields": [{"name": "api_key", "label": "API key", "type": "password"}],
            }],
        }

        await _probe(form, {"api_key": "typed-key"}, method="oauth")

        assert _Probe.runs == 1
        assert _rows(engine) == []

    @pytest.mark.asyncio
    async def test_an_oauth_flag_on_a_method_without_an_oauth_block_is_still_probed(self, engine, vault):
        _Probe.verdict = "failure"
        form = {
            "form_id": "fm_0000000000", "engine": "httpbin", "title": "Connect httpbin",
            "methods": [{"id": "oauth", "label": "Sign in", "submit_action": "oauth_launch"}],
        }

        await _probe(form, {"access_token": "t"}, method="oauth")

        assert _Probe.runs == 1
        assert _rows(engine) == []


class TestSubmit:
    @pytest.mark.asyncio
    async def test_an_invalid_handcrafted_form_is_a_422_before_anything_is_staged(self, monkeypatch):
        staged = []
        monkeypatch.setattr(submissions_endpoints.store, "stage", lambda **kw: staged.append(kw))

        async def run_db(fn, *, scope):
            return None

        monkeypatch.setattr(submissions_endpoints, "run_db", run_db)
        req = SubmitFormRequest(form_spec={"form_id": "fm_dddddddddd", "engine": "httpbin", "fields": [{"name": "token"}]})

        with pytest.raises(HTTPException) as exc:
            await submissions_endpoints.submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 422
        assert staged == []


OAUTH_METHOD = {
    "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
    "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
}


class TestUnstorableForms:
    """A model-written form is checked before it renders and before it is kept."""

    @pytest.mark.parametrize("form, why", [
        ({**HTTPBIN_FORM, "help_url": "http://example.com/help"}, "help_url"),
        ({**HTTPBIN_FORM, "help_url": "javascript:alert(1)"}, "help_url"),
        ({**HTTPBIN_FORM, "methods": [{**OAUTH_METHOD, "oauth": {
            "auth_url": "https://example.com/auth", "token_url": "http://example.com/token",
        }}]}, "oauth.token_url"),
        ({**HTTPBIN_FORM, "logo_color": "red; background: url(x)"}, "logo_color"),
        ({**HTTPBIN_FORM, "connector": {"usage_notes": "x" * 1601}}, "usage_notes"),
        ({**HTTPBIN_FORM, "methods": [{**OAUTH_METHOD, "id": "apiKey"}]}, "not valid"),
    ])
    def test_check_refuses(self, form, why):
        with pytest.raises(UnstorableFormError, match=why):
            check_handcrafted_form(form)

    def test_a_safe_form_passes(self):
        check_handcrafted_form({**HTTPBIN_FORM, "help_url": "https://example.com/help", "methods": [OAUTH_METHOD]})

    @pytest.mark.asyncio
    async def test_the_tool_refuses_before_the_form_renders(self, engine, monkeypatch):
        async def run_db(fn, *, scope):
            with Session(engine) as session:
                return fn(ScopedSession(session, scope))

        monkeypatch.setattr(tools, "run_db", run_db)
        result = await tools._cowork_request_credentials(None, {
            **HTTPBIN_FORM, "methods": [{**OAUTH_METHOD, "id": "apiKey"}],
        })

        assert "data-vault-form" not in result
        assert "not valid" in result

    @pytest.mark.asyncio
    async def test_submit_refuses_an_insecure_oauth_endpoint(self, monkeypatch):
        staged = []
        monkeypatch.setattr(submissions_endpoints.store, "stage", lambda **kw: staged.append(kw))

        async def run_db(fn, *, scope):
            return None

        monkeypatch.setattr(submissions_endpoints, "run_db", run_db)
        req = SubmitFormRequest(form_spec={**HTTPBIN_FORM, "methods": [{**OAUTH_METHOD, "oauth": {
            "auth_url": "http://example.com/auth", "token_url": "https://example.com/token",
        }}]})

        with pytest.raises(HTTPException) as exc:
            await submissions_endpoints.submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 422
        assert "auth_url" in exc.value.detail
        assert staged == []
