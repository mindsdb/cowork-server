"""Every reader that asks "is this a connector?" sees a scope's custom connectors.

The agent's lookup and form tool, the form submission, the connections list
and direct save all read the same catalog, against a real SQLite database.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints
from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoints
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.harnesses.anton_harness import tools
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import DirectSaveRequest, SubmitFormRequest
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.persist import persist_connection
from tests._fakes import FakeRequest

HTTPBIN_FORM = {
    "form_id": "httpbin-connector",
    "title": "Connect httpbin",
    "fields": [
        {"name": "token", "label": "Bearer token", "type": "password", "required": True, "secret": True},
    ],
}


@pytest.fixture()
def engine(monkeypatch):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        session.add(CustomConnector(
            connector_id="httpbin", label="httpbin", description="Bearer test API",
            category="developer", logo_color="#3a7", spec=HTTPBIN_FORM,
        ))
        session.commit()

    async def run_db(fn, *, scope):
        with Session(eng) as session:
            return fn(ScopedSession(session, scope))

    @contextmanager
    def unit_session(*, scope):
        with Session(eng) as session:
            yield ScopedSession(session, scope)

    for module in (tools, submissions_endpoints, connections_endpoints):
        monkeypatch.setattr(module, "run_db", run_db)
    monkeypatch.setattr(connections_endpoints, "unit_session", unit_session)
    return eng


@pytest.fixture()
def staged(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(submissions_endpoints.store, "stage", lambda **kw: calls.append(kw) or "sub_test")

    class _NoProbe:
        def __init__(self, *, scope):
            pass

        async def run(self, *args):
            yield ""

    monkeypatch.setattr(submissions_endpoints, "ProbeHandler", _NoProbe)
    return calls


class TestAgentTools:
    @pytest.mark.asyncio
    async def test_lookup_by_id_returns_the_custom_form_stamped(self, engine):
        result = json.loads(await tools._cowork_lookup_connector(None, {"id": "httpbin"}))

        assert result["id"] == "httpbin"
        assert result["form"]["_connector_id"] == "httpbin"
        assert result["form"]["fields"][0]["name"] == "token"

    @pytest.mark.asyncio
    async def test_lookup_by_query_finds_the_custom_connector(self, engine):
        result = json.loads(await tools._cowork_lookup_connector(None, {"query": "httpbin"}))

        assert result["id"] == "httpbin"

    @pytest.mark.asyncio
    async def test_a_handcrafted_form_for_a_saved_custom_connector_is_refused(self, engine):
        result = await tools._cowork_request_credentials(
            None, {"engine": "httpbin", "title": "Again", "fields": []},
        )

        assert "data-vault-form" not in result
        assert "lookup_connector" in result

    @pytest.mark.asyncio
    async def test_a_handcrafted_follow_up_step_of_a_custom_connector_renders(self, engine):
        result = await tools._cowork_request_credentials(
            None,
            {"engine": "httpbin", "title": "Finish", "fields": [], "extends_connection": "httpbin-1a2b3c4d"},
        )

        assert '"_extends_connection": "httpbin-1a2b3c4d"' in result


class TestSubmitForm:
    @pytest.mark.asyncio
    async def test_a_handcrafted_form_reusing_a_custom_id_is_a_422(self, engine, staged):
        req = SubmitFormRequest(form_spec={"form_id": "fm_aaaaaaaaaa", "engine": "httpbin", "fields": []})

        with pytest.raises(HTTPException) as exc:
            await submissions_endpoints.submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 422
        assert staged == []

    @pytest.mark.asyncio
    async def test_a_stamped_custom_form_is_checked_against_the_stored_spec(self, engine, staged):
        req = SubmitFormRequest(form_spec={"_connector_id": "httpbin", "form_id": "httpbin-connector"}, values={})

        with pytest.raises(HTTPException) as exc:
            await submissions_endpoints.submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 400
        assert "token" in exc.value.detail

    @pytest.mark.asyncio
    async def test_a_stamped_custom_submission_stages_the_stored_spec(self, engine, staged):
        req = SubmitFormRequest(
            form_spec={"_connector_id": "httpbin", "form_id": "httpbin-connector"},
            values={"token": "any-token"},
        )

        await submissions_endpoints.submit_form(req, LOCAL_SCOPE)

        [call] = staged
        assert call["connector_id"] == "httpbin"
        assert call["custom_spec"]["custom"] is True
        assert call["custom_spec"]["form"]["fields"][0]["name"] == "token"

    @pytest.mark.asyncio
    async def test_a_handcrafted_follow_up_keeps_its_own_form(self, engine, staged, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("httpbin", "httpbin-1a2b3c4d", {"token": "t", "_connector_id": "httpbin"}, secure_keys=["token"])
        monkeypatch.setattr(submissions_endpoints, "vault_for_scope", lambda scope: vault)
        req = SubmitFormRequest(
            form_spec={
                "form_id": "fm_bbbbbbbbbb", "engine": "httpbin", "title": "httpbin", "fields": [],
                "_extends_connection": "httpbin-1a2b3c4d",
            },
            values={"refresh": "r"},
        )

        await submissions_endpoints.submit_form(req, LOCAL_SCOPE)

        [call] = staged
        assert call["form_id"] == "fm_bbbbbbbbbb"
        assert call["custom_spec"] is None


class TestConnectionsAndSaves:
    @pytest.mark.asyncio
    async def test_the_connections_list_names_a_custom_connection(self, engine, tmp_path, monkeypatch):
        vault_dir = Path(tmp_path) / "vault"
        LocalDataVault(vault_dir).save("httpbin", "httpbin-1a2b3c4d", {"token": "t"}, secure_keys=["token"])
        monkeypatch.setattr(
            "cowork.services.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(vault_dir)})(),
        )

        [card] = await connections_endpoints.list_connections(LOCAL_SCOPE, FakeRequest())

        assert card.label == "httpbin"
        assert card.logo_color == "#3a7"
        assert card.custom is True

    def test_the_service_defaults_to_the_static_registry(self, tmp_path, monkeypatch):
        vault_dir = Path(tmp_path) / "vault"
        LocalDataVault(vault_dir).save("postgres", "db", {"host": "h"}, secure_keys=[])
        monkeypatch.setattr(
            "cowork.services.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(vault_dir)})(),
        )

        [card] = ConnectionsService(LOCAL_SCOPE).list()

        assert card.label and card.custom is False

    def test_direct_save_accepts_a_custom_connector(self, engine, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(Path(tmp_path) / "vault")})(),
        )
        body = DirectSaveRequest(connector_id="httpbin", values={"token": "any-token"})

        result = connections_endpoints.save_connection_direct(body, LOCAL_SCOPE)

        assert result["ok"] is True
        saved = LocalDataVault(Path(tmp_path) / "vault").read_record("httpbin", result["name"])
        assert saved["engine"] == "httpbin"

    def test_direct_save_still_refuses_an_unknown_connector(self, engine):
        body = DirectSaveRequest(connector_id="nothing_here", values={})

        with pytest.raises(HTTPException) as exc:
            connections_endpoints.save_connection_direct(body, LOCAL_SCOPE)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_starting_oauth_for_a_custom_connector_is_a_clear_404(self, engine):
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.start_oauth("httpbin", FakeRequest(), LOCAL_SCOPE)
        assert exc.value.status_code == 404


class TestSecretFieldsFromNonRegistrySpecs:
    """A field a custom or handcrafted spec marks secret is masked, even when
    its name doesn't look like a secret."""

    FORM = {
        "form_id": "signer-connector",
        "title": "Connect signer",
        "fields": [
            {"name": "account", "label": "Account", "type": "text"},
            {"name": "signing_material", "label": "Signing material", "type": "text", "secret": True},
        ],
    }

    def test_the_spec_form_marks_the_field_secret(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "signer", None, "", {"account": "a1", "signing_material": "m"},
            spec_form=self.FORM, vault=vault,
        )

        assert "signing_material" in vault.read_record("signer", slug)["secure_keys"]

    def test_without_the_form_the_name_check_misses_it(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "signer", None, "", {"account": "a1", "signing_material": "m"}, vault=vault,
        )

        assert "signing_material" not in vault.read_record("signer", slug)["secure_keys"]

    def test_a_registry_connector_still_reads_its_own_spec(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection("postgres", None, "", {"host": "h", "password": "p"}, vault=vault)

        assert "password" in vault.read_record("postgres", slug)["secure_keys"]

    def test_direct_save_of_a_custom_connector_uses_its_stored_form(self, engine, tmp_path, monkeypatch):
        with Session(engine) as session:
            session.add(CustomConnector(
                connector_id="signer", label="Signer", spec=self.FORM, description="",
            ))
            session.commit()
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(Path(tmp_path) / "vault")})(),
        )
        body = DirectSaveRequest(connector_id="signer", values={"account": "a1", "signing_material": "m"})

        result = connections_endpoints.save_connection_direct(body, LOCAL_SCOPE)

        record = LocalDataVault(Path(tmp_path) / "vault").read_record("signer", result["name"])
        assert "signing_material" in record["secure_keys"]


class TestBuildFlowPrompt:
    @pytest.mark.asyncio
    async def test_the_connector_block_reaches_the_form(self, engine):
        result = await tools._cowork_request_credentials(None, {
            "engine": "kinaxis", "title": "Connect Kinaxis",
            "fields": [{"name": "api_key", "label": "API key", "type": "password", "secret": True}],
            "connector": {"label": "Kinaxis", "category": "erp", "usage_notes": "Base URL is per tenant."},
        })

        block = json.loads(result.split("```data-vault-form\n", 1)[1].rsplit("\n```", 1)[0])
        assert block["connector"] == {"label": "Kinaxis", "category": "erp", "usage_notes": "Base URL is per tenant."}

    def test_the_schema_declares_the_connector_block(self):
        props = tools._REQUEST_CREDENTIALS_SCHEMA["properties"]["connector"]["properties"]

        assert set(props) == {"label", "description", "category", "usage_notes"}

    @pytest.mark.asyncio
    async def test_a_failed_lookup_points_to_building_a_custom_connector(self, engine):
        result = json.loads(await tools._cowork_lookup_connector(None, {"query": "kinaxis rapidresponse"}))

        assert result["match"] == "none"
        assert "custom connector" in result["message"]
        assert "request_credentials" in result["message"]
