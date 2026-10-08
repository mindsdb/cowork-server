"""A multi-step handcrafted connect lands in one vault record.

App credentials are saved first; the tokens from the OAuth grant arrive in a
later form. Without ``extends`` each step saved a separate record holding
only part of the connection, so no record was usable.
"""

from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors.submissions import submit_form
from cowork.db.scoped import LOCAL_SCOPE
from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
from cowork.schemas.connectors import SubmitFormRequest
from cowork.services.connectors.persist import ConnectionNotFoundError, persist_connection

APP_CREDENTIALS = {
    "client_id": "86nwdt9sl34cuy",
    "client_secret": "app-secret",
    "redirect_uri": "http://localhost:8765/callback",
}
GRANT = {
    "client_id": "86nwdt9sl34cuy",
    "access_token": "access-1",
    "refresh_token": "refresh-1",
    "organization_urn": "urn:li:organization:27204044",
}


@pytest.fixture
def vault(tmp_path):
    return LocalDataVault(Path(tmp_path) / "vault")


class TestPersistExtends:
    def test_follow_up_step_merges_into_the_first_record(self, vault):
        slug = persist_connection("linkedin", "app", "", APP_CREDENTIALS, vault=vault)

        again = persist_connection("linkedin", "oauth", slug, GRANT, extends=True, vault=vault)

        assert again == slug
        assert len(vault.list_connections()) == 1
        fields = vault.read_record("linkedin", slug)["fields"]
        for key in ("client_id", "client_secret", "redirect_uri", "access_token",
                    "refresh_token", "organization_urn"):
            assert key in fields
        assert fields["access_token"] == "access-1"

    def test_merged_record_keeps_secrets_masked(self, vault):
        slug = persist_connection("linkedin", "app", "", APP_CREDENTIALS, vault=vault)
        persist_connection("linkedin", "oauth", slug, GRANT, extends=True, vault=vault)

        secure = set(vault.read_record("linkedin", slug)["secure_keys"])
        assert {"client_secret", "access_token", "refresh_token"} <= secure

    def test_resubmitted_field_replaces_the_stored_one(self, vault):
        slug = persist_connection("linkedin", "app", "", APP_CREDENTIALS, vault=vault)
        persist_connection(
            "linkedin", "oauth", slug, {**GRANT, "client_secret": "rotated"},
            extends=True, vault=vault,
        )
        assert vault.read_record("linkedin", slug)["fields"]["client_secret"] == "rotated"

    def test_extending_a_missing_connection_saves_nothing(self, vault):
        with pytest.raises(ConnectionNotFoundError):
            persist_connection("linkedin", "oauth", "nope", GRANT, extends=True, vault=vault)
        assert vault.list_connections() == []

    def test_extending_under_another_engine_does_not_find_the_record(self, vault):
        slug = persist_connection("linkedin", "app", "", APP_CREDENTIALS, vault=vault)
        with pytest.raises(ConnectionNotFoundError):
            persist_connection("twitter", "oauth", slug, GRANT, extends=True, vault=vault)

    def test_without_extends_a_second_step_is_still_a_separate_record(self, vault):
        persist_connection("linkedin", "app", "", APP_CREDENTIALS, vault=vault)
        persist_connection("linkedin", "oauth", "", GRANT, vault=vault)
        assert len(vault.list_connections()) == 2


class TestSubmitEndpointExtends:
    @pytest.mark.asyncio
    async def test_unknown_extends_target_is_a_422(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.submissions.vault_for_scope",
            lambda scope: vault,
        )
        staged = []
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.submissions.store.stage",
            lambda **kw: staged.append(kw),
        )
        req = SubmitFormRequest(
            form_spec={
                "form_id": "fm_ec163d25cf",
                "engine": "linkedin",
                "_extends_connection": "linkedin-deadbeef",
                "fields": [],
            },
        )
        with pytest.raises(HTTPException) as exc:
            await submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 422
        assert staged == []


class TestRequestCredentialsExtends:
    @pytest.mark.asyncio
    async def test_extends_connection_is_stamped_for_the_renderer_and_server(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={"engine": "linkedin", "title": "Finish", "extends_connection": "linkedin-1a2b3c4d"},
        )
        assert '"_extends_connection": "linkedin-1a2b3c4d"' in result
        assert '"_existing_name": "linkedin-1a2b3c4d"' in result
        assert '"extends_connection"' not in result

    @pytest.mark.asyncio
    async def test_blank_extends_connection_is_refused(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={"engine": "linkedin", "title": "Finish", "extends_connection": "  "},
        )
        assert "data-vault-form" not in result
