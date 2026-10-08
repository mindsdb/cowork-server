"""Which connector id a credential submission is saved under.

A handcrafted form declares its service in ``engine``; the id it resolves to
names the vault record, so it must be that declared engine and never a
synthesized ``fm_<hex>`` form id or an unsafe string.
"""

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors.submissions import submit_form
from cowork.db.scoped import LOCAL_SCOPE
from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
from cowork.schemas.connectors import (
    InvalidConnectorIdError,
    SubmitFormRequest,
    validate_engine_id,
)
from cowork.services.connectors.specs._registry import registry

_REJECTED_IDS = [
    "fm_ec163d25cf",
    "a-b",
    "../x",
    "Linked",
    "línkedin",
    "linkedin\n",
    "x",
    "1password",
    "a" * 65,
    "",
]


class TestResolveConnectorId:
    def test_handcrafted_spec_resolves_to_its_declared_engine(self):
        req = SubmitFormRequest(
            form_id="fm_ec163d25cf",
            form_spec={"form_id": "fm_ec163d25cf", "engine": "linkedin", "title": "LinkedIn"},
        )
        assert req.resolve_connector_id() == "linkedin"

    def test_registry_stamped_connector_id_wins_over_engine(self):
        req = SubmitFormRequest(
            form_id="gmail-connector",
            form_spec={"_connector_id": "gmail", "engine": "google-mail"},
        )
        assert req.resolve_connector_id() == "gmail"

    def test_explicit_connector_id_wins(self):
        req = SubmitFormRequest(connector_id="postgres", form_spec={"engine": "mysql"})
        assert req.resolve_connector_id() == "postgres"

    def test_legacy_form_id_fallthrough_still_resolves(self):
        req = SubmitFormRequest(form_id="postgres-connector")
        assert req.resolve_connector_id() == "postgres"

    def test_synthesized_form_id_never_becomes_an_engine(self):
        req = SubmitFormRequest(form_id="fm_ec163d25cf", form_spec={"form_id": "fm_ec163d25cf"})
        with pytest.raises(InvalidConnectorIdError):
            req.resolve_connector_id()

    @pytest.mark.parametrize("bad", [b for b in _REJECTED_IDS if b])
    def test_unsafe_declared_engine_is_rejected_not_rewritten(self, bad):
        req = SubmitFormRequest(form_spec={"engine": bad})
        with pytest.raises(InvalidConnectorIdError):
            req.resolve_connector_id()

    def test_nothing_naming_a_connector_is_an_error(self):
        with pytest.raises(ValueError, match="connector_id is required"):
            SubmitFormRequest(form_spec={"title": "x"}).resolve_connector_id()


class TestValidateEngineId:
    @pytest.mark.parametrize("bad", _REJECTED_IDS)
    def test_rejects(self, bad):
        with pytest.raises(InvalidConnectorIdError):
            validate_engine_id(bad)

    def test_rejects_non_strings(self):
        with pytest.raises(InvalidConnectorIdError):
            validate_engine_id(None)

    def test_every_registry_connector_id_is_valid(self):
        ids = list(registry.get_connectors())
        assert ids
        for connector_id in ids:
            assert validate_engine_id(connector_id) == connector_id


class TestSubmitEndpoint:
    @pytest.mark.asyncio
    async def test_unsafe_engine_is_a_422_before_anything_is_staged(self, monkeypatch):
        staged = []
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.submissions.store.stage",
            lambda **kw: staged.append(kw),
        )
        req = SubmitFormRequest(
            form_id="fm_ec163d25cf",
            form_spec={"form_id": "fm_ec163d25cf", "engine": "../etc", "fields": []},
        )
        with pytest.raises(HTTPException) as exc:
            await submit_form(req, LOCAL_SCOPE)
        assert exc.value.status_code == 422
        assert staged == []


class TestRequestCredentialsTool:
    @pytest.mark.asyncio
    async def test_spec_without_engine_or_connector_id_is_refused(self):
        result = await _cowork_request_credentials(session=None, tc_input={"title": "Connect"})
        assert "data-vault-form" not in result
        assert "engine" in result

    @pytest.mark.asyncio
    async def test_unsafe_engine_is_refused_before_the_form_renders(self):
        result = await _cowork_request_credentials(
            session=None, tc_input={"engine": "linked-in", "title": "Connect"},
        )
        assert "data-vault-form" not in result
        assert "Invalid connector id" in result

    @pytest.mark.asyncio
    async def test_handcrafted_spec_for_a_builtin_connector_is_refused(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={
                "engine": "postgres",
                "title": "Connect",
                "fields": [{"name": "host", "label": "Host", "type": "text"}],
            },
        )
        assert "data-vault-form" not in result
        assert "lookup_connector" in result

    @pytest.mark.asyncio
    async def test_extends_connection_on_a_builtin_form_is_refused(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={"_connector_id": "gmail", "title": "Gmail", "extends_connection": "nope"},
        )
        assert "data-vault-form" not in result
        assert "extends_connection" in result

    @pytest.mark.asyncio
    async def test_extends_connection_on_a_non_registry_stamped_form_renders(self):
        # `linkedin` is not a registry id, so copying it into `_connector_id`
        # still leaves the form handcrafted and mergeable.
        result = await _cowork_request_credentials(
            session=None,
            tc_input={
                "engine": "linkedin",
                "_connector_id": "linkedin",
                "title": "Finish",
                "extends_connection": "linkedin-1a2b3c4d",
            },
        )
        assert '"_extends_connection": "linkedin-1a2b3c4d"' in result

    @pytest.mark.asyncio
    async def test_stamped_connector_id_is_checked_instead_of_engine(self):
        result = await _cowork_request_credentials(
            session=None,
            tc_input={"_connector_id": "google_drive", "engine": "google-drive", "title": "Drive"},
        )
        assert "data-vault-form" in result
