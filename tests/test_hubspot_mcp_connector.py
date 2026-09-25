"""HubSpot's MCP connector (ENG-487, Stage 2) — the pieces that are new,
not generic across every connector: the connect-time `_access_mode` default
that must survive a reconnect, the `access-mode` edit endpoint, and the
MCP-based identity bridge Electron calls in place of a direct provider call.

Direct function calls, no TestClient/app fixture — matches this repo's
existing convention (see test_connectors_endpoints.py, test_connections_org_mode.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import connections as connections_endpoints
from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints
from cowork.api.v1.endpoints.connectors.connections import (
    PatchAccessModeBody,
    save_connection_direct,
)
from cowork.api.v1.endpoints.connectors.oauth import McpIdentityRequest, _parse_mcp_identity
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.schemas.connectors import DirectSaveRequest
from cowork.services.connectors.persist import persist_connection
from tests._fakes import FakeRequest

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")


def _vault(tmp_path, monkeypatch):
    fake_settings = lambda: type("S", (), {"vault_dir": str(tmp_path / "vault")})()
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings", fake_settings,
    )
    # ConnectionsService (used by patch_access_mode's local-mode branch) reads
    # its own module-level ConnectorSettings import, separate from the one
    # above — both must point at the same test vault.
    monkeypatch.setattr(
        "cowork.services.connectors.connections.ConnectorSettings", fake_settings,
    )


class TestAccessModeDefaultsAndSurvivesReconnect:
    def test_new_mcp_connection_defaults_access_mode_to_read(self, tmp_path, monkeypatch):
        _vault(tmp_path, monkeypatch)
        body = DirectSaveRequest(
            connector_id="hubspot", method="mcp", name="",
            values={"access_token": "tok-1", "account_email": "acme.hubspot"},
        )
        save_connection_direct(body, LOCAL_SCOPE)

        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        record = vault.read_record("hubspot", "acme-hubspot")
        assert record["fields"]["_access_mode"] == "read"

    def test_non_mcp_connection_gets_no_access_mode_field(self, tmp_path, monkeypatch):
        _vault(tmp_path, monkeypatch)
        body = DirectSaveRequest(
            connector_id="hubspot", method="private-app", name="",
            values={"access_token": "pat-na1-xxx"},
        )
        save_connection_direct(body, LOCAL_SCOPE)

        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        # No identity field on private-app credentials -> random slug; just
        # confirm no connection anywhere picked up an _access_mode key.
        for conn in vault.list_connections():
            record = vault.read_record(conn["engine"], conn["name"])
            assert "_access_mode" not in record["fields"]

    def test_reconnect_does_not_reset_an_upgraded_access_mode_back_to_read(self, tmp_path, monkeypatch):
        """The exact bug found in artifact review: a token-refresh reconnect
        reaching the same save path must not clobber a user's "write" upgrade."""
        _vault(tmp_path, monkeypatch)
        first = DirectSaveRequest(
            connector_id="hubspot", method="mcp", name="",
            values={"access_token": "tok-1", "account_email": "acme.hubspot"},
        )
        result = save_connection_direct(first, LOCAL_SCOPE)

        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        # Simulate the user having used "edit access" to upgrade to write.
        record = vault.read_record("hubspot", result["name"])
        fields = dict(record["fields"])
        fields["_access_mode"] = "write"
        vault.save("hubspot", result["name"], fields, secure_keys=record.get("secure_keys"))

        # HubSpot session expires; Electron re-authorizes and calls /save again
        # for the SAME account (same identity field -> same slug, is_edit path).
        reconnect = DirectSaveRequest(
            connector_id="hubspot", method="mcp", name="",
            values={"access_token": "tok-2", "account_email": "acme.hubspot"},
        )
        save_connection_direct(reconnect, LOCAL_SCOPE)

        record = vault.read_record("hubspot", result["name"])
        assert record["fields"]["_access_mode"] == "write"
        assert record["fields"]["access_token"] == "tok-2"


class TestPersistConnectionDefaultFields:
    def test_default_fields_applied_only_to_a_genuinely_new_connection(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        persist_connection(
            "hubspot", "mcp", "", {"access_token": "tok", "account_email": "a.hubspot"},
            default_fields={"_access_mode": "read"}, vault=vault,
        )
        record = vault.read_record("hubspot", "a-hubspot")
        assert record["fields"]["_access_mode"] == "read"

    def test_default_fields_never_overwrites_an_existing_value(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        slug = persist_connection(
            "hubspot", "mcp", "", {"access_token": "tok-1", "account_email": "a.hubspot"},
            default_fields={"_access_mode": "read"}, vault=vault,
        )
        record = vault.read_record("hubspot", slug)
        fields = dict(record["fields"])
        fields["_access_mode"] = "write"
        vault.save("hubspot", slug, fields, secure_keys=record.get("secure_keys"))

        persist_connection(
            "hubspot", "mcp", "", {"access_token": "tok-2", "account_email": "a.hubspot"},
            default_fields={"_access_mode": "read"}, vault=vault,
        )
        record = vault.read_record("hubspot", slug)
        assert record["fields"]["_access_mode"] == "write"

    def test_backfills_a_default_field_on_a_reconnect_of_a_record_that_predates_it(self, tmp_path):
        """Found in code review: a record saved before `_access_mode` existed
        (or migrated from private-app) reconnecting via the mcp method must
        still end up with the field set, not silently left absent."""
        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        # Simulate a legacy record with no _access_mode key at all.
        vault.save("hubspot", "a-hubspot", {"access_token": "tok-1", "account_email": "a.hubspot", "_connector_id": "hubspot"})

        persist_connection(
            "hubspot", "mcp", "", {"access_token": "tok-2", "account_email": "a.hubspot"},
            default_fields={"_access_mode": "read"}, vault=vault,
        )
        record = vault.read_record("hubspot", "a-hubspot")
        assert record["fields"]["_access_mode"] == "read"


class TestPatchAccessMode:
    async def test_local_mode_writes_the_vault_field_directly(self, tmp_path, monkeypatch):
        _vault(tmp_path, monkeypatch)
        save_connection_direct(
            DirectSaveRequest(
                connector_id="hubspot", method="mcp", name="",
                values={"access_token": "tok", "account_email": "acme.hubspot"},
            ),
            LOCAL_SCOPE,
        )
        result = await connections_endpoints.patch_access_mode(
            "hubspot", "acme-hubspot", PatchAccessModeBody(access_mode="write"), LOCAL_SCOPE, FakeRequest(),
        )
        assert result == {"ok": True, "access_mode": "write"}

        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(tmp_path / "vault")
        assert vault.read_record("hubspot", "acme-hubspot")["fields"]["_access_mode"] == "write"

    async def test_local_mode_404s_for_an_unknown_connection(self, tmp_path, monkeypatch):
        _vault(tmp_path, monkeypatch)
        with pytest.raises(HTTPException) as exc:
            await connections_endpoints.patch_access_mode(
                "hubspot", "nope", PatchAccessModeBody(access_mode="write"), LOCAL_SCOPE, FakeRequest(),
            )
        assert exc.value.status_code == 404

    async def test_rejects_an_invalid_access_mode(self, tmp_path, monkeypatch):
        _vault(tmp_path, monkeypatch)
        with pytest.raises(HTTPException) as exc:
            await connections_endpoints.patch_access_mode(
                "hubspot", "acme-hubspot", PatchAccessModeBody(access_mode="admin"), LOCAL_SCOPE, FakeRequest(),
            )
        assert exc.value.status_code == 422

    async def test_org_mode_proxies_to_auth(self, monkeypatch):
        calls = []

        async def fake_proxy_access_mode(engine, name, access_mode, request, settings):
            calls.append((engine, name, access_mode))
            return {"ok": True, "access_mode": access_mode}

        monkeypatch.setattr(connections_endpoints.auth_proxy, "proxy_access_mode", fake_proxy_access_mode)

        result = await connections_endpoints.patch_access_mode(
            "hubspot", "acme-hubspot", PatchAccessModeBody(access_mode="write"), ORG_SCOPE, FakeRequest(),
        )
        assert result == {"ok": True, "access_mode": "write"}
        assert calls == [("hubspot", "acme-hubspot", "write")]


#: These tests monkeypatch `anton.core.mcp.wiring.call_mcp_tool`, which needs
#: the module to exist. cowork-server's published anton floor
#: (`anton-agent>=2.26.8.23.1`) still resolves builds that predate the MCP
#: client — promoting it to anton's `main` is a separate release step
#: (ENG-1816) — so on those builds there is genuinely nothing to exercise.
#: Skipping is the honest outcome; the endpoint's own behaviour under that
#: skew is covered by `TestMcpIdentityBridgeWithoutAntonMcp` below, which
#: runs either way.
try:  # pragma: no cover - import probe, not logic
    import anton.core.mcp.wiring  # noqa: F401

    HAS_ANTON_MCP = True
except ImportError:  # pragma: no cover
    HAS_ANTON_MCP = False


@pytest.mark.skipif(not HAS_ANTON_MCP, reason="installed anton has no anton.core.mcp (ENG-1816 not yet on anton main)")
class TestMcpIdentityBridge:
    async def test_resolves_email_and_portal_name_from_json_string_tool_results(self, monkeypatch):
        """Real shapes, live-verified 2026-09-15 against an actual HubSpot
        MCP server — both nest the useful fields one level down."""
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            assert engine == "hubspot"
            assert access_token == "tok"
            if tool_name == "get_user_details":
                return json.dumps({
                    "userInformation": {"email": "user@acme.com", "firstName": "Jordan", "lastName": "Lee"},
                })
            return json.dumps({"accountInformation": {"portalName": "Acme Inc"}})

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

        result = await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert result == {"account_email": "user@acme.com", "account_name": "Acme Inc"}

    async def test_falls_back_to_person_name_when_no_portal_name(self, monkeypatch):
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            if tool_name == "get_user_details":
                return {"userInformation": {"email": "user@acme.com", "firstName": "Jordan", "lastName": "Lee"}}
            return {"accountInformation": {}}

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

        result = await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert result == {"account_email": "user@acme.com", "account_name": "Jordan Lee"}

    async def test_missing_email_502s_instead_of_returning_a_blank_identity(self, monkeypatch):
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            return "{}"

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert exc.value.status_code == 502

    async def test_a_tool_call_failure_502s_cleanly(self, monkeypatch):
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            raise RuntimeError("hubspot MCP tool get_user_details returned an error: token rejected")

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert exc.value.status_code == 502

    async def test_an_organization_lookup_failure_does_not_discard_a_successful_email(self, monkeypatch):
        """Found in code review: get_user_details and get_organization_details
        were both under one try/except, so an org-lookup failure discarded an
        already-successful email instead of degrading to email-only identity."""
        async def fake_call_mcp_tool(engine, access_token, tool_name, **kwargs):
            if tool_name == "get_user_details":
                return json.dumps({
                    "userInformation": {"email": "user@acme.com", "firstName": "Jordan", "lastName": "Lee"},
                })
            raise RuntimeError("hubspot MCP tool get_organization_details returned an error: insufficient scope")

        monkeypatch.setattr("anton.core.mcp.wiring.call_mcp_tool", fake_call_mcp_tool)

        result = await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert result == {"account_email": "user@acme.com", "account_name": "Jordan Lee"}


class TestMcpIdentityBridgeWithoutAntonMcp:
    """The two paths that must hold on an anton with no MCP client.

    Deliberately NOT skipped: these run on every build, including ones that
    do have `anton.core.mcp`, by making the import fail on demand — otherwise
    the skew behaviour would only ever be exercised on the builds least
    likely to be the ones running CI.
    """

    async def test_unknown_engine_404s_before_anton_is_imported_at_all(self):
        """The engine guard precedes the import, so this answers correctly
        even where there is no MCP client to import."""
        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("linear", McpIdentityRequest(access_token="tok"))
        assert exc.value.status_code == 404

    async def test_501_when_the_installed_anton_has_no_mcp_client(self, monkeypatch):
        """A `None` in sys.modules makes `import anton.core.mcp.wiring` raise
        ImportError, which is exactly what an older anton does. Before the
        guard this surfaced to Electron as an opaque 500."""
        monkeypatch.setitem(sys.modules, "anton.core.mcp.wiring", None)

        with pytest.raises(HTTPException) as exc:
            await oauth_endpoints.get_mcp_identity("hubspot", McpIdentityRequest(access_token="tok"))
        assert exc.value.status_code == 501


class TestParseMcpIdentity:
    def test_tolerates_dict_input_not_just_json_strings(self):
        email, name = _parse_mcp_identity(
            {"userInformation": {"email": "a@b.com", "firstName": "Ann", "lastName": "Lee"}},
            {"accountInformation": {"portalName": "Org"}},
        )
        assert (email, name) == ("a@b.com", "Org")

    def test_returns_empty_strings_on_unparseable_input(self):
        assert _parse_mcp_identity("not json", "also not json") == ("", "")


class TestOAuthIsTheOnlyOfferedMethod:
    """HubSpot's App Marketplace requires OAuth as the app's *sole*
    authorization method, so the connect UI must offer nothing else.
    `DataVaultForm.jsx` filters the picker on `!m.hidden`, so a method is
    "offered" exactly when it is not hidden.

    `private-app` stays defined rather than deleted on purpose: it is the
    only HubSpot method that was ever live for users, and existing vault
    records still carry `method: "private-app"`. Hiding retires it from the
    UI without breaking those connections; deleting would break them.

    If a listing submission is ever withdrawn and PAT is deliberately
    brought back, this test is the thing to change — do not just unhide the
    method and leave the marketplace listing standing.
    """

    @staticmethod
    def _methods():
        spec_path = (
            Path(__file__).parent.parent
            / "cowork" / "services" / "connectors" / "specs" / "hubspot.json"
        )
        return json.loads(spec_path.read_text())["form"]["methods"]

    def test_exactly_one_method_is_offered_and_it_is_oauth(self):
        offered = [m for m in self._methods() if not m.get("hidden")]
        assert [m["id"] for m in offered] == ["mcp"]
        assert offered[0]["oauth"]["auth_url"].startswith("https://mcp.hubspot.com/")

    def test_the_private_app_token_method_is_retired_but_still_defined(self):
        by_id = {m["id"]: m for m in self._methods()}
        assert by_id["private-app"]["hidden"] is True
        # Still carries its field, so a stored PAT connection stays readable.
        assert [f["name"] for f in by_id["private-app"]["fields"]] == ["access_token"]
