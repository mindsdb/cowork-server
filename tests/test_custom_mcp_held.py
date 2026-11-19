"""A custom MCP server's settings and token come only from server memory.

Agent code can read and edit the vault file and call the loopback routes,
but cannot read the app secret. These tests write the file as agent code
would and check that turns read only the held entry, that every generic
writer refuses ``custom_mcp``, and that nothing serves it as a connector.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from anton.core.datasources.data_vault import LocalDataVault, withheld_fields
from anton.utils.datasources import clear_ds_env, restore_namespaced_env, scrub_credentials
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork import app_secret
from cowork.api.v1.router import api_router
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import InvalidConnectorIdError, validate_engine_id
from cowork.services.connectors import app_held
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.custom_connectors import CustomConnectorService, UnstorableFormError
from cowork.services.connectors.custom_mcp import CUSTOM_MCP_ENGINE, ReservedEngineError, is_connection_id
from cowork.services.connectors.layered_vault import FilteredDataVault, LayeredDataVault
from cowork.services.connectors.persist import own_vault_for_scope, own_vault_root, turn_vault_for_scope
from tests._connector_test_request import with_test_request

SECRET = "ab" * 32
HEADERS = {app_secret.HEADER: SECRET}
BASE = "/api/v1/connectors/connections"
CONN = "cm-" + "0123456789abcdef" * 2
HELD = {
    "server_url": "https://mcp.example.com/mcp",
    "label": "Acme Tracker",
    "resource": "https://mcp.example.com/mcp",
    "token_endpoint": "https://auth.example.com/token",
    "client_id": "client-1",
    "allowed_tools": ["search_issues"],
    "access_token": "AT-held",
    "expires_at": "2030-01-01T00:00:00Z",
}
AGENT_WRITTEN = {
    "_method": "mcp",
    "_server_url": "https://attacker.example/mcp",
    "_allowed_tools": json.dumps(["delete_everything"]),
    "_label": "Renamed by agent",
    "_access_mode": "write",
    "access_token": "AT-agent",
}


@pytest.fixture
def stores(sharepoint_stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", None)
    app_held._entries.clear()
    app_held._custom.clear()
    yield sharepoint_stores
    app_held._entries.clear()
    app_held._custom.clear()


@pytest.fixture
def protected(stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", SECRET)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _write_file(fields: dict) -> None:
    """The vault file, written directly, as agent code could."""
    own_vault_for_scope(LOCAL_SCOPE).save(CUSTOM_MCP_ENGINE, CONN, dict(fields))


def _turn_vault():
    source = turn_vault_for_scope(LOCAL_SCOPE)
    return app_held.turn_vault(source, source_vault=source, root=own_vault_root(LOCAL_SCOPE))


def _hold(values: dict = HELD) -> None:
    app_held.hold_custom(own_vault_root(LOCAL_SCOPE), CONN, values)


EXPECTED_RECORD_FIELDS = {
    "_method": "mcp",
    "_server_url": "https://mcp.example.com/mcp",
    "_allowed_tools": json.dumps(["search_issues"]),
    "_label": "Acme Tracker",
    "_access_mode": "write",
}
RECORD_WITH_TOKEN = {**EXPECTED_RECORD_FIELDS, "access_token": "AT-held"}
HELD_FIELDS = {**RECORD_WITH_TOKEN, "expires_at": "2030-01-01T00:00:00Z"}


class TestTurnsReadOnlyHeldSettings:
    def test_a_turn_reads_the_held_settings_not_the_file(self, protected):
        _write_file(AGENT_WRITTEN)
        _hold()
        vault = _turn_vault()
        assert vault.read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == RECORD_WITH_TOKEN
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) == HELD_FIELDS

    def test_a_tool_result_that_echoes_the_held_token_is_scrubbed(self, protected):
        _write_file(AGENT_WRITTEN)
        _hold()
        vault = _turn_vault()
        try:
            restore_namespaced_env(vault)
            scrubbed = scrub_credentials("401 from the server: invalid bearer AT-held")
        finally:
            clear_ds_env()
        assert "AT-held" not in scrubbed
        # Withheld from the environment and the agent's listing all the same.
        assert withheld_fields(vault.read_record(CUSTOM_MCP_ENGINE, CONN)) == ["access_token"]
        assert "AT-held" not in (vault.env_for(CUSTOM_MCP_ENGINE, CONN) or {}).values()

    def test_with_nothing_held_the_connection_carries_its_method_only(self, protected):
        _write_file(AGENT_WRITTEN)
        vault = _turn_vault()
        assert vault.read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == {"_method": "mcp"}
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) == {"_method": "mcp"}

    def test_a_file_swapped_after_the_hold_changes_nothing(self, protected):
        _write_file({"_method": "mcp"})
        _hold()
        vault = _turn_vault()
        before = vault.load(CUSTOM_MCP_ENGINE, CONN)
        _write_file(AGENT_WRITTEN)
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) == before

    def test_an_old_copy_of_the_file_restored_changes_nothing(self, protected):
        _write_file({"_method": "mcp", **{k: v for k, v in AGENT_WRITTEN.items() if k != "_method"}})
        root = own_vault_root(LOCAL_SCOPE)
        saved = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        _hold()
        for path, content in saved.items():
            path.write_bytes(content)
        assert _turn_vault().read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == RECORD_WITH_TOKEN

    def test_deleting_through_the_turn_vault_drops_the_held_entry(self, protected):
        _write_file({"_method": "mcp"})
        _hold()
        _turn_vault().delete(CUSTOM_MCP_ENGINE, CONN)
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN) == {}
        _write_file(AGENT_WRITTEN)
        assert _turn_vault().load(CUSTOM_MCP_ENGINE, CONN) == {"_method": "mcp"}

    def test_the_generic_delete_route_needs_the_app_secret(self, protected):
        _write_file({"_method": "mcp"})
        _hold()
        response = _client().delete(f"{BASE}/{CUSTOM_MCP_ENGINE}/{CONN}")
        assert response.status_code == 403
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN) == HELD
        assert own_vault_for_scope(LOCAL_SCOPE).read_record(CUSTOM_MCP_ENGINE, CONN) is not None

    @pytest.mark.parametrize(
        ("engine", "name"),
        [
            ("custom.mcp", CONN),
            ("custom%20mcp", CONN),
            ("_custom_mcp", CONN),
            ("custom_mcp_", CONN),
            ("CUSTOM_MCP", CONN),
            (CUSTOM_MCP_ENGINE, "cm-" + CONN[3:].upper()),
            (CUSTOM_MCP_ENGINE, CONN + "."),
        ],
    )
    def test_no_other_spelling_deletes_it_without_the_app_secret(self, protected, engine, name):
        _write_file({"_method": "mcp"})
        _hold()
        response = _client().delete(f"{BASE}/{engine}/{name}")
        # 404 where the disk tells names apart by case, 403 where it doesn't.
        assert response.status_code in (403, 404)
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN) == HELD
        assert own_vault_for_scope(LOCAL_SCOPE).read_record(CUSTOM_MCP_ENGINE, CONN) is not None

    def test_the_generic_delete_route_drops_the_held_entry(self, protected):
        _write_file({"_method": "mcp"})
        _hold()
        response = _client().delete(f"{BASE}/{CUSTOM_MCP_ENGINE}/{CONN}", headers=HEADERS)
        assert response.status_code in (200, 204)
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN) == {}
        _write_file(AGENT_WRITTEN)
        assert _turn_vault().load(CUSTOM_MCP_ENGINE, CONN) == {"_method": "mcp"}

    def test_a_record_in_a_managed_layer_reads_as_nothing_held(self, protected, sharepoint_stores):
        managed = LocalDataVault(vault_dir=sharepoint_stores / "managed")
        managed.save(CUSTOM_MCP_ENGINE, CONN, {"_method": "mcp"})
        _hold()
        layered = LayeredDataVault(own=own_vault_for_scope(LOCAL_SCOPE), managed=managed)
        vault = app_held.turn_vault(layered, source_vault=layered, root=own_vault_root(LOCAL_SCOPE))
        assert layered.answers_from_own(CUSTOM_MCP_ENGINE, CONN) is False
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) == {"_method": "mcp"}

    def test_the_connecting_view_still_reads_from_memory_and_refuses_saves(self, protected):
        _write_file(AGENT_WRITTEN)
        _hold()
        connecting = _turn_vault().for_connecting()
        assert connecting.load(CUSTOM_MCP_ENGINE, CONN)["_server_url"] == "https://mcp.example.com/mcp"
        with pytest.raises(ReservedEngineError):
            connecting.save(CUSTOM_MCP_ENGINE, CONN, dict(AGENT_WRITTEN))

    def test_a_turned_off_connection_reads_as_its_label_alone_in_the_protected_connecting_view(self, protected):
        own_vault_for_scope(LOCAL_SCOPE).save("postgres", "erp", {"host": "db.example", "password": "pw", "_label": "ERP"})
        source = turn_vault_for_scope(LOCAL_SCOPE)
        filtered = FilteredDataVault(source=source, hidden=frozenset({("postgres", "erp")}))
        vault = app_held.turn_vault(filtered, source_vault=source, root=own_vault_root(LOCAL_SCOPE))
        assert vault.load("postgres", "erp") is None
        loaded = vault.for_connecting().load("postgres", "erp")
        assert loaded is not None and "password" not in loaded and "host" not in loaded

    def test_a_turned_off_custom_connection_reads_as_its_label_alone_in_the_protected_connecting_view(self, protected):
        _write_file({"_method": "mcp"})
        _hold()
        source = turn_vault_for_scope(LOCAL_SCOPE)
        filtered = FilteredDataVault(source=source, hidden=frozenset({(CUSTOM_MCP_ENGINE, CONN)}))
        vault = app_held.turn_vault(filtered, source_vault=source, root=own_vault_root(LOCAL_SCOPE))
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) is None
        connecting = vault.for_connecting()
        assert connecting.load(CUSTOM_MCP_ENGINE, CONN) == {}
        assert connecting.read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == {}

    def test_a_turned_off_custom_connection_keeps_only_its_file_labels_in_the_connecting_view(self, protected):
        _write_file({**AGENT_WRITTEN, "_user_label": "Tracker"})
        _hold()
        source = turn_vault_for_scope(LOCAL_SCOPE)
        filtered = FilteredDataVault(source=source, hidden=frozenset({(CUSTOM_MCP_ENGINE, CONN)}))
        connecting = app_held.turn_vault(filtered, source_vault=source, root=own_vault_root(LOCAL_SCOPE)).for_connecting()
        labels = {"_label": "Renamed by agent", "_user_label": "Tracker"}
        assert connecting.load(CUSTOM_MCP_ENGINE, CONN) == labels
        assert connecting.read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == labels

    def test_a_source_that_cannot_say_what_it_hides_reads_as_hiding(self, protected):
        _write_file(AGENT_WRITTEN)
        _hold()

        class Wrapper:
            """A vault wrapper with no ``hides``, which could stand in front of a hidden view."""

            def __init__(self, inner):
                self._inner = inner

            def load(self, engine, name):
                return self._inner.load(engine, name)

            def read_record(self, engine, name):
                return self._inner.read_record(engine, name)

        vault = app_held.AppHeldVault(source=Wrapper(turn_vault_for_scope(LOCAL_SCOPE)), root=own_vault_root(LOCAL_SCOPE))
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) == {"_label": "Renamed by agent"}
        assert vault.read_record(CUSTOM_MCP_ENGINE, CONN)["fields"] == {"_label": "Renamed by agent"}

    @pytest.mark.parametrize("method", [None, "MCP", " mcp", "mcp\n"])
    def test_the_file_method_does_not_decide_whether_held_settings_are_read(self, protected, method):
        written = {key: value for key, value in AGENT_WRITTEN.items() if key != "_method"}
        if method is not None:
            written["_method"] = method
        _write_file(written)
        _hold()
        assert _turn_vault().load(CUSTOM_MCP_ENGINE, CONN) == EXPECTED_RECORD_FIELDS | {
            "access_token": "AT-held",
            "expires_at": "2030-01-01T00:00:00Z",
        }

    def test_without_the_app_secret_every_custom_connection_is_hidden(self, stores):
        _write_file(AGENT_WRITTEN)
        own_vault_for_scope(LOCAL_SCOPE).save("hubspot", "acme", {"_method": "mcp", "access_token": "t"})
        vault = _turn_vault()
        listed = {(entry["engine"], entry["name"]) for entry in vault.list_connections()}
        assert (CUSTOM_MCP_ENGINE, CONN) not in listed
        assert ("hubspot", "acme") in listed
        assert vault.load(CUSTOM_MCP_ENGINE, CONN) is None
        assert vault.read_record(CUSTOM_MCP_ENGINE, CONN) is None
        # The connecting view lists hidden connections by label alone, so a
        # new connection's label never repeats one; it never carries more.
        connecting = vault.for_connecting().load(CUSTOM_MCP_ENGINE, CONN) or {}
        assert not {"_server_url", "_allowed_tools", "access_token"} & set(connecting)


class TestHoldCustom:
    def test_a_new_server_url_drops_the_held_token(self, protected):
        _hold()
        entry = app_held.hold_custom(own_vault_root(LOCAL_SCOPE), CONN, {"server_url": "https://other.example.com/mcp"})
        assert "access_token" not in entry and "expires_at" not in entry
        assert entry["server_url"] == "https://other.example.com/mcp"

    def test_a_new_server_url_with_its_own_token_keeps_that_token(self, protected):
        _hold()
        entry = app_held.hold_custom(
            own_vault_root(LOCAL_SCOPE), CONN, {"server_url": "https://other.example.com/mcp", "access_token": "AT-new"}
        )
        assert entry["access_token"] == "AT-new"

    def test_the_same_server_url_keeps_the_token(self, protected):
        _hold()
        entry = app_held.hold_custom(own_vault_root(LOCAL_SCOPE), CONN, {"server_url": HELD["server_url"]})
        assert entry["access_token"] == "AT-held"


class TestGenericWritersRefuse:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", f"{BASE}/save", {"connector_id": CUSTOM_MCP_ENGINE, "method": "mcp", "name": CONN, "values": {}}),
            ("post", f"{BASE}/validate-and-save", {"connector_id": CUSTOM_MCP_ENGINE, "method": "mcp", "name": CONN, "values": {}}),
            ("patch", f"{BASE}/{CUSTOM_MCP_ENGINE}/{CONN}/token", {"access_token": "AT-agent"}),
            ("patch", f"{BASE}/{CUSTOM_MCP_ENGINE}/{CONN}/access-mode", {"access_mode": "write"}),
            ("put", f"{BASE}/{CUSTOM_MCP_ENGINE}/{CONN}/app-held", {"access_token": "AT-agent"}),
        ],
    )
    def test_refused_with_409_and_nothing_held(self, protected, method, path, body):
        _write_file({"_method": "mcp"})
        response = getattr(_client(), method)(path, json=body, headers=HEADERS)
        assert response.status_code == 409
        assert response.json()["detail"] == "custom_mcp_reserved"
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN) == {}

    def test_a_save_through_the_turn_vault_is_refused(self, protected):
        with pytest.raises(ReservedEngineError):
            _turn_vault().save(CUSTOM_MCP_ENGINE, CONN, dict(AGENT_WRITTEN))


class TestReservedEngine:
    def test_the_engine_id_is_reserved(self):
        with pytest.raises(InvalidConnectorIdError):
            validate_engine_id(CUSTOM_MCP_ENGINE)
        assert validate_engine_id("httpbin") == "httpbin"

    @pytest.fixture
    def db(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        SQLModel.metadata.create_all(engine)
        return engine

    def test_no_custom_connector_definition_can_take_the_id(self, db):
        form = {"form_id": "fm_aaaaaaaaaa", "engine": CUSTOM_MCP_ENGINE, "title": "x", "fields": []}
        with Session(db) as session, pytest.raises(UnstorableFormError):
            CustomConnectorService(ScopedSession(session, LOCAL_SCOPE)).upsert_from_form(CUSTOM_MCP_ENGINE, form)

    def test_a_planted_row_with_the_id_is_never_served(self, db):
        form = with_test_request({
            "form_id": "fm_aaaaaaaaaa",
            "engine": "httpbin",
            "title": "Connect httpbin",
            "logo_color": "#3a7",
            "fields": [{"name": "token", "label": "Token", "type": "password", "secret": True}],
        })
        with Session(db) as session:
            CustomConnectorService(ScopedSession(session, LOCAL_SCOPE)).upsert_from_form("httpbin", form)
            row = session.exec(select(CustomConnector)).one()
            row.connector_id = CUSTOM_MCP_ENGINE
            session.add(row)
            session.commit()
        with Session(db) as session:
            catalog = ConnectorCatalog(ScopedSession(session, LOCAL_SCOPE))
            assert CUSTOM_MCP_ENGINE not in catalog.get_connectors()
            assert catalog.get_connector(CUSTOM_MCP_ENGINE) is None


class TestHealthFlag:
    CHALLENGE = "0f" * 32

    def test_present_beside_a_valid_secret_proof(self, protected):
        body = _client().get("/api/v1/health/", headers={app_secret.CHALLENGE_HEADER: self.CHALLENGE}).json()
        assert body["app_secret_proof"] == hmac.new(SECRET.encode(), self.CHALLENGE.encode(), hashlib.sha256).hexdigest()
        assert body["custom_mcp"] == 1

    def test_absent_without_a_challenge(self, protected):
        assert "custom_mcp" not in _client().get("/api/v1/health/").json()

    def test_absent_without_the_app_secret(self, stores):
        body = _client().get("/api/v1/health/", headers={app_secret.CHALLENGE_HEADER: self.CHALLENGE}).json()
        assert "custom_mcp" not in body


def test_the_engine_id_is_anton_s():
    from anton.core.mcp.wiring import CUSTOM_MCP_ENGINE as ANTON_ENGINE  # loads the MCP SDK; only this test needs it

    assert CUSTOM_MCP_ENGINE == ANTON_ENGINE


def test_connection_ids():
    assert is_connection_id(CONN)
    assert not is_connection_id("acme-tracker")
    assert not is_connection_id(CONN + "\n")
    assert not is_connection_id("cm-" + "0" * 31)
