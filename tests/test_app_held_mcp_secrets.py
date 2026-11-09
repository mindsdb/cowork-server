"""MCP tokens and access modes held in server memory while the desktop app's
secret is held (cowork.app_secret, cowork.services.connectors.app_held).

Requests go through the real API router over loopback; outcomes are read
back from the vault file on disk, the turn vault anton reads and the
connection detail.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.router import api_router
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors import app_held
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.persist import (
    own_vault_for_scope,
    own_vault_root,
    persist_connection,
    turn_vault_for_scope,
)

SECRET = "ab" * 32
WRONG = "cd" * 32
BASE = "/api/v1/connectors/connections"
CONN = "acme-hubspot"
CONN_URL = f"{BASE}/hubspot/{CONN}"
HEADERS = {app_secret.HEADER: SECRET}


@pytest.fixture
def stores(sharepoint_stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", None)
    app_held._entries.clear()
    yield sharepoint_stores
    app_held._entries.clear()


@pytest.fixture
def protected(stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", SECRET)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _seed_mcp(mode: str = "read", token: str = "file-token") -> None:
    """A connection saved before protected mode existed: token and mode in the file."""
    persist_connection(
        "hubspot", "mcp", "",
        {"access_token": token, "account_email": "acme.hubspot", "expires_at": "2026-01-01T00:00:00Z"},
        default_fields={"_access_mode": mode},
        scope=LOCAL_SCOPE,
    )


def _file_fields(engine: str = "hubspot", name: str = CONN) -> dict:
    return own_vault_for_scope(LOCAL_SCOPE).read_record(engine, name)["fields"]


def _raw_file() -> str:
    return "".join(p.read_text() for p in own_vault_root(LOCAL_SCOPE).iterdir() if p.is_file())


class TestLegacyMode:
    def test_health_has_no_protocol_key(self, stores):
        assert "app_secret_protocol" not in _client().get("/api/v1/health/").json()

    def test_access_mode_and_token_write_the_file_without_a_secret(self, stores):
        _seed_mcp()
        client = _client()

        assert client.patch(f"{CONN_URL}/access-mode", json={"access_mode": "write"}).status_code == 200
        assert client.patch(f"{CONN_URL}/token", json={"access_token": "new"}).status_code == 200

        fields = _file_fields()
        assert fields["_access_mode"] == "write"
        assert fields["access_token"] == "new"

    def test_put_app_held_is_refused_with_409(self, stores):
        _seed_mcp()
        response = _client().put(f"{CONN_URL}/app-held", json={"access_mode": "read"}, headers=HEADERS)
        assert response.status_code == 409
        assert response.json() == {"detail": "app_secret_protocol_off"}
        assert _file_fields()["access_token"] == "file-token"

    def test_save_keeps_writing_the_token_to_the_file(self, stores):
        response = _client().post(
            f"{BASE}/save",
            json={"connector_id": "hubspot", "method": "mcp", "name": "",
                  "values": {"access_token": "tok-1", "account_email": "acme.hubspot"}},
        )
        assert response.status_code == 200, response.text
        assert _file_fields()["access_token"] == "tok-1"
        assert _file_fields()["_access_mode"] == "read"


class TestSecretCheck:
    @pytest.mark.parametrize("headers", [{}, {app_secret.HEADER: WRONG}, {app_secret.HEADER: ""}])
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("patch", "access-mode", {"access_mode": "write"}),
            ("patch", "token", {"access_token": "stolen"}),
            ("put", "app-held", {"access_mode": "write"}),
            ("delete", "app-held", None),
        ],
    )
    def test_refused_without_the_right_secret(self, protected, headers, method, path, body):
        _seed_mcp()
        app_held.hold(own_vault_root(LOCAL_SCOPE), "hubspot", CONN, {"access_mode": "read"})
        kwargs = {"headers": headers} | ({"json": body} if body is not None else {})

        response = getattr(_client(), method)(f"{CONN_URL}/{path}", **kwargs)

        assert response.status_code == 403
        assert response.json() == {"detail": "app_secret_required"}
        assert _file_fields()["access_token"] == "file-token"
        assert _file_fields()["_access_mode"] == "read"
        assert app_held.held(own_vault_root(LOCAL_SCOPE), "hubspot", CONN) == {"access_mode": "read"}

    def test_accepted_with_the_secret(self, protected):
        _seed_mcp()
        response = _client().patch(f"{CONN_URL}/access-mode", json={"access_mode": "write"}, headers=HEADERS)
        assert response.status_code == 200
        assert response.json() == {"ok": True, "access_mode": "write"}

    def test_unrelated_routes_need_no_secret(self, protected):
        _seed_mcp()
        client = _client()
        assert client.get(f"{BASE}/").status_code == 200
        assert client.get(CONN_URL).status_code == 200
        assert client.patch(f"{CONN_URL}/picked-files", json={"files": []}).status_code == 200
        assert client.get("/api/v1/health/").status_code == 200

    def test_health_advertises_the_protocol(self, protected):
        assert _client().get("/api/v1/health/").json()["app_secret_protocol"] == 1


class TestAppHeldMigration:
    def test_put_strips_the_file_only_once_both_are_held(self, protected):
        _seed_mcp(mode="read")
        client = _client()

        first = client.put(f"{CONN_URL}/app-held", json={"access_token": "mem-token"}, headers=HEADERS)
        assert first.json() == {"ok": True, "stripped": False}
        assert _file_fields()["access_token"] == "file-token"
        assert _file_fields()["_access_mode"] == "read"

        second = client.put(f"{CONN_URL}/app-held", json={"access_mode": "read"}, headers=HEADERS)
        assert second.json() == {"ok": True, "stripped": True}
        fields = _file_fields()
        assert "access_token" not in fields and "_access_mode" not in fields
        assert "file-token" not in _raw_file() and "mem-token" not in _raw_file()
        assert fields["account_email"] == "acme.hubspot"

        again = client.put(f"{CONN_URL}/app-held", json={"access_mode": "read"}, headers=HEADERS)
        assert again.json() == {"ok": True, "stripped": True}

    def test_a_migrated_write_mode_survives(self, protected):
        _seed_mcp(mode="write")
        client = _client()
        # The app read the legacy mode from the detail, before any push.
        assert client.get(CONN_URL).json()["fields"]["_access_mode"] == "write"

        response = client.put(
            f"{CONN_URL}/app-held",
            json={"access_token": "mem-token", "expires_at": "2027-01-01T00:00:00Z", "access_mode": "write"},
            headers=HEADERS,
        )

        assert response.json() == {"ok": True, "stripped": True}
        assert "_access_mode" not in _file_fields()
        assert client.get(CONN_URL).json()["fields"]["_access_mode"] == "write"
        turn = app_held.AppHeldVault(source=turn_vault_for_scope(LOCAL_SCOPE), root=own_vault_root(LOCAL_SCOPE))
        from anton.core.mcp.wiring import _access_mode_of

        assert _access_mode_of(turn, "hubspot", CONN) == "write"

    @pytest.mark.parametrize(
        ("name", "body", "code"),
        [
            ("missing", {"access_mode": "read"}, 404),
            (CONN, {"access_mode": "admin"}, 422),
        ],
    )
    def test_put_rejects_a_missing_connection_or_a_bad_mode(self, protected, name, body, code):
        _seed_mcp()
        response = _client().put(f"{BASE}/hubspot/{name}/app-held", json=body, headers=HEADERS)
        assert response.status_code == code

    def test_put_rejects_a_non_mcp_connection(self, protected):
        persist_connection("postgres", None, "db", {"host": "h", "password": "pw"}, scope=LOCAL_SCOPE)
        response = _client().put(f"{BASE}/postgres/db/app-held", json={"access_mode": "read"}, headers=HEADERS)
        assert response.status_code == 422
        assert _file_fields("postgres", "db")["password"] == "pw"

    def test_delete_app_held_and_delete_connection_clear_memory(self, protected):
        _seed_mcp()
        client = _client()
        root = own_vault_root(LOCAL_SCOPE)
        client.put(f"{CONN_URL}/app-held", json={"access_token": "t", "access_mode": "write"}, headers=HEADERS)

        assert client.delete(f"{CONN_URL}/app-held", headers=HEADERS).json() == {"ok": True}
        assert app_held.held(root, "hubspot", CONN) == {}

        client.put(f"{CONN_URL}/app-held", json={"access_token": "t", "access_mode": "write"}, headers=HEADERS)
        assert client.delete(CONN_URL).status_code == 204
        assert app_held.held(root, "hubspot", CONN) == {}


class TestProtectedWrites:
    def test_save_leaves_no_token_or_mode_in_the_file(self, protected):
        response = _client().post(
            f"{BASE}/save",
            json={"connector_id": "hubspot", "method": "mcp", "name": "",
                  "values": {"access_token": "tok-1", "account_email": "acme.hubspot",
                             "expires_at": "2027-01-01T00:00:00Z"}},
        )

        assert response.status_code == 200, response.text
        fields = _file_fields()
        assert "access_token" not in fields and "_access_mode" not in fields
        assert "tok-1" not in _raw_file()
        assert app_held.held(own_vault_root(LOCAL_SCOPE), "hubspot", CONN) == {
            "access_token": "tok-1", "expires_at": "2027-01-01T00:00:00Z", "access_mode": "read",
        }

    def test_reconnect_save_keeps_a_held_write_mode_and_drops_the_legacy_token(self, protected):
        _seed_mcp(mode="read", token="old-token")
        root = own_vault_root(LOCAL_SCOPE)
        app_held.hold(root, "hubspot", CONN, {"access_mode": "write"})

        response = _client().post(
            f"{BASE}/save",
            json={"connector_id": "hubspot", "method": "mcp", "name": "",
                  "values": {"access_token": "new-token", "account_email": "acme.hubspot"}},
        )

        assert response.status_code == 200, response.text
        assert "old-token" not in _raw_file() and "new-token" not in _raw_file()
        held = app_held.held(root, "hubspot", CONN)
        assert held["access_token"] == "new-token"
        assert held["access_mode"] == "write"

    def test_token_refresh_goes_to_memory_and_the_expiry_to_the_file(self, protected):
        _seed_mcp()
        client = _client()
        client.put(f"{CONN_URL}/app-held", json={"access_token": "t0", "access_mode": "read"}, headers=HEADERS)

        response = client.patch(
            f"{CONN_URL}/token",
            json={"access_token": "t1", "expires_at": "2027-06-01T00:00:00Z"},
            headers=HEADERS,
        )

        assert response.status_code == 200
        fields = _file_fields()
        assert "access_token" not in fields
        assert fields["expires_at"] == "2027-06-01T00:00:00Z"
        assert ConnectionsService(LOCAL_SCOPE).mcp_access_token("hubspot", CONN) == "t1"

    def test_access_mode_change_goes_to_memory_not_the_file(self, protected):
        _seed_mcp(mode="read")
        client = _client()

        client.patch(f"{CONN_URL}/access-mode", json={"access_mode": "write"}, headers=HEADERS)

        assert _file_fields()["_access_mode"] == "read"
        assert client.get(CONN_URL).json()["fields"]["_access_mode"] == "write"

    def test_a_non_mcp_token_refresh_still_writes_the_file(self, protected):
        persist_connection("linear", "browser_oauth_builtin", "lin", {"access_token": "a"}, scope=LOCAL_SCOPE)
        response = _client().patch(f"{BASE}/linear/lin/token", json={"access_token": "b"}, headers=HEADERS)
        assert response.status_code == 200
        assert _file_fields("linear", "lin")["access_token"] == "b"


class TestReadersSeeMemory:
    def test_turn_vault_service_and_detail_read_the_held_values(self, protected):
        _seed_mcp(mode="read")
        _client().put(
            f"{CONN_URL}/app-held",
            json={"access_token": "mem-token", "expires_at": "2027-01-01T00:00:00Z", "access_mode": "write"},
            headers=HEADERS,
        )
        root = own_vault_root(LOCAL_SCOPE)
        turn = app_held.AppHeldVault(source=turn_vault_for_scope(LOCAL_SCOPE), root=root)
        from anton.core.mcp.wiring import _access_mode_of

        assert turn.load("hubspot", CONN)["access_token"] == "mem-token"
        assert _access_mode_of(turn, "hubspot", CONN) == "write"
        # Never into a Scratchpad's environment, nor named in the agent's listing.
        assert "mem-token" not in (turn.env_for("hubspot", CONN) or {}).values()
        assert "access_token" not in turn.read_record("hubspot", CONN)["fields"]
        assert ConnectionsService(LOCAL_SCOPE).mcp_access_token("hubspot", CONN) == "mem-token"
        assert ConnectionsService(LOCAL_SCOPE).get("hubspot", CONN).fields["_access_mode"] == "write"

    def test_before_the_first_push_readers_fall_back_to_the_file(self, protected):
        _seed_mcp(mode="write")
        turn = app_held.AppHeldVault(source=turn_vault_for_scope(LOCAL_SCOPE), root=own_vault_root(LOCAL_SCOPE))
        from anton.core.mcp.wiring import _access_mode_of

        assert turn.load("hubspot", CONN)["access_token"] == "file-token"
        assert _access_mode_of(turn, "hubspot", CONN) == "write"
        assert ConnectionsService(LOCAL_SCOPE).mcp_access_token("hubspot", CONN) == "file-token"

    def test_legacy_mode_ignores_anything_held(self, stores):
        _seed_mcp(mode="read")
        app_held.hold(own_vault_root(LOCAL_SCOPE), "hubspot", CONN, {"access_token": "mem", "access_mode": "write"})
        assert ConnectionsService(LOCAL_SCOPE).mcp_access_token("hubspot", CONN) == "file-token"
        assert ConnectionsService(LOCAL_SCOPE).get("hubspot", CONN).fields["_access_mode"] == "read"

    def test_neither_secret_nor_token_reaches_os_environ(self, protected):
        _seed_mcp()
        _client().put(f"{CONN_URL}/app-held", json={"access_token": "mem-token", "access_mode": "read"}, headers=HEADERS)
        values = set(os.environ.values())
        assert SECRET not in values
        assert "mem-token" not in values


class TestStartup:
    def test_read_from_stdin_holds_the_secret(self, monkeypatch):
        monkeypatch.setattr(app_secret, "_secret", None)
        assert app_secret.read_from_stdin(io.StringIO(json.dumps({"app_secret": SECRET}) + "\n"))
        assert app_secret.get_app_secret() == SECRET
        assert SECRET not in os.environ.values()

    @pytest.mark.parametrize("line", ["", "not json\n", '{"app_secret": ""}\n', '{"other": 1}\n', "[1]\n"])
    def test_a_bad_line_means_legacy_mode(self, monkeypatch, line):
        monkeypatch.setattr(app_secret, "_secret", None)
        assert not app_secret.read_from_stdin(io.StringIO(line))
        assert app_secret.get_app_secret() is None

    def test_a_parent_that_never_writes_costs_only_the_timeout(self, monkeypatch):
        monkeypatch.setattr(app_secret, "_secret", None)
        read_end, write_end = os.pipe()
        never_written = os.fdopen(read_end)
        try:
            assert not app_secret.read_from_stdin(never_written, timeout=0.2)
        finally:
            # EOF ends the reader thread still blocked on the pipe.
            os.close(write_end)
            never_written.close()
        assert app_secret.get_app_secret() is None

    @pytest.mark.parametrize(("flag", "expected"), [("1", SECRET), (None, None)])
    def test_cli_reads_stdin_only_when_the_flag_is_set(self, tmp_path, flag, expected):
        """Runs cowork.cli.main in a child process with uvicorn stubbed out, so
        the real startup order (flag, .env, settings, stdin) is what runs."""
        script = (
            "import sys, uvicorn, os\n"
            "uvicorn.run = lambda *a, **k: None\n"
            "from cowork import cli, app_secret\n"
            "cli.main()\n"
            "print(repr(app_secret.get_app_secret()))\n"
            "print(app_secret.get_app_secret() in os.environ.values())\n"
        )
        env = {k: v for k, v in os.environ.items() if k != app_secret.STDIN_FLAG}
        env["COWORK_HOME"] = str(tmp_path)
        if flag is not None:
            env[app_secret.STDIN_FLAG] = flag
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps({"app_secret": SECRET}) + "\n",
            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60, check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[-2] == repr(expected)
        assert lines[-1] == "False"
