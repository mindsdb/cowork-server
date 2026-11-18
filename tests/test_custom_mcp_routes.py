"""The custom MCP routes: the only writers of a custom connection's held
settings, guarded by the app secret and protected mode.

Requests go through the real API router over loopback; outcomes are read
from the vault file, the held entry and the turn vault anton reads. Only
anton's tool listing, which would reach a real server, is replaced.
"""
from __future__ import annotations

import json

import pytest
from anton.core.mcp import wiring as anton_wiring
from anton.core.mcp.errors import McpEgressBlocked, McpHostUnresolved, McpTokenRejected
from anton.utils.datasources import clear_ds_env, restore_namespaced_env, scrub_credentials
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import app_secret
from cowork.api.v1.endpoints.connectors.custom_mcp import _record_file_present
from cowork.api.v1.permissions import LoopbackDesktopOnly
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.api.v1.router import api_router
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, get_tenant_scope
from cowork.services.connectors import app_held
from cowork.services.connectors.custom_mcp import CUSTOM_MCP_ENGINE
from cowork.services.connectors.persist import own_vault_for_scope, own_vault_root, turn_vault_for_scope

SECRET = "ab" * 32
HEADERS = {app_secret.HEADER: SECRET}
ROUTES = "/api/v1/connectors/connections/custom_mcp"
CONN = "cm-" + "0123456789abcdef" * 2
OTHER = "cm-" + "fedcba9876543210" * 2
CREATE = {
    "id": CONN,
    "label": "Acme Tracker",
    "server_url": "https://mcp.example.com/mcp",
    "resource": "https://mcp.example.com/mcp",
    "token_endpoint": "https://auth.example.com/token",
    "client_id": "client-1",
    "access_token": "AT-1",
    "expires_at": "2030-01-01T00:00:00Z",
}


@pytest.fixture
def stores(sharepoint_stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", None)
    app_held._custom.clear()
    yield sharepoint_stores
    app_held._custom.clear()


@pytest.fixture
def protected(stores, monkeypatch):
    monkeypatch.setattr(app_secret, "_secret", SECRET)


def _client(app: FastAPI | None = None) -> TestClient:
    if app is None:
        app = FastAPI()
        app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _held() -> dict:
    return app_held.held_custom(own_vault_root(LOCAL_SCOPE), CONN)


def _turn_fields() -> dict:
    source = turn_vault_for_scope(LOCAL_SCOPE)
    vault = app_held.turn_vault(source, source_vault=source, root=own_vault_root(LOCAL_SCOPE))
    return vault.load(CUSTOM_MCP_ENGINE, CONN)


def _create(**overrides) -> None:
    response = _client().post(ROUTES, json={**CREATE, **overrides}, headers=HEADERS)
    assert response.status_code == 201, response.text


class TestCheckUrl:
    @pytest.mark.parametrize(
        ("url", "canonical"),
        [
            ("https://mcp.example.com/mcp", "https://mcp.example.com/mcp"),
            ("https://MCP.Example.com", "https://mcp.example.com/"),
            ("https://straße.example/mcp", "https://xn--strae-oqa.example/mcp"),
            ("https://mcp.example.com:8443/mcp/", "https://mcp.example.com:8443/mcp/"),
            ("https://mcp.example.com:443/mcp", "https://mcp.example.com/mcp"),
        ],
    )
    def test_returns_the_canonical_form(self, protected, url, canonical):
        response = _client().post(f"{ROUTES}/check-url", json={"url": url}, headers=HEADERS)
        assert response.status_code == 200
        assert response.json() == {"url": canonical}

    @pytest.mark.parametrize(
        "url",
        ["http://mcp.example.com/mcp", "https://mcp.example.com/mcp?key=s", "https://u:p@mcp.example.com/", " https://mcp.example.com/"],
    )
    def test_refuses_an_unusable_url(self, protected, url):
        response = _client().post(f"{ROUTES}/check-url", json={"url": url}, headers=HEADERS)
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_url"


def test_every_custom_route_declares_loopback_desktop_only():
    """The router's guard is what refuses a non-loopback peer, a hostile Host
    or Origin, and org mode, before the app secret is checked."""
    routes = [route for route in api_router.routes if route_key(route)[0].startswith("/api/v1/connectors/connections/custom_mcp")]
    assert len(routes) == 6
    assert all(declared_permissions(route) == [LoopbackDesktopOnly] for route in routes)


@pytest.mark.parametrize("name", ["../custom_mcp-x", "cm-" + "0" * 31, "cm-" + "0" * 32 + "/x", ""])
def test_a_name_that_is_no_connection_id_has_no_record_file(protected, name):
    assert _record_file_present(LOCAL_SCOPE, name) is False


def test_a_name_that_leads_out_of_the_vault_folder_has_no_record_file(protected):
    root = own_vault_root(LOCAL_SCOPE)
    (root / f"{CUSTOM_MCP_ENGINE}-a").mkdir(parents=True)
    (root.parent / "outside").write_text("x", encoding="utf-8")
    assert _record_file_present(LOCAL_SCOPE, "a/../../outside") is False


class TestGuards:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", f"{ROUTES}/check-url", {"url": "https://mcp.example.com/mcp"}),
            ("post", ROUTES, CREATE),
            ("put", f"{ROUTES}/{CONN}/held", {"access_token": "x"}),
            ("put", f"{ROUTES}/{CONN}/allowed-tools", {"tools": ["a"]}),
            ("get", f"{ROUTES}/{CONN}/held-summary", None),
            ("get", f"{ROUTES}/{CONN}/tools", None),
        ],
    )
    def test_refused_without_the_app_secret(self, protected, method, path, body):
        kwargs = {"json": body} if body is not None else {}
        response = getattr(_client(), method)(path, headers={app_secret.HEADER: "cd" * 32}, **kwargs)
        assert response.status_code == 403
        assert _held() == {}

    def test_refused_in_unprotected_mode(self, stores):
        response = _client().post(ROUTES, json=CREATE)
        assert response.status_code == 409
        assert response.json()["detail"] == "custom_mcp_unavailable"
        assert own_vault_for_scope(LOCAL_SCOPE).read_record(CUSTOM_MCP_ENGINE, CONN) is None

    def test_refused_in_org_mode(self, protected):
        app = FastAPI()
        app.include_router(api_router)
        app.dependency_overrides[get_tenant_scope] = lambda: TenantScope(org_mode=True, org_id="org-a", user_id="u")
        response = _client(app).post(ROUTES, json=CREATE, headers=HEADERS)
        assert response.status_code == 409


class TestCreate:
    def test_creates_a_record_with_nothing_of_use_in_the_file(self, protected):
        _create(server_url="https://MCP.example.com/mcp")
        record = own_vault_for_scope(LOCAL_SCOPE).read_record(CUSTOM_MCP_ENGINE, CONN)
        # Bookkeeping only: nothing that says where a token goes or what runs.
        assert record["fields"] == {"_method": "mcp", "_connector_id": CUSTOM_MCP_ENGINE}
        assert _held()["server_url"] == "https://mcp.example.com/mcp"
        assert _held()["allowed_tools"] == []
        assert _turn_fields()["access_token"] == "AT-1"

    def test_an_existing_id_is_never_reused(self, protected):
        _create()
        response = _client().post(ROUTES, json={**CREATE, "server_url": "https://other.example.com/"}, headers=HEADERS)
        assert response.status_code == 409
        assert _held()["server_url"] == "https://mcp.example.com/mcp"

    @pytest.mark.parametrize(
        "overrides",
        [{"id": "acme-tracker"}, {"id": CONN + "\n"}, {"label": ""}, {"token_endpoint": "http://auth.example.com/token"}, {"extra": 1}],
    )
    def test_refuses_a_bad_body(self, protected, overrides):
        response = _client().post(ROUTES, json={**CREATE, **overrides}, headers=HEADERS)
        assert response.status_code == 422
        assert _held() == {}

    def test_refuses_an_unusable_server_url(self, protected):
        response = _client().post(ROUTES, json={**CREATE, "server_url": "http://mcp.example.com/"}, headers=HEADERS)
        assert response.status_code == 422
        assert own_vault_for_scope(LOCAL_SCOPE).read_record(CUSTOM_MCP_ENGINE, CONN) is None


class TestHeldAndTools:
    def test_held_updates_reach_the_next_turn(self, protected):
        _create()
        response = _client().put(f"{ROUTES}/{CONN}/held", json={"access_token": "AT-2", "expires_at": "2031-01-01T00:00:00Z"}, headers=HEADERS)
        assert response.status_code == 200
        assert _turn_fields()["access_token"] == "AT-2"

    def test_a_created_connection_s_token_is_scrubbed_from_tool_output(self, protected):
        _create()
        source = turn_vault_for_scope(LOCAL_SCOPE)
        vault = app_held.turn_vault(source, source_vault=source, root=own_vault_root(LOCAL_SCOPE))
        try:
            restore_namespaced_env(vault)
            scrubbed = scrub_credentials("the server said: bad token AT-1")
        finally:
            clear_ds_env()
        assert "AT-1" not in scrubbed

    @pytest.mark.parametrize("planted", ["access", {"access_token": False}, [1], ["expires_at"]])
    def test_secure_keys_planted_in_the_file_never_unhide_the_token(self, protected, planted):
        _create()
        vault_dir = own_vault_root(LOCAL_SCOPE)
        [path] = [entry for entry in vault_dir.iterdir() if entry.name == f"{CUSTOM_MCP_ENGINE}-{CONN}"]
        record = json.loads(path.read_text())
        path.write_text(json.dumps({**record, "secure_keys": planted}))
        source = turn_vault_for_scope(LOCAL_SCOPE)
        vault = app_held.turn_vault(source, source_vault=source, root=vault_dir)
        try:
            restore_namespaced_env(vault)
            scrubbed = scrub_credentials("the server said: bad token AT-1")
        finally:
            clear_ds_env()
        assert "AT-1" not in scrubbed

    def test_a_new_server_url_drops_the_token(self, protected):
        _create()
        _client().put(f"{ROUTES}/{CONN}/held", json={"server_url": "https://other.example.com/mcp"}, headers=HEADERS)
        assert "access_token" not in _held()

    def test_a_missing_connection_answers_its_own_404(self, protected):
        for method, path, body in [
            ("put", f"{ROUTES}/{OTHER}/held", {"access_token": "x"}),
            ("put", f"{ROUTES}/{OTHER}/allowed-tools", {"tools": []}),
            ("get", f"{ROUTES}/{OTHER}/held-summary", None),
            ("get", f"{ROUTES}/not-an-id/held-summary", None),
        ]:
            kwargs = {"json": body} if body is not None else {}
            response = getattr(_client(), method)(path, headers=HEADERS, **kwargs)
            assert response.status_code == 404
            assert response.json()["detail"] == {"error": "custom_connection_missing"}

    def test_switching_tools_reaches_the_next_turn(self, protected):
        _create()
        response = _client().put(f"{ROUTES}/{CONN}/allowed-tools", json={"tools": ["search_issues", "get_issue"]}, headers=HEADERS)
        assert response.json() == {"allowed_tools": ["search_issues", "get_issue"]}
        assert json.loads(_turn_fields()["_allowed_tools"]) == ["search_issues", "get_issue"]

    def test_held_summary(self, protected):
        _create()
        assert _client().get(f"{ROUTES}/{CONN}/held-summary", headers=HEADERS).json() == {
            "label": "Acme Tracker",
            "server_url": "https://mcp.example.com/mcp",
        }

    def test_create_starts_from_an_empty_held_entry(self, protected):
        app_held.hold_custom(own_vault_root(LOCAL_SCOPE), CONN, {"access_token": "AT-old", "client_id": "old-client"})
        _create(access_token=None, client_id=None)
        assert "access_token" not in _held()
        assert "client_id" not in _held()

    def test_a_missing_connection_drops_what_is_still_held(self, protected):
        app_held.hold_custom(own_vault_root(LOCAL_SCOPE), OTHER, {"access_token": "AT-leftover"})
        response = _client().get(f"{ROUTES}/{OTHER}/held-summary", headers=HEADERS)
        assert response.status_code == 404
        assert app_held.held_custom(own_vault_root(LOCAL_SCOPE), OTHER) == {}

    @pytest.mark.parametrize("damage", ["not json", "unreadable"])
    def test_a_record_that_cannot_be_read_is_never_called_missing(self, protected, damage):
        _create()
        path = own_vault_root(LOCAL_SCOPE) / f"{CUSTOM_MCP_ENGINE}-{CONN}"
        # The file the vault wrote is where the route looks for it.
        assert path.is_file()
        if damage == "not json":
            path.write_text("{not json", encoding="utf-8")
        else:
            path.chmod(0o000)
        try:
            response = _client().get(f"{ROUTES}/{CONN}/held-summary", headers=HEADERS)
        finally:
            path.chmod(0o600)
        assert response.status_code == 503
        assert response.json()["detail"] == {"error": "custom_connection_unreadable"}
        assert _held()["access_token"] == "AT-1"

    def test_the_list_and_detail_show_the_held_label_not_the_file(self, protected):
        _create()
        # The label tool writes the file through the turn vault, as agent code can.
        own_vault_for_scope(LOCAL_SCOPE).save(
            CUSTOM_MCP_ENGINE,
            CONN,
            {"_method": "mcp", "_user_label": "GitHub (official)", "_label": "GitHub", "_server_url": "https://attacker.example/"},
        )
        listing = _client().get("/api/v1/connectors/connections/").json()
        [card] = [entry for entry in listing if entry["engine"] == CUSTOM_MCP_ENGINE]
        assert card["user_label"] == "Acme Tracker"
        detail = _client().get(f"/api/v1/connectors/connections/{CUSTOM_MCP_ENGINE}/{CONN}").json()
        assert detail["user_label"] == "Acme Tracker"
        assert detail["fields"] == {}

    def test_lists_tools_with_the_held_url_and_token(self, protected, monkeypatch):
        _create(allowed_tools=["create_issue"])
        calls = []

        async def fake_list(url, token, *, connection_name):
            calls.append((url, token, connection_name))
            return [
                anton_wiring.ServerTool(name="search_issues", description="Search", read_only=True),
                anton_wiring.ServerTool(name="create_issue", description="Create", read_only=False),
            ]

        monkeypatch.setattr(anton_wiring, "list_custom_server_tools", fake_list)
        response = _client().get(f"{ROUTES}/{CONN}/tools", headers=HEADERS)
        assert response.status_code == 200
        assert response.json() == {
            "tools": [
                {"name": "search_issues", "description": "Search", "read_only": True, "allowed": False},
                {"name": "create_issue", "description": "Create", "read_only": False, "allowed": True},
            ]
        }
        assert calls == [("https://mcp.example.com/mcp", "AT-1", CONN)]

    def test_a_switch_made_while_the_tools_are_listed_shows_as_made(self, protected, monkeypatch):
        _create(allowed_tools=["create_issue"])

        async def slow_list(url, token, *, connection_name):
            response = _client().put(f"{ROUTES}/{CONN}/allowed-tools", json={"tools": ["search_issues"]}, headers=HEADERS)
            assert response.status_code == 200
            return [
                anton_wiring.ServerTool(name="search_issues", description="Search", read_only=True),
                anton_wiring.ServerTool(name="create_issue", description="Create", read_only=False),
            ]

        monkeypatch.setattr(anton_wiring, "list_custom_server_tools", slow_list)
        tools = _client().get(f"{ROUTES}/{CONN}/tools", headers=HEADERS).json()["tools"]
        assert {tool["name"]: tool["allowed"] for tool in tools} == {"search_issues": True, "create_issue": False}

    @pytest.mark.parametrize(
        ("error", "status_code", "code"),
        [
            (McpHostUnresolved("does not resolve"), 502, "unresolved"),
            (McpEgressBlocked("not public"), 422, "refused_address"),
            (McpTokenRejected(401), 409, "token_rejected"),
            (ConnectionError("https://mcp.example.com/SECRET-PATH"), 502, "unreachable"),
        ],
    )
    def test_tool_listing_failures_have_their_own_status(self, protected, monkeypatch, caplog, error, status_code, code):
        _create()

        async def failing(url, token, *, connection_name):
            raise error

        monkeypatch.setattr(anton_wiring, "list_custom_server_tools", failing)
        caplog.set_level("DEBUG")
        response = _client().get(f"{ROUTES}/{CONN}/tools", headers=HEADERS)
        assert response.status_code == status_code
        assert response.json()["detail"] == {"error": code}
        assert "SECRET-PATH" not in caplog.text
