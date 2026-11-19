"""The web routes for a member's custom MCP servers: relays to auth on the member's own
credential, and the first listing through the inference gateway in-cluster.

Requests go through the real router; auth and the gateway are answered at the
transport, so what this server sends, with which credential and timeout, is
what the tests read.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import custom_mcp_cloud
from cowork.api.v1.permissions import AuthenticatedInOrgMode
from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, get_tenant_scope
from cowork.principal import Principal, get_principal
from cowork.services.connectors.oauth import auth_proxy

MEMBER = TenantScope(org_mode=True, org_id="org-a", user_id="member-a")
BEARER = "Bearer sentinel-member-credential"
AUTH = "auth.example.com"
GATEWAY = "mindshub-inference"
NAME = "cm-" + "0123456789abcdef" * 2
ROUTES = "/connectors/custom-mcp"
COMPLETION = "A" * 43

#: Shaped like auth's CustomMcpToolsResponseSerializer, as the gateway and auth answer it.
SNAPSHOT = {
    "tools_revision": 3,
    "tools": [
        {"name": "search", "description": "", "read_only": True, "allowed": True, "usable": True, "reason": None},
        {"name": "write", "description": "", "read_only": False, "allowed": False, "usable": True, "reason": None},
    ],
}


def _row(revision: int, **overrides) -> dict:
    """One item of auth's member connection list."""
    row = {"id": 7, "name": NAME, "label": "Issues", "server_url": "https://mcp.example.com/mcp",
           "status": "active", "tools_revision": revision, "tools_on": 1}
    row.update(overrides)
    return row


@pytest.fixture()
def hosts(monkeypatch):
    """Org mode, the switch on; auth and the gateway answered from ``answers`` by (method, path).

    Each answer is ``(status, body)`` or an exception to raise. Every request
    is recorded with the timeout of the client that sent it.
    """
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_TURN_CUSTOM_MCP_ENABLED", "true")
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", f"https://{AUTH}")
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_GATEWAY_BASE_URL", f"http://{GATEWAY}")
    get_app_settings.cache_clear()
    state: dict = {"answers": {}, "sent": []}
    timeouts: list = []

    def handle(request: httpx.Request) -> httpx.Response:
        state["sent"].append((request, timeouts[-1]))
        answer = state["answers"].get((request.method, request.url.path), (404, {"detail": "Not found."}))
        if isinstance(answer, BaseException):
            raise answer
        status_code, body = answer
        return httpx.Response(status_code) if body is None else httpx.Response(status_code, json=body)

    real_client = httpx.AsyncClient

    def client(**kwargs):
        timeouts.append(kwargs.get("timeout"))
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(auth_proxy.httpx, "AsyncClient", client)
    yield state
    get_app_settings.cache_clear()


def _client(scope: TenantScope = MEMBER) -> TestClient:
    app = FastAPI()
    app.include_router(custom_mcp_cloud.router, prefix=ROUTES)
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=scope.user_id, org_id=scope.org_id)
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    return TestClient(app, headers={"Authorization": BEARER})


def _sent(state, host: str) -> list[tuple[str, str, float]]:
    """(method, path, timeout) of each request that reached ``host``."""
    return [(r.method, r.url.path, t) for r, t in state["sent"] if r.url.host == host]


_EVERY_ROUTE = [
    ("GET", f"{ROUTES}/status?state=s1", None),
    ("POST", f"{ROUTES}/connect", {"server_url": "https://mcp.example.com/mcp", "label": "Issues"}),
    ("GET", f"{ROUTES}/connections/{NAME}/tools", None),
    ("PUT", f"{ROUTES}/connections/{NAME}/tools", {"tools": ["search"], "tools_revision": 3}),
    ("POST", f"{ROUTES}/connections/{NAME}/reconnect", {}),
    ("POST", f"{ROUTES}/finish", {"completion": COMPLETION}),
]


def test_every_route_is_the_signed_in_members_own():
    routes = [r for r in api_router.routes if route_key(r)[0].startswith("/api/v1/connectors/custom-mcp")]
    assert len(routes) == 8
    assert all(declared_permissions(route) == [AuthenticatedInOrgMode] for route in routes)


@pytest.mark.parametrize(("method", "path", "body"), [*_EVERY_ROUTE, ("GET", f"{ROUTES}/config", None),
                                                      ("DELETE", f"{ROUTES}/connections/{NAME}", None)])
def test_outside_org_mode_every_route_is_refused(hosts, method, path, body):
    response = _client(LOCAL_SCOPE).request(method, path, json=body)

    assert response.status_code == 409
    assert hosts["sent"] == []


@pytest.mark.parametrize(("method", "path", "body"), _EVERY_ROUTE)
def test_with_the_switch_off_every_route_but_delete_is_not_found(hosts, monkeypatch, method, path, body):
    monkeypatch.setenv("COWORK_TURN_CUSTOM_MCP_ENABLED", "false")

    response = _client().request(method, path, json=body)

    assert response.status_code == 404
    assert hosts["sent"] == []


def test_with_the_switch_off_config_says_unavailable_and_delete_still_relays(hosts, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_CUSTOM_MCP_ENABLED", "false")
    hosts["answers"][("DELETE", f"/v1/custom-mcp/connections/{NAME}")] = (204, None)

    config = _client().get(f"{ROUTES}/config")
    deleted = _client().delete(f"{ROUTES}/connections/{NAME}")

    assert config.json() == {"available": False, "redirect_uri": None}
    assert deleted.status_code == 204
    assert _sent(hosts, AUTH) == [("DELETE", f"/v1/custom-mcp/connections/{NAME}", 10.0)]
    assert hosts["sent"][0][0].headers["authorization"] == BEARER


def test_config_relays_auths_redirect_uri_and_reads_auths_404_as_unavailable(hosts):
    hosts["answers"][("GET", "/v1/custom-mcp/config")] = (200, {"redirect_uri": "https://auth/cb"})
    assert _client().get(f"{ROUTES}/config").json() == {"available": True, "redirect_uri": "https://auth/cb"}

    hosts["answers"][("GET", "/v1/custom-mcp/config")] = (404, {"detail": "Not found."})
    assert _client().get(f"{ROUTES}/config").json() == {"available": False, "redirect_uri": None}


def test_a_name_that_is_no_connection_id_never_reaches_auth(hosts):
    response = _client().put(f"{ROUTES}/connections/cm-NOT-AN-ID/tools", json={"tools": [], "tools_revision": 1})

    assert response.status_code == 404
    assert hosts["sent"] == []


def test_each_relay_and_the_listing_carry_the_members_credential_and_no_service_key(hosts):
    hosts["answers"][("POST", "/v1/custom-mcp/start")] = (200, {"status": "connected", "name": NAME})
    hosts["answers"][("GET", "/v1/custom-mcp/connections")] = (200, {"items": [_row(0, tools_on=0)]})
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = (200, SNAPSHOT)

    response = _client().post(f"{ROUTES}/connect", json={"server_url": "https://mcp.example.com/mcp", "label": "x"})

    assert response.json() == {"status": "connected", "name": NAME, "id": 7, "label": "Issues",
                               "server_url": "https://mcp.example.com/mcp", "tools_on": 1,
                               "tools_unavailable": False}
    assert _sent(hosts, AUTH) == [("POST", "/v1/custom-mcp/start", 10.0), ("GET", "/v1/custom-mcp/connections", 10.0)]
    assert _sent(hosts, GATEWAY) == [("POST", "/v1/datasources/custom-mcp/member-tools", 32.0)]
    gateway = next(r for r, _ in hosts["sent"] if r.url.host == GATEWAY)
    assert json.loads(gateway.content) == {"protocol_version": 1, "name": NAME}
    for request, _ in hosts["sent"]:
        assert request.headers["authorization"] == BEARER
        assert not [h for h in request.headers if h.lower().startswith("x-datasource-service")]


def test_a_start_that_needs_a_login_lists_nothing(hosts):
    pending = {"status": "pending", "auth_url": "https://login.example.com/a", "state": "s1", "name": NAME}
    hosts["answers"][("POST", "/v1/custom-mcp/start")] = (200, pending)

    response = _client().post(f"{ROUTES}/connect", json={"server_url": "https://mcp.example.com/mcp", "label": "x"})

    assert response.json() == pending
    assert _sent(hosts, GATEWAY) == []


@pytest.mark.parametrize(("status_code", "code"), [(400, "port"), (400, "own_domain"), (429, "too_many")])
def test_auths_start_refusal_code_passes_through(hosts, status_code, code):
    hosts["answers"][("POST", "/v1/custom-mcp/start")] = (status_code, {"error": code, "detail": "Refused."})

    response = _client().post(f"{ROUTES}/connect", json={"server_url": "https://mcp.example.com:8443/", "label": "x"})

    assert response.status_code == status_code
    assert response.json() == {"error": code, "detail": "Refused."}


@pytest.mark.parametrize(("status_code", "code"), [(403, "login_not_yours"), (409, "wrong_organization"),
                                                   (410, "expired")])
def test_finish_relays_on_the_members_credential_and_passes_auths_code(hosts, status_code, code):
    hosts["answers"][("POST", "/v1/custom-mcp/finish")] = (status_code, {"error": code, "detail": "No."})

    response = _client().post(f"{ROUTES}/finish", json={"completion": COMPLETION})

    assert response.status_code == status_code
    assert response.json() == {"error": code, "detail": "No."}
    request, timeout = hosts["sent"][0]
    assert (request.url.path, json.loads(request.content), timeout) == (
        "/v1/custom-mcp/finish", {"completion": COMPLETION}, 10.0)
    assert request.headers["authorization"] == BEARER


def test_finish_answers_auths_connected(hosts):
    hosts["answers"][("POST", "/v1/custom-mcp/finish")] = (200, {"status": "connected", "name": NAME})

    assert _client().post(f"{ROUTES}/finish", json={"completion": COMPLETION}).json() == {
        "status": "connected", "name": NAME}


@pytest.mark.parametrize("outcome", [{"status": "refused", "name": NAME}, {"status": "pending"},
                                     {"status": "error", "name": ""}])
def test_status_passes_any_outcome_but_success_through_and_lists_nothing(hosts, outcome):
    hosts["answers"][("GET", "/v1/oauth/status")] = (200, outcome)

    response = _client().get(f"{ROUTES}/status", params={"state": "s1"})

    assert response.json() == outcome
    assert _sent(hosts, AUTH) == [("GET", "/v1/oauth/status", 10.0)]
    assert _sent(hosts, GATEWAY) == []


def test_status_runs_the_first_listing_while_the_server_has_no_snapshot(hosts):
    hosts["answers"][("GET", "/v1/oauth/status")] = (200, {"status": "success", "name": NAME})
    hosts["answers"][("GET", "/v1/custom-mcp/connections")] = (200, {"items": [_row(0, tools_on=0)]})
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = (200, SNAPSHOT)

    response = _client().get(f"{ROUTES}/status", params={"state": "s1"})

    assert response.json() == {"status": "success", "name": NAME, "id": 7, "label": "Issues",
                               "server_url": "https://mcp.example.com/mcp", "tools_on": 1,
                               "tools_unavailable": False}
    assert [(m, p, t) for m, p, t in _sent(hosts, AUTH) + _sent(hosts, GATEWAY)] == [
        ("GET", "/v1/oauth/status", 10.0), ("GET", "/v1/custom-mcp/connections", 10.0),
        ("POST", "/v1/datasources/custom-mcp/member-tools", 32.0)]


def test_a_second_poll_or_a_reconnect_does_not_list_again(hosts):
    hosts["answers"][("GET", "/v1/oauth/status")] = (200, {"status": "success", "name": NAME})
    hosts["answers"][("GET", "/v1/custom-mcp/connections")] = (200, {"items": [_row(3, tools_on=2)]})

    response = _client().get(f"{ROUTES}/status", params={"state": "s1"})

    assert response.json()["tools_on"] == 2
    assert response.json()["tools_unavailable"] is False
    assert _sent(hosts, GATEWAY) == []


def test_a_failed_first_listing_leaves_the_server_connected_with_no_tools(hosts):
    hosts["answers"][("GET", "/v1/oauth/status")] = (200, {"status": "success", "name": NAME})
    hosts["answers"][("GET", "/v1/custom-mcp/connections")] = (200, {"items": [_row(0, tools_on=0)]})
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = httpx.ReadTimeout("slow")

    response = _client().get(f"{ROUTES}/status", params={"state": "s1"})

    assert response.status_code == 200
    assert (response.json()["status"], response.json()["tools_on"], response.json()["tools_unavailable"]) == (
        "success", 0, True)


@pytest.mark.parametrize(("method", "path", "body", "auth_path", "status_word"), [
    ("GET", f"{ROUTES}/status?state=s1", None, "/v1/oauth/status", "success"),
    ("POST", f"{ROUTES}/connect", {"server_url": "https://mcp.example.com/mcp", "label": "x"},
     "/v1/custom-mcp/start", "connected"),
], ids=["sign-in finished", "no-login server connected"])
def test_a_connected_server_whose_list_cannot_be_read_still_answers_connected(
        hosts, method, path, body, auth_path, status_word):
    hosts["answers"][(method, auth_path)] = (200, {"status": status_word, "name": NAME})
    hosts["answers"][("GET", "/v1/custom-mcp/connections")] = httpx.ConnectError("down")

    response = _client().request(method, path, json=body)

    assert response.status_code == 200
    assert response.json() == {"status": status_word, "name": NAME, "tools_on": 0, "tools_unavailable": True}
    assert _sent(hosts, GATEWAY) == []


@pytest.mark.parametrize("body", [{"label": ["This field may not be blank."]}, ["not", "a", "dict"],
                                  {"error": "", "detail": "blank code"}],
                         ids=["field errors", "a list", "an empty code"])
def test_a_codeless_400_from_auth_answers_a_fixed_refusal(hosts, body):
    hosts["answers"][("POST", "/v1/custom-mcp/start")] = (400, body)

    response = _client().post(f"{ROUTES}/connect", json={"server_url": "https://mcp.example.com/mcp", "label": "x"})

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_request", "detail": "The request was refused."}


@pytest.mark.parametrize(("body", "relayed"), [
    ({"tools": [f"t{i}" for i in range(12_684)], "tools_revision": 3}, True),
    ({"tools": [f"t{i}" for i in range(12_685)], "tools_revision": 3}, False),
    ({"tools": ["search", "search"], "tools_revision": 3}, False),
    ({"tools": ["search"], "tools_revision": True}, False),
    ({"tools": ["search"], "tools_revision": 1.0}, False),
], ids=["as many as a member can switch on", "one more", "a name twice", "a bool revision", "a float revision"])
def test_a_switch_save_auth_would_refuse_never_leaves(hosts, body, relayed):
    hosts["answers"][("PUT", f"/v1/custom-mcp/connections/{NAME}/tools")] = (200, SNAPSHOT)

    response = _client().put(f"{ROUTES}/connections/{NAME}/tools", json=body)

    assert response.status_code == (200 if relayed else 422)
    assert len(hosts["sent"]) == (1 if relayed else 0)


def test_the_tools_screen_lists_live_and_says_it_is_fresh(hosts):
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = (200, SNAPSHOT)

    response = _client().get(f"{ROUTES}/connections/{NAME}/tools")

    assert response.json() == {**SNAPSHOT, "stale": False}
    assert _sent(hosts, AUTH) == []


def test_a_failed_listing_shows_the_stored_snapshot_marked_stale(hosts):
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = (
        502, {"code": "upstream_unavailable", "detail": "The server did not answer.", "request_id": "r"})
    hosts["answers"][("GET", f"/v1/custom-mcp/connections/{NAME}/tools")] = (200, SNAPSHOT)

    response = _client().get(f"{ROUTES}/connections/{NAME}/tools")

    assert response.status_code == 200
    assert response.json() == {**SNAPSHOT, "stale": True, "code": "upstream_unavailable"}
    assert _sent(hosts, AUTH) == [("GET", f"/v1/custom-mcp/connections/{NAME}/tools", 10.0)]


@pytest.mark.parametrize("code", ["listing_too_large", "needs_reconnect"])
def test_with_no_snapshot_the_gateways_code_passes_through(hosts, code):
    hosts["answers"][("POST", "/v1/datasources/custom-mcp/member-tools")] = (
        502, {"code": code, "detail": "Refused.", "request_id": "r"})
    hosts["answers"][("GET", f"/v1/custom-mcp/connections/{NAME}/tools")] = (
        200, {"tools_revision": 0, "tools": []})

    response = _client().get(f"{ROUTES}/connections/{NAME}/tools")

    assert response.status_code == 502
    assert response.json() == {"error": code, "detail": "Refused."}


def test_a_switch_save_reaches_auth_while_the_server_is_down(hosts):
    hosts["answers"][("PUT", f"/v1/custom-mcp/connections/{NAME}/tools")] = (200, SNAPSHOT)

    response = _client().put(f"{ROUTES}/connections/{NAME}/tools", json={"tools": ["search"], "tools_revision": 3})

    assert response.json() == SNAPSHOT
    request, timeout = hosts["sent"][0]
    assert json.loads(request.content) == {"tools": ["search"], "tools_revision": 3}
    assert (request.headers["authorization"], timeout) == (BEARER, 10.0)
    assert _sent(hosts, GATEWAY) == []


def test_reconnect_relays_the_confirmed_hosts_and_lists_nothing(hosts):
    answer = {"status": "pending", "auth_url": "https://login.example.com/a", "state": "s2", "name": NAME}
    hosts["answers"][("POST", f"/v1/custom-mcp/connections/{NAME}/reconnect")] = (200, answer)

    response = _client().post(f"{ROUTES}/connections/{NAME}/reconnect",
                              json={"confirmed_login_hosts": ["login.example.com"]})

    assert response.json() == answer
    assert json.loads(hosts["sent"][0][0].content) == {"confirmed_login_hosts": ["login.example.com"]}
    assert _sent(hosts, GATEWAY) == []


def test_a_client_secret_on_connect_reaches_auth_and_no_log_line(hosts, monkeypatch):
    planted = "planted-client-secret-7f3c9e2b"
    lines: list[str] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage() + repr(record.__dict__))

    keep = _Keep(level=logging.DEBUG)
    # Every logger that could see the request, at DEBUG, through setLevel so the level cache is cleared.
    loggers = [logging.getLogger(name) for name in ("", "httpx", "httpcore", "cowork.connectors.custom_mcp_cloud")]
    levels = [logger.level for logger in loggers]
    monkeypatch.setattr(logging.root.manager, "disable", 0)
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(keep)
    hosts["answers"][("POST", "/v1/custom-mcp/start")] = (400, {"error": "registration_failed", "detail": "No."})
    try:
        response = _client().post(f"{ROUTES}/connect", json={
            "server_url": "https://mcp.example.com/mcp", "label": "x", "client_id": "c", "client_secret": planted})
    finally:
        for logger, level in zip(loggers, levels):
            logger.removeHandler(keep)
            logger.setLevel(level)

    assert json.loads(hosts["sent"][0][0].content)["client_secret"] == planted
    assert planted not in response.text
    assert lines, "nothing was logged, so the check proved nothing"
    assert not [line for line in lines if planted in line]
