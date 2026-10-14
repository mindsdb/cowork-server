"""DNS-rebound requests are refused before any handler when token auth is off.

A rebound page reaches the loopback server under the attacker's hostname and
the browser sends that name as Host, with no Origin (the page is same-origin
after rebinding) or with the attacker's origin. Everything here goes through
create_app() so the real middleware registration is what is under test.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cowork.auth_middleware import _read_token
from cowork.common.settings.app_settings import get_app_settings
from cowork.host_middleware import TrustedHostMiddleware

FOREIGN_HOST = "rebind.attacker-controlled.example:26866"
ATTACKER_ORIGIN = f"http://{FOREIGN_HOST}"

# The ticket's matrix: secret routes, conversations, project files, settings
# mutation, and response creation.
SENSITIVE_REQUESTS = [
    ("GET", "/api/v1/settings/reveal-key/openai", None),
    ("GET", "/api/v1/settings/raw", None),
    ("GET", "/api/v1/connectors/oauth/google/credentials", None),
    ("GET", "/api/v1/conversations/", None),
    ("GET", "/api/v1/conversations/conv_1/items", None),
    ("GET", "/api/v1/projects/", None),
    ("GET", "/api/v1/projects/general/files", None),
    ("PUT", "/api/v1/projects/general/files/notes.txt", {"content": "pwned"}),
    ("PUT", "/api/v1/settings/openai_api_key", {"value": "sk-attacker"}),
    ("POST", "/api/v1/responses/", {"input": "hi", "model": "x"}),
]


def _app(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    get_app_settings.cache_clear()
    from cowork.server import create_app

    return create_app()


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def auth_off(monkeypatch, tmp_path):
    return _app(monkeypatch, tmp_path, COWORK_REQUIRE_AUTH="false")


@pytest.fixture
def legacy_desktop(monkeypatch, tmp_path):
    # A pre-token desktop build: loopback host, no owner, auth never set.
    app = _app(
        monkeypatch,
        tmp_path,
        COWORK_REQUIRE_AUTH=None,
        COWORK_SERVER_HOST="127.0.0.1",
        COWORK_SERVER_OWNER=None,
    )
    assert get_app_settings().require_auth is False
    return app


def _send(client, method, path, body, **headers):
    return client.request(method, path, json=body, headers=headers)


@pytest.mark.parametrize("app_fixture", ["auth_off", "legacy_desktop"])
@pytest.mark.parametrize("origin", [None, ATTACKER_ORIGIN])
@pytest.mark.parametrize(("method", "path", "body"), SENSITIVE_REQUESTS)
def test_foreign_host_is_refused_on_every_sensitive_route(request, app_fixture, origin, method, path, body):
    client = TestClient(request.getfixturevalue(app_fixture), client=("127.0.0.1", 50000))
    headers = {"host": FOREIGN_HOST}
    if origin:
        headers["origin"] = origin

    resp = _send(client, method, path, body, **headers)

    assert resp.status_code == 403
    assert resp.json() == {"detail": "untrusted host"}


def test_foreign_host_never_reaches_the_agent(auth_off):
    client = TestClient(auth_off, client=("127.0.0.1", 50000))
    body = {"input": "hi", "model": "x"}
    with patch("cowork.api.v1.endpoints.responses.ResponsesHandler") as handler:
        handler.return_value.handle = AsyncMock(return_value={})
        refused = client.post("/api/v1/responses/", json=body, headers={"host": FOREIGN_HOST})
        assert refused.status_code == 403
        assert handler.call_count == 0

        # Same request from the desktop reaches the handler, so the patch target is live.
        client.post("/api/v1/responses/", json=body, headers={"host": "127.0.0.1:26866"})
        assert handler.call_count == 1


def test_foreign_host_changes_no_setting(auth_off):
    client = TestClient(auth_off, client=("127.0.0.1", 50000))
    local = {"host": "127.0.0.1:26866"}
    before = client.get("/api/v1/settings/", headers=local).json()

    resp = client.put("/api/v1/settings/openai_api_key", json={"value": "sk-attacker"}, headers={"host": FOREIGN_HOST})

    assert resp.status_code == 403
    assert client.get("/api/v1/settings/", headers=local).json() == before


@pytest.mark.parametrize(
    "host",
    [
        "localhost:26866",
        "LOCALHOST:26866",
        "localhost",
        "127.0.0.1:26866",
        "127.0.0.1:9999",
        "[::1]:26866",
        "192.168.1.10:26866",
        "10.0.0.5",
    ],
)
def test_local_names_and_ip_addresses_are_accepted(auth_off, host):
    client = TestClient(auth_off, client=("127.0.0.1", 50000))
    assert client.get("/api/v1/conversations/", headers={"host": host}).status_code == 200


def test_desktop_request_without_origin_is_accepted(auth_off):
    client = TestClient(auth_off, base_url="http://127.0.0.1:26866")
    assert client.get("/api/v1/conversations/").status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.local",
        "app.localhost",
        "localhost.attacker.example",
        "127.0.0.1.nip.io",
        "2130706433",
        "[::1",
    ],
)
def test_names_a_page_can_be_served_from_are_refused(auth_off, host):
    client = TestClient(auth_off, client=("127.0.0.1", 50000))
    assert client.get("/api/v1/conversations/", headers={"host": host}).status_code == 403


def test_configured_hostnames_are_accepted(monkeypatch, tmp_path):
    app = _app(
        monkeypatch,
        tmp_path,
        COWORK_REQUIRE_AUTH="false",
        COWORK_ALLOWED_HOSTS='["cowork.example.com", "Tunnel.Example:8443"]',
    )
    client = TestClient(app)

    for host in ("cowork.example.com", "cowork.example.com:443", "tunnel.example"):
        assert client.get("/api/v1/conversations/", headers={"host": host}).status_code == 200
    assert client.get("/api/v1/conversations/", headers={"host": "other.example.com"}).status_code == 403


def test_forwarded_host_does_not_override_host(auth_off):
    client = TestClient(auth_off, client=("127.0.0.1", 50000))
    resp = client.get(
        "/api/v1/conversations/",
        headers={"host": FOREIGN_HOST, "x-forwarded-host": "localhost", "x-forwarded-for": "127.0.0.1"},
    )
    assert resp.status_code == 403


def test_token_auth_is_the_boundary_when_on(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, COWORK_REQUIRE_AUTH="true")
    client = TestClient(app, client=("127.0.0.1", 50000))
    token = _read_token(tmp_path / ".env")

    assert client.get("/api/v1/conversations/", headers={"host": FOREIGN_HOST}).status_code == 401
    authorized = {"host": FOREIGN_HOST, "authorization": f"Bearer {token}"}
    assert client.get("/api/v1/conversations/", headers=authorized).status_code == 200


@pytest.mark.parametrize(("method", "path", "body"), SENSITIVE_REQUESTS)
def test_non_loopback_sensitive_requests_still_need_the_token(monkeypatch, tmp_path, method, path, body):
    app = _app(monkeypatch, tmp_path, COWORK_REQUIRE_AUTH="true")
    client = TestClient(app, client=("192.168.1.20", 50000), base_url="http://192.168.1.5:26866")

    assert _send(client, method, path, body).status_code == 401


def test_org_mode_has_no_host_check(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, COWORK_TENANCY_MODE="org")
    client = TestClient(app)
    assert client.get("/api/v1/health/", headers={"host": "cowork.mindshub.ai"}).status_code == 200


def _tiny_app(exempt=()):
    app = FastAPI()

    @app.post("/hook")
    def hook():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hi")
        await websocket.close()

    app.add_middleware(TrustedHostMiddleware, exempt_paths=set(exempt))
    return app


def test_signed_webhook_paths_skip_the_check():
    assert TestClient(_tiny_app()).post("/hook", headers={"host": FOREIGN_HOST}).status_code == 403
    assert TestClient(_tiny_app({"/hook"})).post("/hook", headers={"host": FOREIGN_HOST}).status_code == 200


def test_websocket_with_foreign_host_is_closed():
    client = TestClient(_tiny_app(), base_url="http://127.0.0.1")
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "hi"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"host": FOREIGN_HOST}):
            pass
