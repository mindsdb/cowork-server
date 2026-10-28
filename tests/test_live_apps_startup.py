"""Live apps' switch: off unless ANTON_LIVE_APP_SLOTS is set, refused at start
without what keeps each app to its launcher, and with every live-app route
404 and the health flag false while off."""

from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.code_sandbox import live_apps_on
from cowork.common.settings.app_settings import get_app_settings
from tests._live_apps import (
    A,
    BEARER_TOKEN,
    MODULE,
    SLOT_ADDRESSES,
    SLOTS_VARIABLE,
    clear_live_app_rows,
    turn_on_live_apps,
)

SHARED_PROJECTS = "conversations,files,skills,connections,schedules,pins,memory,settings"


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    monkeypatch.delenv(SLOTS_VARIABLE, raising=False)
    get_app_settings.cache_clear()
    yield monkeypatch
    clear_live_app_rows()
    get_app_settings.cache_clear()


def _hide_the_slots(monkeypatch) -> None:
    """The installed anton as one without anton.core.artifacts.live_app_slots."""
    import anton.core.artifacts as artifacts_package

    monkeypatch.setitem(sys.modules, MODULE, None)
    monkeypatch.delattr(artifacts_package, "live_app_slots", raising=False)


def _refusal() -> str:
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError) as refused:
        cowork_server.create_app()
    return str(refused.value)


# ── the switch ───────────────────────────────────────────────────────────────


def test_live_apps_are_off_without_the_variable(stack):
    assert live_apps_on() is False


def test_live_apps_are_off_on_an_anton_without_the_slots(stack):
    _hide_the_slots(stack)
    stack.setenv(SLOTS_VARIABLE, ",".join(SLOT_ADDRESSES))
    assert live_apps_on() is False


def test_live_apps_are_on_with_the_overlay(stack):
    turn_on_live_apps(stack)
    assert live_apps_on() is True


# ── startup ──────────────────────────────────────────────────────────────────


def test_a_stack_with_every_precondition_starts(stack):
    turn_on_live_apps(stack)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_live_apps_with_the_sandbox_off_are_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_CODE_SANDBOX", "off")
    stack.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    assert "ANTON_LIVE_APP_SLOTS needs COWORK_CODE_SANDBOX=landlock" in _refusal()


def test_live_apps_without_identity_headers_are_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    assert "ANTON_LIVE_APP_SLOTS needs COWORK_TRUST_IDENTITY_HEADERS" in _refusal()


def test_live_apps_with_projects_shared_are_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_PERSONAL_RESOURCES", SHARED_PROJECTS)
    assert "ANTON_LIVE_APP_SLOTS needs projects in COWORK_PERSONAL_RESOURCES" in _refusal()


def test_live_apps_on_an_anton_without_the_slots_are_refused(stack):
    turn_on_live_apps(stack)
    _hide_the_slots(stack)
    message = _refusal()
    assert "ANTON_LIVE_APP_SLOTS needs an anton with live-app slots" in message
    assert "has no anton.core.artifacts.live_app_slots" in message


def test_live_apps_on_an_anton_whose_session_takes_no_host_are_refused(stack):
    """Whatever anton is installed: a ChatSessionConfig with no live_app_host."""
    import dataclasses

    import anton.core.session as session_module

    @dataclasses.dataclass
    class NoHost:
        pass

    turn_on_live_apps(stack)
    stack.setattr(session_module, "ChatSessionConfig", NoHost)
    assert "has a ChatSessionConfig that takes no live_app_host" in _refusal()


def test_live_apps_on_an_anton_whose_slots_lack_a_name_are_refused(stack):
    module = turn_on_live_apps(stack)
    stack.delattr(module, "SlotStatus")
    assert "has a live_app_slots without SlotStatus" in _refusal()


def test_live_apps_anton_reads_as_on_are_checked_without_the_variable(stack):
    """Every runtime decision asks anton, so the checks run when anton reads
    live apps as on even with the variable unset."""
    import cowork.code_sandbox as code_sandbox

    stack.setattr(code_sandbox, "live_apps_on", lambda: True)
    stack.setenv("COWORK_CODE_SANDBOX", "off")
    stack.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    assert "ANTON_LIVE_APP_SLOTS needs COWORK_CODE_SANDBOX=landlock" in _refusal()


def test_a_slot_list_that_doesnt_parse_is_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv(SLOTS_VARIABLE, "http://live-app-1:8001/path")
    assert "ANTON_LIVE_APP_SLOTS doesn't parse" in _refusal()


@pytest.mark.parametrize("value", [",", " , ", "http://live-app-1:8001,,http://live-app-2:8001"])
def test_a_slot_list_with_a_blank_entry_is_refused(stack, value):
    """anton keeps a blank entry rather than renumbering the slots after it,
    and refuses it, so the server refuses to start."""
    turn_on_live_apps(stack)
    stack.setenv(SLOTS_VARIABLE, value)
    assert "ANTON_LIVE_APP_SLOTS doesn't parse" in _refusal()


@pytest.mark.parametrize(
    "value",
    [" ", "viewer-login:4180", "ftp://viewer-login:4180", "http://viewer-login:4180/?next=x", " http://viewer-login:4180"],
)
def test_live_apps_without_a_viewer_login_address_are_refused(stack, value):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_LIVE_APPS_VIEWER_LOGIN_URL", value)
    assert "needs COWORK_LIVE_APPS_VIEWER_LOGIN_URL" in _refusal()


@pytest.mark.parametrize("value", ["http://viewer-login:4180", "https://login.example.com/viewer/"])
def test_a_viewer_login_address_with_a_path_starts(stack, value):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_LIVE_APPS_VIEWER_LOGIN_URL", value)
    get_app_settings.cache_clear()
    cowork_server.create_app()


@pytest.mark.parametrize("token", ["x" * 31, " " + "x" * 32, "x" * 32 + " ", "é" * 32, "x" * 31 + "\x07"])
def test_live_apps_with_a_launch_token_a_header_cant_carry_are_refused(stack, token):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_LIVE_APPS_LAUNCH_TOKEN", token)
    message = _refusal()
    assert "needs COWORK_LIVE_APPS_LAUNCH_TOKEN of at least 32 printable ASCII characters" in message
    assert token.strip() not in message


# ── off: every route 404, the flag false ─────────────────────────────────────

_ROUTES = [
    ("GET", "/api/v1/live-apps/1/access", None),
    ("GET", "/api/v1/live-apps/1/stopped-page", None),
    ("POST", "/api/v1/live-apps/1/stopped", {"boot_id": "b" * 32, "reason": "idle"}),
    ("GET", "/api/v1/live-apps/mine", None),
    ("POST", "/api/v1/artifacts/live-app-share", {"projectId": "00000000-0000-4000-8000-000000000001", "slug": "app", "everyone": True}),
    ("GET", "/api/v1/artifacts/shared-live-apps", None),
    ("POST", "/api/v1/artifacts/live-app-duplicate", {"shareId": "00000000-0000-4000-8000-000000000001", "projectId": "00000000-0000-4000-8000-000000000002", "confirmed": True}),
    ("GET", "/api/v1/artifacts/live-app-shares?projectId=00000000-0000-4000-8000-000000000001&slug=app", None),
    ("DELETE", "/api/v1/artifacts/live-app-share/00000000-0000-4000-8000-000000000001", None),
    # Requests no live-app route would take: they are 404 too, never 422.
    ("POST", "/api/v1/artifacts/live-app-share", {}),
    ("POST", "/api/v1/artifacts/live-app-duplicate", {}),
    ("POST", "/api/v1/live-apps/1/stopped", {}),
    ("GET", "/api/v1/artifacts/live-app-shares", None),
    ("DELETE", "/api/v1/artifacts/live-app-share/not-a-uuid", None),
    ("GET", "/api/v1/live-apps/x/access", None),
]


@pytest.mark.parametrize(
    "path", ["/api/v1/artifacts/live-app-share", "/api/v1/artifacts/live-app-duplicate", "/api/v1/live-apps/1/stopped"]
)
def test_a_body_that_isnt_json_is_404_with_live_apps_off(stack, path):
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app())

    response = client.post(path, content=b"{not json", headers={**A, "Content-Type": "application/json"})

    assert response.status_code == 404


@pytest.mark.parametrize(("method", "path", "body"), _ROUTES)
def test_every_live_app_route_is_404_with_live_apps_off(stack, method, path, body):
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app())

    response = client.request(method, path, json=body, headers=A)

    assert response.status_code == 404


def test_the_slot_routes_need_the_bearer_while_live_apps_are_off(stack):
    """Off, the slot paths aren't let through without the bearer token."""
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app())

    assert client.get("/api/v1/live-apps/1/access").status_code == 401


def test_health_says_live_apps_are_off(stack):
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app())

    assert client.get("/api/v1/health/").json()["live_apps"] is False


def test_health_says_live_apps_are_on(stack):
    turn_on_live_apps(stack)
    client = TestClient(cowork_server.create_app())

    assert client.get("/api/v1/health/").json()["live_apps"] is True


# ── on: the preview route and the in-server backend stay refused ─────────────


def _never(*_args, **_kwargs):
    raise AssertionError("live apps never reach the in-server backend paths")


async def test_the_preview_proxy_still_forwards_nothing(stack, tmp_path):
    from anton.core.backends import sandbox
    from starlette.requests import Request

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import preview_proxy

    turn_on_live_apps(stack)
    folder = tmp_path / "proj" / ".anton" / "artifacts" / "app"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"port": 26866}), encoding="utf-8")
    stack.setattr(preview_proxy, "preview_mount_for", lambda token, *, scope: folder)
    stack.setattr(preview_proxy, "get_proxy_client", _never)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {"type": "http", "method": "GET", "path": "/x", "query_string": b"", "headers": []},
        receive=receive,
    )
    response = await preview_proxy.proxy_artifact_request("token", "api/items", request, scope=LOCAL_SCOPE)

    assert response.status_code == 403
    assert response.body.decode() == sandbox.BACKENDS_OFF_MESSAGE


async def test_no_backend_starts_in_this_server(stack, tmp_path):
    from anton.core.backends import sandbox

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import artifacts, scratchpad_runtime

    turn_on_live_apps(stack)
    stack.setattr(scratchpad_runtime, "WorkspaceScopedPool", _never)
    stack.setattr(artifacts, "_probe_port", _never)
    stack.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    folder = tmp_path / "projects" / "proj" / ".anton" / "artifacts" / "app"
    folder.mkdir(parents=True)

    status = await artifacts._ensure_backend_running(folder, 8000, scope=LOCAL_SCOPE)

    assert status == (False, sandbox.BACKENDS_OFF_MESSAGE, 0)
    assert artifacts._LAUNCHED_BACKENDS == {}
