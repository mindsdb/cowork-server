"""Live apps' switch: off unless ANTON_LIVE_APP_SLOTS is set, refused at start
without what keeps each app to its launcher, and with every live-app route
404 and the health flag false while off."""

from __future__ import annotations

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
    install_live_app_slots,
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
    install_live_app_slots(stack)
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
    import anton.core.session as session_module

    real = session_module.ChatSessionConfig
    turn_on_live_apps(stack)
    stack.setattr(session_module, "ChatSessionConfig", real)
    if "live_app_host" in real.__dataclass_fields__:
        pytest.skip("the installed anton's ChatSessionConfig takes live_app_host")
    assert "has a ChatSessionConfig that takes no live_app_host" in _refusal()


def test_a_slot_list_that_doesnt_parse_is_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv(SLOTS_VARIABLE, "http://live-app-1:8001/path")
    assert "ANTON_LIVE_APP_SLOTS doesn't parse" in _refusal()


def test_live_apps_without_the_viewer_login_are_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_LIVE_APPS_VIEWER_LOGIN_URL", " ")
    assert "needs COWORK_LIVE_APPS_VIEWER_LOGIN_URL" in _refusal()


def test_live_apps_with_a_short_launch_token_are_refused(stack):
    turn_on_live_apps(stack)
    stack.setenv("COWORK_LIVE_APPS_LAUNCH_TOKEN", "x" * 31)
    assert "needs COWORK_LIVE_APPS_LAUNCH_TOKEN of at least 32 characters" in _refusal()


# ── off: every route 404, the flag false ─────────────────────────────────────

_ROUTES = [
    ("GET", "/api/v1/live-apps/1/access", None),
    ("GET", "/api/v1/live-apps/1/stopped-page", None),
    ("POST", "/api/v1/live-apps/1/stopped", {"boot_id": "b" * 32, "reason": "idle"}),
    ("GET", "/api/v1/live-apps/mine", None),
    ("POST", "/api/v1/artifacts/live-app-share", {"projectId": "00000000-0000-4000-8000-000000000001", "slug": "app", "everyone": True}),
    ("GET", "/api/v1/artifacts/shared-live-apps", None),
    ("POST", "/api/v1/artifacts/live-app-duplicate", {"shareId": "00000000-0000-4000-8000-000000000001", "projectId": "00000000-0000-4000-8000-000000000002", "confirmed": True}),
]


@pytest.mark.parametrize(("method", "path", "body"), _ROUTES)
def test_every_live_app_route_is_404_with_live_apps_off(stack, method, path, body):
    install_live_app_slots(stack)
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
