"""Live apps for tests: anton's live-app slots, a viewer sign-in and app slots
answering over HTTP, and the people who use them.

anton's ``anton.core.artifacts.live_app_slots`` (owner: anton,
``anton/core/artifacts/live_app_slots.py``) isn't in the anton this server
pins yet. Where the installed anton lacks it, install_live_app_slots puts a
stand-in written from that module's published names and limits in its place:
the switch, the slot addresses, the launch list and its caps, the reply cap,
and the models this server reads. Where the installed anton has it, the real
module is used, and an error importing it fails the test.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import personal_partition_id

SLOTS_VARIABLE = "ANTON_LIVE_APP_SLOTS"
SLOT_ADDRESSES = ("http://live-app-1:8001", "http://live-app-2:8001", "http://live-app-3:8001")
VIEWER_LOGIN = "http://viewer-login:4180"
LAUNCH_TOKEN = "launch-token-for-tests-" + "x" * 20
MODULE = "anton.core.artifacts.live_app_slots"

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
SUB_C = "33333333-3333-4333-8333-333333333333"
EMAIL_A = "alice@example.com"
EMAIL_B = "bob@example.com"
EMAIL_C = "carol@example.com"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
PARTITION_C = personal_partition_id(DEPLOYMENT, SUB_C)


# ── the stand-in for anton's module ──────────────────────────────────────────


class _LiveAppLauncher(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=254)


class _LiveAppLaunch(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot: int
    slug: str
    artifact_dir: str
    address: str
    boot_id: str


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _AppHashes(_Wire):
    backend: str
    requirements: str
    connections: str


class _SlotApp(_Wire):
    artifact: str
    launcher: _LiveAppLauncher
    started_at: str
    expires_at: str
    last_request_at: str | None = None
    hashes: _AppHashes
    env_names: tuple[str, ...] = ()


class _SlotStatus(_Wire):
    version: Literal[1]
    boot_id: str = Field(min_length=1)
    booted_at: str
    state: Literal["empty", "running", "exited"]
    app: _SlotApp | None = None


_SLOT_ADDRESS = re.compile(r"^http://[A-Za-z0-9.-]+:(?P<port>\d{1,5})$")


def _live_apps_enabled() -> bool:
    return bool(os.environ.get(SLOTS_VARIABLE, "").strip())


def _live_app_slot_addresses() -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in os.environ.get(SLOTS_VARIABLE, "").split(",") if entry.strip())
    for entry in entries:
        match = _SLOT_ADDRESS.match(entry)
        if match is None or not 1 <= int(match["port"]) <= 65535:
            raise ValueError(f"{SLOTS_VARIABLE} entry {entry!r} isn't http://<host>:<port>")
    return entries


def _stand_in() -> types.ModuleType:
    module = types.ModuleType(MODULE)
    module.live_apps_enabled = _live_apps_enabled
    module.live_app_slot_addresses = _live_app_slot_addresses
    module.LiveAppLauncher = _LiveAppLauncher
    module.LiveAppLaunch = _LiveAppLaunch
    module.SlotStatus = _SlotStatus
    module.LAUNCH_FILES = ("backend.py", "requirements.txt", "state_manifest.json")
    module.LAUNCH_DIRS = ("static",)
    module.MAX_LAUNCH_FILES = 500
    module.MAX_LAUNCH_FILE_BYTES = 8 * 1024 * 1024
    module.MAX_LAUNCH_TOTAL_BYTES = 64 * 1024 * 1024
    module.MAX_REPLY_BYTES = 1024 * 1024
    return module


def install_live_app_slots(monkeypatch) -> types.ModuleType:
    """anton's live-app slots, real or stood in, and a ChatSessionConfig that
    takes ``live_app_host``. The stand-in goes in only where the installed
    anton has no such module, so a real module that fails to import fails
    the test instead of passing on the stand-in."""
    import anton.core.artifacts as artifacts_package
    import anton.core.session as session_module

    if sys.modules.get(MODULE) is None and importlib.util.find_spec(MODULE) is None:
        module = _stand_in()
        monkeypatch.setitem(sys.modules, MODULE, module)
        monkeypatch.setattr(artifacts_package, "live_app_slots", module, raising=False)
    else:
        import anton.core.artifacts.live_app_slots as module
    config = session_module.ChatSessionConfig
    if "live_app_host" not in {item.name for item in dataclasses.fields(config)}:

        @dataclass
        class ChatSessionConfigWithHost(config):
            live_app_host: object = None

        monkeypatch.setattr(session_module, "ChatSessionConfig", ChatSessionConfigWithHost)
    return module


def turn_on_live_apps(monkeypatch) -> types.ModuleType:
    """Live apps on, as the overlay sets them on the cowork service."""
    module = install_live_app_slots(monkeypatch)
    monkeypatch.setenv(SLOTS_VARIABLE, ",".join(SLOT_ADDRESSES))
    monkeypatch.setenv("COWORK_LIVE_APPS_VIEWER_LOGIN_URL", VIEWER_LOGIN)
    monkeypatch.setenv("COWORK_LIVE_APPS_LAUNCH_TOKEN", LAUNCH_TOKEN)
    get_app_settings.cache_clear()
    return module


# ── viewer-login and the slots, over HTTP ────────────────────────────────────


@dataclass
class FakeSlot:
    boot_id: str = "0" * 32
    booted_at: str = "2026-10-21T04:40:00Z"
    state: Literal["empty", "running", "exited"] = "empty"
    artifact: str = ""
    launcher_subject: str = ""
    launcher_email: str = ""
    # Answer status with this code instead, or None.
    fails_with: int | None = None


@dataclass
class ViewerUser:
    """A viewer-login session: the ID token's ``sub`` and email."""

    sub: str
    email: str


@dataclass
class FakeServices:
    """viewer-login's /oauth2/userinfo and each slot's GET /status.

    ``sessions`` maps a viewer cookie's value to the person signed in with
    it. ``viewer_replies`` maps a cookie's value to a raw reply viewer-login
    gives instead (a body oauth2-proxy can send, a redirect, an exception the
    transport raises). ``refresh_cookies`` adds the Set-Cookie headers
    oauth2-proxy sends when it refreshes a session. Every request each
    service gets is kept in ``requests``.
    """

    sessions: dict[str, ViewerUser] = field(default_factory=dict)
    slots: dict[int, FakeSlot] = field(default_factory=lambda: {n: FakeSlot() for n in (1, 2, 3)})
    viewer_fails_with: int | None = None
    viewer_replies: dict[str, httpx.Response | Exception] = field(default_factory=dict)
    refresh_cookies: tuple[str, ...] = ()
    requests: list[httpx.Request] = field(default_factory=list)

    def run(self, slot: int, *, boot_id: str, artifact: str, subject: str, email: str) -> None:
        self.slots[slot] = FakeSlot(
            boot_id=boot_id, state="running", artifact=artifact, launcher_subject=subject, launcher_email=email
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == "viewer-login":
            return self._userinfo(request)
        for n, address in enumerate(SLOT_ADDRESSES, start=1):
            if httpx.URL(address).host == host:
                return self._status(n, request)
        return httpx.Response(404)

    def _userinfo(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/oauth2/userinfo":
            return httpx.Response(404)
        if self.viewer_fails_with is not None:
            return httpx.Response(self.viewer_fails_with)
        cookie = request.headers.get("cookie", "")
        value = cookie.split(";")[0].removeprefix("_oauth2_proxy=")
        reply = self.viewer_replies.get(value)
        if isinstance(reply, Exception):
            raise reply
        if reply is not None:
            return reply
        user = self.sessions.get(value)
        if user is None:
            return httpx.Response(401)
        headers = [("set-cookie", cookie) for cookie in self.refresh_cookies]
        return httpx.Response(200, json={"user": user.sub, "email": user.email}, headers=headers)

    def _status(self, n: int, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/status":
            return httpx.Response(404)
        if request.headers.get("authorization") != f"Bearer {LAUNCH_TOKEN}":
            return httpx.Response(401)
        slot = self.slots[n]
        if slot.fails_with is not None:
            return httpx.Response(slot.fails_with)
        app = None
        # The slot program keeps its app in the status after the app exits.
        if slot.state in ("running", "exited"):
            app = {
                "artifact": slot.artifact,
                "app_id": "0" * 64,
                "launcher": {"subject": slot.launcher_subject, "email": slot.launcher_email},
                "started_at": "2026-10-21T04:40:00Z",
                "expires_at": "2026-10-21T12:40:00Z",
                "stop_reason": None,
                "hashes": {"backend": "0" * 64, "requirements": "0" * 64, "connections": "0" * 64, "files": "0" * 64},
            }
        body = {"version": 1, "boot_id": slot.boot_id, "booted_at": slot.booted_at, "state": slot.state, "app": app}
        # Built through the installed anton's own model, so a reply this
        # server's code can't read fails here rather than passing a test.
        status = sys.modules[MODULE].SlotStatus.model_validate(body)
        return httpx.Response(200, content=status.model_dump_json().encode())


def answer_over_http(monkeypatch, services: FakeServices) -> None:
    """Send live apps' viewer-login and slot calls to ``services``."""
    from cowork.services import live_apps

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(services.handle), follow_redirects=False, trust_env=False
        )

    monkeypatch.setattr(live_apps, "_http_client", client)


def viewer_cookie(value: str) -> str:
    return f"_oauth2_proxy={value}"


# ── a stack with people on it ────────────────────────────────────────────────

BEARER_TOKEN = "live-apps-test-token"


def headers_for(sub: str, email: str | None) -> dict[str, str]:
    """A signed-in person's request, as the recording proxy sends it."""
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "X-User-Id": sub, "X-Organization-Id": DEPLOYMENT}
    if email is not None:
        headers["X-User-Email"] = email
    return headers


A = headers_for(SUB_A, EMAIL_A)
B = headers_for(SUB_B, EMAIL_B)
C = headers_for(SUB_C, EMAIL_C)


def person_scope(partition: str, sub: str):
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
    from cowork.db.scoped import TenantScope

    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=ALL_PERSONAL_RESOURCES)


def person(partition: str, sub: str, email: str):
    from cowork.services.live_apps import LiveAppPerson

    return LiveAppPerson(user_id=sub, partition_id=partition, email=email)


ALICE = person(PARTITION_A, SUB_A, EMAIL_A)
BOB = person(PARTITION_B, SUB_B, EMAIL_B)
VIEWER_A = ViewerUser(SUB_A, EMAIL_A)
VIEWER_B = ViewerUser(SUB_B, EMAIL_B)


def clear_live_app_rows() -> None:
    from sqlalchemy import delete
    from sqlmodel import Session

    from cowork.db.session import get_engine
    from cowork.models.live_app import LiveAppCopy, LiveAppLaunchRecord, LiveAppShare
    from cowork.models.project import Project

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for model in (LiveAppCopy, LiveAppLaunchRecord, LiveAppShare):
            raw.exec(delete(model))
        raw.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B, PARTITION_C])))
        raw.commit()


def general_project(scope):
    """The person's own `general` project row, provisioned as a request would."""
    from sqlmodel import Session

    from cowork.db.scoped import ScopedSession
    from cowork.db.session import get_engine
    from cowork.services.projects import ProjectService

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        project = ProjectService(ScopedSession(raw, scope)).ensure_general_for_scope()
        assert project is not None
        return project.id, project.path
