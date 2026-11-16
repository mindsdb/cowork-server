"""The agent's browser: the browser router on the web, the desktop app's own
browser on the desktop, the /browse routes on both, each turn's browser, and
the `response.browser_session_opened` event on both turn paths."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

import httpx
import pytest
from anton.core.browser.config import BrowserConfig
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

from cowork.api.v1.router import api_router
from cowork.common.settings import runtime_credential
from cowork.common.settings.runtime_credential import LocalBrowserHost
from cowork.handlers.jev_route import decisions_model as real_decisions_model
from cowork.principal import Principal, get_principal
from cowork.services import browser as svc
from cowork.services.browser_contract import SESSION_ID

ROUTER = "https://app.example.test/browser"
SCOPE_KEY = "scope-key-" + "s" * 32
ORG = "org-1"
APP_BROWSER = LocalBrowserHost(url="http://127.0.0.1:51234", token=SecretStr("k" * 43))
FIXTURES = Path(__file__).parent / "fixtures"


# ── fixture files ──────────────────────────────────────────────────────────


class _Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    where: str
    kind: Literal["request", "response", "event", "turn_block", "constructor", "classification"]
    status: int | None = None
    refused: bool = False
    body: Any


class _FixtureFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    limits: dict[str, int] = {}
    examples: dict[str, _Example]


def _fixture(folder: str, name: str) -> _FixtureFile:
    return _FixtureFile.model_validate_json((FIXTURES / folder / name).read_text())


# ── fakes ──────────────────────────────────────────────────────────────────


class Call(NamedTuple):
    method: str
    url: str
    body: Any
    auth: str
    scope_key: str = ""


def _embed_answer(session_id: str, *, view_url: str | None = None) -> dict:
    return {
        "session_id": session_id,
        "view_url": view_url or f"{ROUTER}/sessions/{session_id}/view?et=tok",
        "ticket_expires_at": 1999999120,
        "lease_expires_at": 1999999900,
    }


def _route_router(monkeypatch, handler) -> list[Call]:
    """Send every browser call this server makes to ``handler``, recording each."""
    calls: list[Call] = []

    def recording(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append(Call(
            request.method, str(request.url), body, request.headers.get("authorization", ""),
            request.headers.get("x-browser-scope-key", ""),
        ))
        return handler(request, body)

    real = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(recording), **kw))
    return calls


@pytest.fixture
def nothing_goes_out(monkeypatch):
    """Any call to the router, or any MindsHub read, fails the test."""
    from cowork.services import mindshub_teams

    def refuse(request, body):
        raise AssertionError(f"called out: {request.method} {request.url}")

    async def no_mindshub(path, bearer):
        raise AssertionError(f"asked MindsHub: {path}")

    monkeypatch.setattr(mindshub_teams, "get_mindshub_reply", no_mindshub)
    return _route_router(monkeypatch, refuse)


@pytest.fixture
def scope_key_file(tmp_path):
    """The router's scope key, mounted as the chart mounts it."""
    path = tmp_path / "scope_key"
    path.write_text(SCOPE_KEY + "\n")
    return path


@pytest.fixture
def web(monkeypatch, scope_key_file):
    """The web app: org tenancy, with the browser router configured."""
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", ROUTER + "/")
    monkeypatch.setenv("COWORK_BROWSER_SCOPE_KEY_FILE", str(scope_key_file))
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def router(monkeypatch, web):
    """A fake router at ROUTER. ``state.answers`` overrides a path's answer."""
    state = SimpleNamespace(status="none", answers={})

    def handler(request: httpx.Request, body: Any) -> httpx.Response:
        path = request.url.path
        if path in state.answers:
            return state.answers[path]
        if path == "/browser/_status":
            return httpx.Response(200, json={"status": state.status})
        if path == "/browser/_warm":
            state.status = "starting"
            return httpx.Response(200, json={"status": "starting"})
        if path == "/browser/_embed":
            return httpx.Response(200, json=_embed_answer(body["session_id"]))
        if path == "/browser/_scope":
            return httpx.Response(200, json=body)
        if path.startswith("/browser/sessions/") and path.endswith("/approve"):
            return httpx.Response(200, json={**body, "approved": True, "expires_at": 1999999300})
        if path.startswith("/browser/sessions/") and path.endswith("/decline"):
            return httpx.Response(200, json={"declined": True})
        if path == "/browser/logins" and request.method == "DELETE":
            return httpx.Response(200, json={"cleared": True})
        return httpx.Response(404, json={"error": "not_found", "message": "Not found."})

    state.calls = _route_router(monkeypatch, handler)
    return state


def _entitlements(enabled: bool, *, block: bool = True, reason: str = "not_permitted") -> dict:
    answer: dict[str, Any] = {"permissions": ["browser.use"] if enabled else []}
    if block:
        answer["browser"] = {
            "enabled": enabled, "daily_bytes": 1073741824,
            "disabled_reason": None if enabled else reason,
        }
    return answer


@pytest.fixture
def mindshub(monkeypatch):
    """MindsHub's entitlements answer, per credential. ``replies`` overrides one."""
    from cowork.services import mindshub_teams
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    state = SimpleNamespace(asked=[], replies={})

    async def reply(path, bearer):
        state.asked.append((path, bearer))
        return state.replies.get(bearer, MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(True)))

    mindshub_teams.reset_caches_for_tests()
    monkeypatch.setattr(mindshub_teams, "get_mindshub_reply", reply)
    yield state
    mindshub_teams.reset_caches_for_tests()


@pytest.fixture
def app_browser():
    """The desktop app registered its browser for this test."""
    runtime_credential.set_browser_host(APP_BROWSER)
    yield APP_BROWSER
    runtime_credential.clear_browser_host()


@pytest.fixture(autouse=True)
def decisions(monkeypatch):
    """A person whose LLM is MindsHub, with Jev on: the browser has a
    decisions model. ``model`` and ``mindshub`` change that for one test."""
    from cowork.handlers import jev_route

    state = SimpleNamespace(model="mindshub", mindshub=True, own=None)
    monkeypatch.setattr(jev_route, "decisions_model", lambda settings: state.model)
    monkeypatch.setattr(jev_route, "own_decision_target", lambda settings: state.own)
    monkeypatch.setattr(svc, "llm_on_mindshub", lambda settings: state.mindshub)
    return state


@pytest.fixture(autouse=True)
def no_app_browser():
    runtime_credential.clear_browser_host()
    yield
    runtime_credential.clear_browser_host()


def _delete_browser_settings(scope) -> None:
    from sqlmodel import Session

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.common.settings.user_settings import UserSettings
    from cowork.db.session import get_engine
    from cowork.services.settings import SettingService

    with Session(get_engine(get_app_settings().database.uri)) as session:
        service = SettingService(session, scope)
        for key in [k for k in UserSettings.model_fields if k.startswith("browser_")]:
            service.delete_setting(key)


@pytest.fixture
def client():
    """The desktop: local tenancy, no principal."""
    from cowork.db.scoped import LOCAL_SCOPE

    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: None
    yield TestClient(app)
    _delete_browser_settings(LOCAL_SCOPE)


@pytest.fixture
def web_client(web):
    """The web app, where each request names its person in X-Test-User and
    carries their credential as the web shell does, in Authorization."""
    from cowork.db.scoped import TenantScope

    app = FastAPI()
    app.include_router(api_router)

    @app.middleware("http")
    async def as_person(request, call_next):
        user = request.headers.get("X-Test-User")
        if user:
            request.state.principal = Principal(user_id=user, org_id=ORG)
        return await call_next(request)

    yield TestClient(app)
    for user in ("alice", "bob"):
        _delete_browser_settings(TenantScope(org_mode=True, org_id=ORG, user_id=user))


def _as(user: str) -> dict[str, str]:
    return {"X-Test-User": user, "Authorization": f"Bearer jwt-{user}"}


def _core(answer: dict) -> dict:
    """A status answer without what picks the browser's steps."""
    return {k: answer[k] for k in ("enabled", "available", "status")}


def _detail(response) -> tuple[int, str]:
    return response.status_code, response.json()["detail"]["code"]


# ── settings ───────────────────────────────────────────────────────────────


def test_removed_browser_settings_are_gone(client):
    from cowork.common.settings.user_settings import UserSettings

    browser_keys = {k for k in UserSettings.model_fields if k.startswith("browser_")}
    assert browser_keys == {
        "browser_enabled", "browser_approve_steps", "browser_approve_steps_off_at", "browser_only_when_needed",
    }
    gone_keys = ("browser_mode", "browser_service_url", "browser_service_api_key", "browser_url", "browser_proxy", "browser_city")
    for gone in gone_keys:
        assert client.put(f"/api/v1/settings/{gone}", json={"value": "x"}).status_code == 400, gone


def test_the_browser_switch_defaults_off_and_is_the_persons_own():
    """Beta: the browser stays off until the person turns it on."""
    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    assert UserSettings().browser_enabled is False
    assert setting_is_org_scoped("browser_enabled") is False


def test_browser_steps_wait_for_approval_by_default_and_the_person_can_say_otherwise(client):
    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    assert UserSettings().browser_approve_steps is True
    assert setting_is_org_scoped("browser_approve_steps") is False
    assert client.put("/api/v1/settings/browser_approve_steps", json={"value": "false"}).status_code == 200
    stored = {row["key"]: row["value"] for row in client.get("/api/v1/settings/").json()}
    assert stored["browser_approve_steps"].lower() == "false"
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.db.scoped import LOCAL_SCOPE

    assert get_user_settings(LOCAL_SCOPE).browser_approve_steps is False


def _settings_as_stored(client, headers: dict[str, str] | None = None) -> dict[str, str]:
    return {row["key"]: row["value"] for row in client.get("/api/v1/settings/", headers=headers or {}).json()}


def _now_ms() -> int:
    return int(time.time() * 1000)


def test_when_approvals_went_off_is_unset_by_default_and_the_persons_own():
    """A string, so an unset stamp reads as no time at all, which approves
    nothing, rather than 0, before which every step was held."""
    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    assert UserSettings().browser_approve_steps_off_at == ""
    assert setting_is_org_scoped("browser_approve_steps_off_at") is setting_is_org_scoped("browser_approve_steps") is False


@pytest.mark.parametrize("route", ["single", "bulk"])
def test_turning_approvals_off_stamps_when_by_the_servers_clock(client, route):
    """The app approves for the person only a step held after this stamp, by
    the `at_ms` this server puts on the step's event, so the stamp is this
    server's clock too. Each write of off moves it, one that finds approvals
    already off included (a held step's Skip approvals); a write of on leaves
    it."""
    def write_approve_steps(value: str) -> None:
        if route == "bulk":
            saved = client.put("/api/v1/settings/", json={"values": {"browser_approve_steps": value}})
        else:
            saved = client.put("/api/v1/settings/browser_approve_steps", json={"value": value})
        assert saved.status_code == 200, saved.text

    before = _now_ms()
    write_approve_steps("false")
    first = int(_settings_as_stored(client)["browser_approve_steps_off_at"])
    assert before <= first <= _now_ms()

    while _now_ms() <= first:
        time.sleep(0.001)
    before = _now_ms()
    write_approve_steps("false")
    second = int(_settings_as_stored(client)["browser_approve_steps_off_at"])
    assert first < before <= second <= _now_ms()

    write_approve_steps("true")
    stored = _settings_as_stored(client)
    assert (stored["browser_approve_steps"].lower(), stored["browser_approve_steps_off_at"]) == ("true", str(second))


def test_a_clock_stepped_back_cant_stamp_approvals_off_before_a_held_step(client, monkeypatch):
    """A step is held at 2,000 s, then the clock steps back to 1,000 s and
    the person turns approvals off. The stamp stays at or after the step's
    `at_ms`, so the app, which approves only a step held after the stamp,
    leaves this one to the person."""
    from anton.core.interaction.elicit import AskRequest
    from anton.core.llm.provider import StreamAskUser

    from cowork.common import event_clock
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def held_step() -> dict:
        async def events():
            yield StreamAskUser(id="browser:approve:t-a:pa-1", request=AskRequest(prompt="Place the order?"))

        sunk: list[tuple[str, dict]] = []
        async for _ in format_responses_stream(events(), model="m", event_sink=lambda t, d: sunk.append((t, d))):
            pass
        return next(data for name, data in sunk if name == "response.ask_user")

    monkeypatch.setattr(event_clock, "_latest_ms", 0)
    with monkeypatch.context() as clock:
        clock.setattr(time, "time", lambda: 2_000.0)
        held = asyncio.run(held_step())
        clock.setattr(time, "time", lambda: 1_000.0)
        saved = client.put("/api/v1/settings/browser_approve_steps", json={"value": "false"})
    assert saved.status_code == 200, saved.text
    assert held["at_ms"] == 2_000_000
    assert int(_settings_as_stored(client)["browser_approve_steps_off_at"]) >= held["at_ms"]


def test_a_client_cant_say_when_approvals_went_off(client):
    """Only the server stamps it: a client's 0 would have the app approve
    steps held before the person turned approvals off. The bulk save writes
    nothing, the switch beside it included."""
    single = client.put("/api/v1/settings/browser_approve_steps_off_at", json={"value": "0"})
    bulk = client.put("/api/v1/settings/", json={"values": {
        "browser_approve_steps": "false", "browser_approve_steps_off_at": "0",
    }})
    assert (single.status_code, bulk.status_code) == (400, 400)
    stored = _settings_as_stored(client)
    assert (stored["browser_approve_steps"].lower(), stored["browser_approve_steps_off_at"]) == ("true", "")


def test_on_the_web_each_person_has_their_own_stamp(web_client):
    """Per person, as browser_approve_steps is: one member turning approvals
    off stamps nobody else's."""
    saved = web_client.put(
        "/api/v1/settings/browser_approve_steps", json={"value": "false"}, headers=_as("alice"),
    )
    assert saved.status_code == 200, saved.text
    assert _settings_as_stored(web_client, _as("alice"))["browser_approve_steps_off_at"] != ""
    assert _settings_as_stored(web_client, _as("bob"))["browser_approve_steps_off_at"] == ""


def test_the_browser_shows_only_when_needed_by_default_and_the_person_can_say_otherwise(client):
    from cowork.common.settings.user_settings import UserSettings, get_user_settings, setting_is_org_scoped
    from cowork.db.scoped import LOCAL_SCOPE

    assert UserSettings().browser_only_when_needed is True
    assert setting_is_org_scoped("browser_only_when_needed") is False
    assert client.put("/api/v1/settings/browser_only_when_needed", json={"value": "false"}).status_code == 200
    stored = {row["key"]: row["value"] for row in client.get("/api/v1/settings/").json()}
    assert stored["browser_only_when_needed"].lower() == "false"
    assert get_user_settings(LOCAL_SCOPE).browser_only_when_needed is False


def test_no_setting_offers_an_exit_city_or_a_proxy():
    """Sites see MindsDB's own addresses: no setting picks where the browser
    leaves from or names a proxy that carries it."""
    from cowork.common.settings.user_settings import UserSettings

    for field in UserSettings.model_fields.values():
        text = f"{field.title} {field.description}".lower()
        for word in ("residential", "exit city", "home connection", "proxy provider"):
            assert word not in text, (field.title, word)


def test_router_url_setting_is_https_only_and_normalized(monkeypatch):
    from cowork.common.settings.app_settings import AppSettings

    for raw, expected in [
        ("", ""),
        (ROUTER + "/", ROUTER),
        ("https://cowork.mindshub.ai/browser", "https://cowork.mindshub.ai/browser"),
        ("https://browser.example.com:8443", "https://browser.example.com:8443"),
    ]:
        monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", raw)
        assert AppSettings().browser_router_url == expected, raw
    for bad in [
        "cowork.mindshub.ai/browser", "http://cowork.mindshub.ai/browser", "http://localhost:8080/browser",
        "http://127.0.0.1:8080", "https://cowork.mindshub.ai/browser?x=1", "https://cowork.mindshub.ai/browser#f",
        "https://u:p@cowork.mindshub.ai/browser", "ftp://cowork.mindshub.ai",
    ]:
        monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", bad)
        with pytest.raises(ValidationError):
            AppSettings()


def test_the_router_is_the_webs_alone(monkeypatch):
    """Outside org tenancy the router setting is ignored: the desktop app has
    its own browser, and an in-process turn never gets the router."""
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", ROUTER)
    get_app_settings.cache_clear()
    try:
        assert svc.router_url() == ""
        monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
        get_app_settings.cache_clear()
        assert svc.router_url() == ROUTER
    finally:
        get_app_settings.cache_clear()


# ── sessions and the turn's browser ────────────────────────────────────────


def test_task_sessions_and_project_keys_hash_the_ids():
    sid = svc.task_session("user-a", "conv-1")
    assert sid == "t-" + hashlib.sha256(b"user-a:conv-1").hexdigest()[:24]
    assert sid != svc.task_session("user-a", "conv-2")
    assert svc.project_key("user-a", "proj-1") == "p-" + hashlib.sha256(b"user-a:proj-1").hexdigest()[:24]
    assert svc.project_key("user-a", None) == "" and svc.project_key("user-a", "") == ""
    assert svc.task_session(None, "conv-1") == "t-" + hashlib.sha256(b":conv-1").hexdigest()[:24]


@pytest.mark.parametrize("candidate", [
    "t-0123456789abcdef01234567", "main", "a", "A_b-9", "x" * 64,
    "", "x" * 65, "has space", "../escape", "dot.dot", "new\nline", "trailing\n",
])
def test_the_session_id_pattern_is_antons(candidate):
    """SESSION_ID accepts exactly the session ids anton's BrowserConfig takes as
    its profile, so a session this server names is one the tool can drive."""
    try:
        BrowserConfig(base_url=ROUTER, profile=candidate)
        anton_takes = True
    except ValidationError:
        anton_takes = False
    assert (SESSION_ID.fullmatch(candidate) is not None) is anton_takes


def test_web_turn_block_round_trips_through_anton(web):
    on = SimpleNamespace(browser_enabled=True)
    config = svc.turn_block(on, user_id="user-a", conversation_id="conv-1", project_id="proj-1")
    block = config.to_dict()
    assert block == {
        "base_url": ROUTER,
        "profile": svc.task_session("user-a", "conv-1"),
        "project": svc.project_key("user-a", "proj-1"),
    }
    assert BrowserConfig.from_dict(block) == config
    assert config.token is None and "token" not in block
    no_project = svc.turn_block(on, user_id="user-a", conversation_id="conv-1", project_id=None)
    assert no_project.to_dict() == {"base_url": ROUTER, "profile": svc.task_session("user-a", "conv-1")}


def test_turn_block_needs_the_router_a_person_and_the_switch(web, monkeypatch):
    on, off = SimpleNamespace(browser_enabled=True), SimpleNamespace(browser_enabled=False)
    assert svc.turn_block(on, user_id="u", conversation_id="c", project_id=None) is not None
    assert svc.turn_block(off, user_id="u", conversation_id="c", project_id=None) is None
    # An organization's channel turn has no person behind it.
    assert svc.turn_block(on, user_id=None, conversation_id="c", project_id=None) is None
    monkeypatch.setattr(svc, "router_url", lambda: "")
    assert svc.turn_block(on, user_id="u", conversation_id="c", project_id=None) is None


def test_the_turn_blocks_this_server_writes_are_antons(web):
    """anton's fixture: the router blocks this server writes have exactly the
    fixture's keys, and anton reads each fixture block back as written."""
    examples = _fixture("browser_contract", "browser_config.json").examples
    on = SimpleNamespace(browser_enabled=True)
    ours = {
        "the router block": svc.turn_block(on, user_id="u", conversation_id="c", project_id="p"),
        "the router block without a project": svc.turn_block(on, user_id="u", conversation_id="c", project_id=None),
    }
    for name, config in ours.items():
        assert config.to_dict().keys() == examples[name].body.keys(), name
    for name, example in examples.items():
        if example.kind != "turn_block":
            continue
        read = BrowserConfig.from_dict(example.body)
        assert (read is None) is example.refused, name
        if read is not None:
            assert read.to_dict() == example.body, name


def test_the_desktop_config_is_built_as_antons_constructor_takes_it(app_browser):
    """anton's fixture: the in-app constructor arguments it accepts are the
    shape desktop_browser_config passes, and those it refuses stay refused."""
    examples = _fixture("browser_contract", "browser_config.json").examples
    config = svc.desktop_browser_config(conversation_id="conv-1", project_id="proj-1")
    accepted = examples["the in-app browser"].body
    assert set(config.model_dump(exclude_defaults=True)) == set(accepted)
    for name, example in examples.items():
        if example.kind != "constructor":
            continue
        if example.refused:
            with pytest.raises(ValidationError):
                BrowserConfig(**example.body)
        else:
            assert BrowserConfig(**example.body).in_app, name


# ── the desktop's own browser ──────────────────────────────────────────────


def test_the_harness_gives_a_desktop_turn_the_apps_browser(app_browser):
    from cowork.harnesses.anton_harness.harness import _browser_config

    on = SimpleNamespace(browser_enabled=True)
    config = asyncio.run(_browser_config(on, channel_context=None, conversation_id="conv-1", project_id="proj-1"))
    assert config.in_app
    assert (config.base_url, config.profile, config.project) == (
        APP_BROWSER.url, svc.task_session(None, "conv-1"), svc.project_key(None, "proj-1"),
    )
    assert config.token.get_secret_value() == APP_BROWSER.token.get_secret_value()
    off = SimpleNamespace(browser_enabled=False)
    assert asyncio.run(_browser_config(off, channel_context=None, conversation_id="conv-1", project_id=None)) is None


def test_without_a_decisions_model_no_turn_gets_the_browser(app_browser, web, decisions):
    """Every browser step would go back to the model, too slow to offer: the
    switch stays on, and the browser is off until there is one again."""
    from cowork.harnesses.anton_harness.harness import _browser_config

    decisions.model = None
    on = SimpleNamespace(browser_enabled=True)
    assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id=None)) is None
    assert svc.turn_block(on, user_id="u", conversation_id="c", project_id=None) is None


def test_a_web_turn_asks_the_organizations_own_model_through_this_server(web, decisions, monkeypatch):
    """The pod must not hold the organization's key: its block names this
    server's proxy and the model name, with no key, and anton reads it back."""
    from anton.core.browser import config as anton_config

    if "decisions_proxy" not in anton_config.BrowserConfig.model_fields:
        pytest.skip("the installed anton predates the decisions proxy")
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.handlers.jev_route import JevTarget

    monkeypatch.setenv("COWORK_BROWSER_DECISIONS_URL", "https://cowork.example.test/api/v1/browse/decisions")
    get_app_settings.cache_clear()
    decisions.model = "own"
    decisions.own = JevTarget(url="https://decide.example.test/v1/systemone", api_key="org-own-key")
    on = SimpleNamespace(browser_enabled=True, decision_model="jev-own",
                         decision_model_api_key="org-own-key", decision_model_url=decisions.own.url)
    block = svc.turn_block(on, user_id="u", conversation_id="c", project_id=None).to_dict()
    assert block["decisions_proxy"] == {"url": "https://cowork.example.test/api/v1/browse/decisions", "model": "jev-own"}
    assert "org-own-key" not in json.dumps(block) and "decide.example.test" not in json.dumps(block)
    assert anton_config.BrowserConfig.from_dict(block).decisions_proxy.model == "jev-own"
    # Jev on MindsHub needs no proxy, and nor does a deployment without one.
    decisions.model = "mindshub"
    assert "decisions_proxy" not in svc.turn_block(on, user_id="u", conversation_id="c", project_id=None).to_dict()
    decisions.model = "own"
    monkeypatch.delenv("COWORK_BROWSER_DECISIONS_URL")
    monkeypatch.setenv("COWORK_PUBLIC_BASE_URL", "")
    get_app_settings.cache_clear()
    assert "decisions_proxy" not in svc.turn_block(on, user_id="u", conversation_id="c", project_id=None).to_dict()


def test_the_web_offers_the_browser_only_on_mindshub(web, decisions):
    """The router takes only a MindsHub credential."""
    decisions.mindshub = False
    assert svc.turn_block(SimpleNamespace(browser_enabled=True), user_id="u", conversation_id="c", project_id=None) is None


def test_the_harness_hands_anton_the_organizations_own_decision_model(app_browser, decisions):
    """It picks the browser's steps, in this process only, as the launch key
    stays: the model name is the organization's, or the gate's default."""
    from anton.core.browser import config as anton_config

    if not hasattr(anton_config, "DecisionsTarget"):
        pytest.skip("the installed anton predates the host's own decision model")
    from cowork.handlers.jev_route import JevTarget
    from cowork.harnesses.anton_harness.harness import _browser_config

    decisions.own = JevTarget(url="https://decide.example.test/v1/systemone", api_key="own-key")
    for model, expected in (("jev-own", "jev-own"), ("", "jev")):
        user = SimpleNamespace(browser_enabled=True, decision_model=model)
        config = asyncio.run(_browser_config(user, channel_context=None, conversation_id="c", project_id=None))
        assert (config.decisions.url, config.decisions.api_key.get_secret_value(), config.decisions.model) == (
            "https://decide.example.test/v1/systemone", "own-key", expected,
        )
    decisions.own = None
    config = asyncio.run(_browser_config(user, channel_context=None, conversation_id="c", project_id=None))
    assert config.decisions is None


# ── where the organization's own decision model reaches the browser ───────


class _StandInDecisionsTarget(BaseModel):
    """anton's ``anton.core.browser.config.DecisionsTarget``, for an installed
    anton that predates it, with the same check on the URL. The tests use
    anton's own wherever it has one."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    url: str
    api_key: SecretStr
    model: str = "jev"

    @field_validator("url")
    @classmethod
    def _url_is_http(cls, value: str) -> str:
        from urllib.parse import urlsplit

        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or "@" in url.netloc:
            raise ValueError("the decision endpoint must be an http(s) URL with a host and no user")
        return value


def _anton_takes_own_model(monkeypatch, takes: bool) -> None:
    """The in-process anton takes the host's own decision model, or predates it."""
    from anton.core.browser import config as anton_config

    if not takes:
        monkeypatch.delattr(anton_config, "DecisionsTarget", raising=False)
    elif not hasattr(anton_config, "DecisionsTarget"):
        monkeypatch.setattr(anton_config, "DecisionsTarget", _StandInDecisionsTarget, raising=False)


def _anton_takes_proxy(monkeypatch, takes: bool) -> None:
    """The pod's anton's browser config takes the decisions proxy, or predates it."""
    fields = {k: v for k, v in BrowserConfig.model_fields.items() if k != "decisions_proxy"}
    if takes:
        fields["decisions_proxy"] = None
    monkeypatch.setattr(svc, "BrowserConfig", SimpleNamespace(model_fields=fields))


@pytest.fixture
def proxy_url():
    """Sets this server's decisions proxy URL, or none. Its own patch, undone
    before the cached app settings are dropped, so no later test reads them."""
    from cowork.common.settings.app_settings import get_app_settings

    with pytest.MonkeyPatch.context() as patch:
        def set_url(url: str) -> None:
            if url:
                patch.setenv("COWORK_BROWSER_DECISIONS_URL", url)
            else:
                patch.delenv("COWORK_BROWSER_DECISIONS_URL", raising=False)
            patch.setenv("COWORK_PUBLIC_BASE_URL", "")
            get_app_settings.cache_clear()

        yield set_url
    get_app_settings.cache_clear()


PROXY = "https://cowork.example.test/api/v1/browse/decisions"


def test_on_the_web_the_organizations_model_reaches_the_browser_only_through_the_proxy(web, proxy_url, monkeypatch):
    _anton_takes_own_model(monkeypatch, True)
    for url, takes, reaches in (("", True, False), (PROXY, False, False), (PROXY, True, True)):
        proxy_url(url)
        _anton_takes_proxy(monkeypatch, takes)
        assert svc.own_model_reaches_browser() is reaches, (url, takes)


def test_on_the_desktop_the_organizations_model_reaches_the_browser_only_through_anton(proxy_url, monkeypatch):
    # The proxy is the web's: its URL changes nothing here.
    proxy_url(PROXY)
    _anton_takes_proxy(monkeypatch, True)
    _anton_takes_own_model(monkeypatch, False)
    assert svc.own_model_reaches_browser() is False
    _anton_takes_own_model(monkeypatch, True)
    assert svc.own_model_reaches_browser() is True


@pytest.fixture
def own_model(decisions, monkeypatch):
    """The organization set up its own decision model, and status reads
    jev_route.decisions_model itself, over it."""
    from cowork.handlers import jev_route

    monkeypatch.setattr(jev_route, "decisions_model", real_decisions_model)
    decisions.own = jev_route.JevTarget(url="https://decide.example.test/v1/systemone", api_key="own-key")
    return decisions


def _on_mindshub(*, jev: bool = True):
    from cowork.common.settings.user_settings import UserSettings

    return UserSettings.model_validate({
        "planning_provider": "minds_cloud", "minds_api_key": "k", "jev_decisions_enabled": jev, "browser_enabled": True,
    })


def _status(settings):
    return asyncio.run(svc.browse_status(settings, bearer="jwt-alice", org_id=ORG, user_id="alice"))


def test_web_status_says_own_only_where_the_proxy_reaches_the_browser(router, mindshub, own_model, proxy_url, monkeypatch):
    """Without a proxy URL, or with an anton that predates the proxy, the
    browser asks Jev on MindsHub: status says so, and with Jev off there is
    nothing to pick the steps, so the browser is off. Either way status says
    the organization's model is set up but unused, so Settings doesn't ask
    for one."""
    _anton_takes_proxy(monkeypatch, True)
    proxy_url("")
    answer = _status(_on_mindshub())
    assert (answer.decisions_model, answer.enabled, answer.own_decisions_unused) == ("mindshub", True, True)
    answer = _status(_on_mindshub(jev=False))
    assert (answer.decisions_model, answer.enabled, answer.available, answer.own_decisions_unused) == (
        None, False, True, True,
    )
    proxy_url(PROXY)
    _anton_takes_proxy(monkeypatch, False)
    answer = _status(_on_mindshub())
    assert (answer.decisions_model, answer.own_decisions_unused) == ("mindshub", True)
    _anton_takes_proxy(monkeypatch, True)
    answer = _status(_on_mindshub(jev=False))
    assert (answer.decisions_model, answer.enabled, answer.own_decisions_unused) == ("own", True, False)


def test_desktop_status_says_own_only_where_anton_takes_the_organizations_model(app_browser, own_model, monkeypatch):
    _anton_takes_own_model(monkeypatch, False)
    answer = _status(_on_mindshub())
    assert (answer.decisions_model, answer.own_decisions_unused) == ("mindshub", True)
    answer = _status(_on_mindshub(jev=False))
    assert (answer.decisions_model, answer.enabled, answer.available, answer.own_decisions_unused) == (
        None, False, True, True,
    )
    _anton_takes_own_model(monkeypatch, True)
    answer = _status(_on_mindshub(jev=False))
    assert (answer.decisions_model, answer.enabled, answer.own_decisions_unused) == ("own", True, False)


def test_a_decisions_target_anton_refuses_leaves_the_desktop_browser_running(app_browser, monkeypatch, caplog):
    """A URL stored without its scheme, from before Settings checked it: the
    turn keeps its browser, without the organization's model, and the log
    names neither the URL nor the key."""
    from cowork.handlers.jev_route import JevTarget

    _anton_takes_own_model(monkeypatch, True)
    stored = JevTarget(url="decisions.example.test/v1", api_key="own-key-secret")
    with caplog.at_level(logging.WARNING, logger="cowork.services.browser"):
        config = svc.desktop_browser_config(conversation_id="c", project_id=None, decisions=(stored, "jev-own"))

    assert config is not None and config.in_app
    assert getattr(config, "decisions", None) is None
    [record] = [r for r in caplog.records if r.name == "cowork.services.browser" and r.levelno == logging.WARNING]
    assert record.event == "desktop_browser_decisions_refused"
    logged = json.dumps(record.__dict__, default=str)
    assert "decisions.example.test" not in logged and "own-key-secret" not in logged


def test_desktop_status_says_what_picks_the_steps(client, decisions):
    runtime_credential.set_browser_host(APP_BROWSER)
    # Anyone on a desktop may set it up.
    assert client.put("/api/v1/settings/browser_enabled", json={"value": "true"}).status_code == 200
    answer = client.get("/api/v1/browse/status").json()
    assert (answer["enabled"], answer["decisions_model"], answer["can_set_decisions_model"]) == (True, "mindshub", True)
    # Cleared later: the switch stays on, and the browser reads as off. No
    # model of the organization's own either, so nothing set up goes unused.
    decisions.model = None
    answer = client.get("/api/v1/browse/status").json()
    assert (answer["enabled"], answer["available"], answer["decisions_model"], answer["own_decisions_unused"]) == (
        False, True, None, False,
    )


def test_on_the_web_only_an_organization_admin_may_set_up_a_decisions_model(web_client, router, mindshub):
    from cowork.principal import ORG_MANAGE_ROLE

    member = web_client.get("/api/v1/browse/status", headers=_as("alice")).json()
    assert member["can_set_decisions_model"] is False
    web_client.app.dependency_overrides[get_principal] = lambda: Principal(
        user_id="alice", org_id=ORG, roles=frozenset({ORG_MANAGE_ROLE}),
    )
    admin = web_client.get("/api/v1/browse/status", headers=_as("alice")).json()
    assert admin["can_set_decisions_model"] is True


def test_web_status_is_unavailable_off_mindshub_and_asks_nothing(web_client, nothing_goes_out, decisions):
    decisions.mindshub = False
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert _core(status.json()) == {"enabled": False, "available": False, "status": "unavailable"}
    assert nothing_goes_out == []


def test_a_channel_turn_gets_no_browser(app_browser):
    """A chat anyone can write to must never drive the desktop owner's
    signed-in browser."""
    from cowork.harnesses.anton_harness.harness import _browser_config
    from cowork.harnesses.base import ChannelContext

    on = SimpleNamespace(browser_enabled=True)
    group = ChannelContext("telegram", is_group=True)
    assert asyncio.run(_browser_config(on, channel_context=group, conversation_id="c", project_id=None)) is None
    assert asyncio.run(_browser_config(on, channel_context=ChannelContext("slack"), conversation_id="c", project_id=None)) is None
    assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id=None)).in_app


def test_the_session_builder_passes_the_channel_to_the_browser():
    from cowork.harnesses.anton_harness import harness

    source = inspect.getsource(harness.AntonHarness._build_turn_session)
    assert re.search(r"turn_browser = await _browser_config\(\s*user, channel_context=channel_context,", source)
    assert "browser=turn_browser," in source
    assert "supported_kwargs(ChatSessionConfig, browser=" not in source


def test_only_a_turn_with_a_browser_hears_about_steps_by_position():
    """The prompt line about clicking by position joins only a turn that has a
    browser, and says nothing about approval: the browser tool's own prompt
    says which steps wait for the person."""
    from cowork.harnesses.anton_harness import harness

    source = inspect.getsource(harness.AntonHarness._build_turn_session)
    line = re.search(r'" In the browser, when no listed element fits.*?"\s*"[^"]*"\s*if turn_browser is not None', source, re.S)
    assert line, "the position line must depend on the turn's browser"
    assert "approval" not in line.group(0)


def test_in_process_turn_never_gets_the_router(monkeypatch, nothing_goes_out):
    """Even with a router configured and the desktop's MindsHub credential
    held, an in-process turn gets only the app's own browser, and here there
    is none."""
    from cowork.harnesses.anton_harness.harness import _browser_config

    monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", ROUTER)
    monkeypatch.setattr(runtime_credential, "get_minds_credential", lambda: "desktop-jwt")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    try:
        on = SimpleNamespace(browser_enabled=True)
        assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id="p")) is None
    finally:
        get_app_settings.cache_clear()
    assert nothing_goes_out == []


def test_personal_deployment_holds_no_browser_host(personal_deployment):
    """A local deployment that trusts identity headers serves several people,
    so no one's browser may answer another's turn."""
    from cowork.harnesses.anton_harness.harness import _browser_config

    runtime_credential.set_browser_host(APP_BROWSER)
    assert runtime_credential.get_browser_host() is None
    on = SimpleNamespace(browser_enabled=True)
    assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id=None)) is None


def test_a_refused_app_browser_is_logged_without_its_key(caplog):
    """anton refuses an address it doesn't take; the turn runs without a
    browser, and the log line carries neither the address check's input nor
    the key."""
    secret = "s" * 43
    runtime_credential.set_browser_host(LocalBrowserHost(url="http://localhost:51234", token=SecretStr(secret)))
    with caplog.at_level(logging.WARNING, logger=svc.__name__):
        assert svc.desktop_browser_config(conversation_id="c", project_id=None) is None
    records = [r for r in caplog.records if r.name == svc.__name__ and r.levelno == logging.WARNING]
    assert len(records) == 1
    assert secret not in records[0].getMessage() and secret not in str(records[0].__dict__)


def test_desktop_routes_answer_from_the_app_and_call_nothing(client, nothing_goes_out):
    status = client.get("/api/v1/browse/status")
    assert status.status_code == 200
    assert _core(status.json()) == {"enabled": False, "available": False, "status": "unavailable"}
    svc.BrowseStatus.model_validate(status.json())

    runtime_credential.set_browser_host(APP_BROWSER)
    assert _core(client.get("/api/v1/browse/status").json()) == {"enabled": False, "available": True, "status": "running"}
    # Available whatever the switch says, so the person can turn it on and off.
    assert client.put("/api/v1/settings/browser_enabled", json={"value": "true"}).status_code == 200
    assert _core(client.get("/api/v1/browse/status").json()) == {"enabled": True, "available": True, "status": "running"}
    assert client.put("/api/v1/settings/browser_enabled", json={"value": "false"}).status_code == 200
    assert _core(client.get("/api/v1/browse/status").json()) == {"enabled": False, "available": True, "status": "running"}

    assert _detail(client.post("/api/v1/browse/provision")) == (409, "unavailable")
    assert _detail(client.post("/api/v1/browse/embed", json={"session_id": "t-abc"})) == (409, "unavailable")
    approve = client.post("/api/v1/browse/approve", json={"session_id": "t-abc", "pending_id": "pa-1"})
    assert _detail(approve) == (409, "unavailable")
    decline = client.post("/api/v1/browse/decline", json={"session_id": "t-abc", "pending_id": "pa-1"})
    assert _detail(decline) == (409, "unavailable")
    assert _detail(client.delete("/api/v1/browse/logins")) == (409, "unavailable")
    assert nothing_goes_out == []


# ── the web: no router ─────────────────────────────────────────────────────


def test_without_a_router_nothing_goes_out(web_client, monkeypatch, nothing_goes_out):
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", "")
    get_app_settings.cache_clear()

    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert status.status_code == 200
    assert _core(status.json()) == {"enabled": False, "available": False, "status": "unavailable"}
    assert _detail(web_client.post("/api/v1/browse/provision", headers=_as("alice"))) == (409, "unavailable")
    embed = web_client.post("/api/v1/browse/embed", headers=_as("alice"), json={"session_id": "t-abc"})
    assert _detail(embed) == (409, "unavailable")
    approve = web_client.post(
        "/api/v1/browse/approve", headers=_as("alice"), json={"session_id": "t-abc", "pending_id": "pa-1"},
    )
    assert _detail(approve) == (409, "unavailable")
    assert _detail(web_client.delete("/api/v1/browse/logins", headers=_as("alice"))) == (409, "unavailable")
    on = SimpleNamespace(browser_enabled=True)
    assert svc.turn_block(on, user_id="alice", conversation_id="c", project_id=None) is None
    assert nothing_goes_out == []


# ── the web: the router ────────────────────────────────────────────────────


def test_web_status_reads_the_entitlement_then_the_router(web_client, router, mindshub):
    router.status = "running"
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert status.status_code == 200
    assert _core(status.json()) == {"enabled": False, "available": True, "status": "running"}
    assert mindshub.asked == [("/entitlements/me/", "jwt-alice")]
    assert router.calls == [Call("GET", f"{ROUTER}/_status", None, "Bearer jwt-alice")]


def test_an_account_without_the_browser_is_not_entitled(web_client, router, mindshub):
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    mindshub.replies["jwt-alice"] = MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(False))
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert status.status_code == 200
    assert _core(status.json()) == {"enabled": False, "available": False, "status": "not_entitled"}
    # Read from MindsHub: the router isn't asked.
    assert router.calls == []

    # MindsHub refusing the read grants nothing, the browser included.
    mindshub.replies["jwt-bob"] = MindsHubReply(MindsHubAnswer.REFUSED, status_code=403)
    assert web_client.get("/api/v1/browse/status", headers=_as("bob")).json()["status"] == "not_entitled"
    assert router.calls == []


@pytest.mark.parametrize("answer", ["unreachable", "throttled", "no block"])
def test_web_status_fails_closed_when_mindshub_cant_say(web_client, router, mindshub, answer):
    """An outage, a throttle with nothing known, or an answer without the
    block is "couldn't check", never "not entitled" and never a browser."""
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    mindshub.replies["jwt-alice"] = {
        "unreachable": MindsHubReply(MindsHubAnswer.UNREACHABLE),
        "throttled": MindsHubReply(MindsHubAnswer.THROTTLED, status_code=429),
        "no block": MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(True, block=False)),
    }[answer]
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert _detail(status) == (503, "mindshub_unreachable")
    assert router.calls == []


def test_web_status_answers_match_the_model(web_client, router, mindshub):
    answer = web_client.get("/api/v1/browse/status", headers=_as("alice")).json()
    assert set(answer) == {
        "enabled", "available", "status", "decisions_model", "can_set_decisions_model", "own_decisions_unused",
    }
    svc.BrowseStatus.model_validate(answer)


def test_an_organization_that_turned_the_browser_off_reads_as_organization_off(web_client, router, mindshub):
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    mindshub.replies["jwt-alice"] = MindsHubReply(
        MindsHubAnswer.OK, payload=_entitlements(False, reason="organization_off"),
    )
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert _core(status.json()) == {"enabled": False, "available": False, "status": "organization_off"}
    assert router.calls == []


@pytest.fixture
def signed_in_desktop(monkeypatch, mindshub):
    """A desktop the person signed in to MindsHub on, with its browser registered."""
    monkeypatch.setattr(svc, "get_minds_credential", lambda: "desktop-jwt")
    monkeypatch.setattr(svc, "_desktop_organization_off", False)
    runtime_credential.set_browser_host(APP_BROWSER)
    return mindshub


def test_a_signed_in_desktop_whose_organization_turned_the_browser_off_gets_none(client, signed_in_desktop):
    from cowork.harnesses.anton_harness.harness import _browser_config
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    on = SimpleNamespace(browser_enabled=True)
    assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id=None)).in_app
    signed_in_desktop.replies["desktop-jwt"] = MindsHubReply(
        MindsHubAnswer.OK, payload=_entitlements(False, reason="organization_off"),
    )
    from cowork.services import mindshub_teams

    mindshub_teams.reset_caches_for_tests()
    assert asyncio.run(_browser_config(on, channel_context=None, conversation_id="c", project_id=None)) is None
    assert _core(client.get("/api/v1/browse/status").json()) == {"enabled": False, "available": False, "status": "organization_off"}
    assert ("/entitlements/me/", "desktop-jwt") in signed_in_desktop.asked


def test_a_desktop_ignores_not_permitted_and_keeps_its_answer_through_an_outage(client, signed_in_desktop):
    """browser.use governs the web only, and the desktop browser runs on the
    person's own machine, so an outage leaves the last answer standing."""
    from cowork.services import mindshub_teams
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    replies = signed_in_desktop.replies
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(False))
    assert client.get("/api/v1/browse/status").json()["status"] == "running"
    mindshub_teams.reset_caches_for_tests()
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(False, reason="organization_off"))
    assert client.get("/api/v1/browse/status").json()["status"] == "organization_off"
    mindshub_teams.reset_caches_for_tests()
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.UNREACHABLE)
    assert client.get("/api/v1/browse/status").json()["status"] == "organization_off"


def test_a_desktop_on_the_persons_own_model_key_asks_mindshub_nothing(client, nothing_goes_out, monkeypatch):
    monkeypatch.setattr(svc, "get_minds_credential", lambda: None)
    runtime_credential.set_browser_host(APP_BROWSER)
    assert _core(client.get("/api/v1/browse/status").json()) == {"enabled": False, "available": True, "status": "running"}
    assert nothing_goes_out == []


def test_org_routes_go_out_as_each_caller(web_client, router, mindshub):
    for user in ("alice", "bob"):
        assert web_client.get("/api/v1/browse/status", headers=_as(user)).status_code == 200
        assert web_client.post("/api/v1/browse/provision", headers=_as(user)).status_code == 200
        sid = svc.task_session(user, "conv-1")
        assert web_client.post("/api/v1/browse/embed", headers=_as(user), json={"session_id": sid}).status_code == 200
        assert web_client.delete("/api/v1/browse/logins", headers=_as(user)).status_code == 200
    alice = [c for c in router.calls if c.auth == "Bearer jwt-alice"]
    bob = [c for c in router.calls if c.auth == "Bearer jwt-bob"]
    assert len(alice) == len(bob) == 4 and len(router.calls) == 8
    assert {c.auth for c in router.calls} == {"Bearer jwt-alice", "Bearer jwt-bob"}
    assert ("/entitlements/me/", "jwt-bob") in mindshub.asked


def test_org_routes_refuse_no_principal(web_client, router, mindshub):
    for method, path, body in [
        ("GET", "/api/v1/browse/status", None), ("POST", "/api/v1/browse/provision", None),
        ("POST", "/api/v1/browse/embed", {"session_id": "t-abc"}),
        ("POST", "/api/v1/browse/approve", {"session_id": "t-abc", "pending_id": "pa-1"}),
        ("POST", "/api/v1/browse/decline", {"session_id": "t-abc", "pending_id": "pa-1"}),
        ("DELETE", "/api/v1/browse/logins", None),
    ]:
        response = web_client.request(method, path, json=body, headers={"Authorization": "Bearer jwt-x"})
        assert response.status_code == 401, path
    assert router.calls == [] and mindshub.asked == []


def test_provision_hands_the_router_the_prefs_and_answers_like_status(web_client, router, mindshub):
    # No body still works.
    plain = web_client.post("/api/v1/browse/provision", headers=_as("alice"))
    assert plain.status_code == 200
    assert _core(plain.json()) == {"enabled": False, "available": True, "status": "starting"}
    hinted = web_client.post(
        "/api/v1/browse/provision", headers=_as("alice"),
        json={"timezone": "America/Los_Angeles", "locale": "en-US"},
    )
    assert hinted.status_code == 200
    bad = web_client.post(
        "/api/v1/browse/provision", headers=_as("alice"), json={"timezone": "Europe/London; rm", "locale": "en_GB<x>"},
    )
    assert bad.status_code == 200
    warms = [c.body for c in router.calls if c.url.endswith("/_warm")]
    assert warms == [
        {"prefs": {}},
        {"prefs": {"timezone": "America/Los_Angeles", "locale": "en-US"}},
        {"prefs": {}},
    ]


def test_approve_forwards_the_persons_approval_as_them(web_client, router, mindshub):
    sid = svc.task_session("alice", "conv-1")
    approved = web_client.post(
        "/api/v1/browse/approve", headers=_as("alice"), json={"session_id": sid, "pending_id": "pa-7"},
    )
    assert approved.status_code == 200
    assert approved.json() == {"pending_id": "pa-7", "approved": True, "expires_at": 1999999300}
    assert router.calls == [Call("POST", f"{ROUTER}/sessions/{sid}/approve", {"pending_id": "pa-7"}, "Bearer jwt-alice")]


def test_approve_checks_its_ids_and_the_routers_answer(web_client, router, mindshub):
    for body in (
        {"session_id": "t-abc"}, {"session_id": "../x", "pending_id": "pa-1"}, {"session_id": "t-abc", "pending_id": "p a"},
        {"session_id": "t-abc", "pending_id": "pa-1", "step": "click"},
    ):
        assert web_client.post("/api/v1/browse/approve", headers=_as("alice"), json=body).status_code == 422, body
    assert router.calls == []
    router.answers["/browser/sessions/t-abc/approve"] = httpx.Response(
        200, json={"pending_id": "pa-other", "approved": True, "expires_at": 1999999300},
    )
    other = web_client.post("/api/v1/browse/approve", headers=_as("alice"), json={"session_id": "t-abc", "pending_id": "pa-1"})
    assert _detail(other) == (svc.UPSTREAM_FAILED, "bad_response")


@pytest.mark.parametrize(("name", "code"), [
    ("an approval that lapsed", "approval_lapsed"),
    ("an approval with no running browser", "session_not_found"),
])
def test_a_refused_approval_reaches_the_app_with_its_code(web_client, router, mindshub, name, code):
    refusal = _fixture("browser_router", "error_answer.json").examples[name]
    router.answers["/browser/sessions/t-abc/approve"] = httpx.Response(refusal.status, json=refusal.body)
    answer = web_client.post("/api/v1/browse/approve", headers=_as("alice"), json={"session_id": "t-abc", "pending_id": "pa-1"})
    assert _detail(answer) == (svc.UPSTREAM_FAILED, code)


def test_decline_forwards_the_persons_skip_as_them(web_client, router, mindshub):
    """The card's Skip drops the held step and any approval that landed late,
    so a resend after it stays held."""
    sid = "t-abc"
    declined = web_client.post(
        "/api/v1/browse/decline", headers=_as("alice"), json={"session_id": sid, "pending_id": "pa-7"},
    )
    assert declined.status_code == 200 and declined.json() == {"declined": True}
    assert router.calls == [Call("POST", f"{ROUTER}/sessions/{sid}/decline", {"pending_id": "pa-7"}, "Bearer jwt-alice")]


def test_decline_checks_its_ids_and_passes_the_routers_refusal_on(web_client, router, mindshub):
    for body in ({"session_id": "t abc", "pending_id": "pa-1"}, {"session_id": "t-abc", "pending_id": "pa:1"},
                 {"session_id": "t-abc", "pending_id": "pa-1", "extra": 1}):
        assert web_client.post("/api/v1/browse/decline", headers=_as("alice"), json=body).status_code == 422, body
    assert router.calls == []
    refusal = _fixture("browser_router", "error_answer.json").examples["an approval with no running browser"]
    router.answers["/browser/sessions/t-abc/decline"] = httpx.Response(refusal.status, json=refusal.body)
    answer = web_client.post("/api/v1/browse/decline", headers=_as("alice"), json={"session_id": "t-abc", "pending_id": "pa-1"})
    assert _detail(answer) == (svc.UPSTREAM_FAILED, "session_not_found")


def test_sign_out_of_every_site_goes_to_the_router_as_the_person(web_client, router, mindshub):
    cleared = web_client.delete("/api/v1/browse/logins", headers=_as("alice"))
    assert cleared.status_code == 200 and cleared.json() == {"cleared": True}
    assert router.calls == [Call("DELETE", f"{ROUTER}/logins", None, "Bearer jwt-alice")]
    # No entitlement read: a person whose browser is off can still clear their sign-ins.
    assert mindshub.asked == []


def test_embed_passes_the_routers_link_through(web_client, router, mindshub):
    sid = svc.task_session("alice", "conv-1")
    embed = web_client.post("/api/v1/browse/embed", headers=_as("alice"), json={"session_id": sid})
    assert embed.status_code == 200
    assert embed.json() == _embed_answer(sid)
    assert router.calls == [Call("POST", f"{ROUTER}/_embed", {"session_id": sid}, "Bearer jwt-alice")]


def test_embed_needs_a_session_id(web_client, router, mindshub):
    for body in ({}, {"session_id": ""}, {"session_id": "../escape"}, {"session_id": "x" * 65}):
        assert web_client.post("/api/v1/browse/embed", headers=_as("alice"), json=body).status_code == 422, body
    assert router.calls == []


@pytest.mark.parametrize("view_url", [
    f"{ROUTER}/sessions/t-other/view?et=tok",
    "https://elsewhere.example/browser/sessions/t-mine/view?et=tok",
    f"{ROUTER}/sessions/t-mine/view/extra?et=tok",
    "https://user@app.example.test/browser/sessions/t-mine/view?et=tok",
    f"{ROUTER}/sessions/t-mine/view#frag?et=tok",
])
def test_embed_refuses_a_viewer_url_outside_the_session(web_client, router, mindshub, view_url):
    router.answers["/browser/_embed"] = httpx.Response(200, json=_embed_answer("t-mine", view_url=view_url))
    embed = web_client.post("/api/v1/browse/embed", headers=_as("alice"), json={"session_id": "t-mine"})
    assert _detail(embed) == (svc.UPSTREAM_FAILED, "bad_response")


def test_embed_refuses_an_answer_for_another_session(web_client, router, mindshub):
    router.answers["/browser/_embed"] = httpx.Response(200, json=_embed_answer("t-other"))
    embed = web_client.post("/api/v1/browse/embed", headers=_as("alice"), json={"session_id": "t-mine"})
    assert _detail(embed) == (svc.UPSTREAM_FAILED, "bad_response")


def test_router_refusals_reach_the_app_with_their_code(web_client, router, mindshub):
    """Each of the router's error bodies (the fixture) reaches the app with
    the router's code: 401, 403 and 429 with their own status, anything else
    as 424."""
    for name, example in _fixture("browser_router", "error_answer.json").examples.items():
        router.answers["/browser/logins"] = httpx.Response(example.status, json=example.body)
        answer = web_client.delete("/api/v1/browse/logins", headers=_as("alice"))
        expected = example.status if example.status in (401, 403, 429) else svc.UPSTREAM_FAILED
        assert answer.status_code == expected, name
        assert answer.json()["detail"] == {"code": example.body["error"], "message": example.body["message"]}, name


def test_an_unreachable_router_says_so_without_the_credential(web_client, monkeypatch, web, mindshub):
    def down(request, body):
        raise httpx.ConnectError("refused")

    _route_router(monkeypatch, down)
    answer = web_client.delete("/api/v1/browse/logins", headers=_as("alice"))
    assert _detail(answer) == (svc.UPSTREAM_FAILED, "browser_unreachable")
    assert "jwt-alice" not in answer.text


def test_no_credential_is_a_401_before_any_call(router):
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.clear_logins(""))
    assert (err.value.status, err.value.detail.code) == (401, "unauthorized")
    assert router.calls == []


# ── a web turn's scope ─────────────────────────────────────────────────────


def test_scope_turn_binds_the_turn_key_to_its_session_and_project(router):
    """The fixture's _scope bodies are what scope_turn sends, with the turn's
    own key as the bearer."""
    examples = _fixture("browser_router", "scope_request.json").examples
    on = SimpleNamespace(browser_enabled=True)
    in_project = svc.turn_block(on, user_id="u", conversation_id="c", project_id="p")
    no_project = svc.turn_block(on, user_id="u", conversation_id="c", project_id=None)

    asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=in_project))
    asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=no_project))
    asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=None))
    sent = [c.body for c in router.calls]
    assert [c.url for c in router.calls] == [f"{ROUTER}/_scope"] * 3
    assert {c.auth for c in router.calls} == {"Bearer mdb_turnkey"}
    # Only this server holds the scope key, and every scope carries it.
    assert {c.scope_key for c in router.calls} == {SCOPE_KEY}
    assert sent == [
        {"session_id": in_project.profile, "project": in_project.project}, {"session_id": no_project.profile},
        {"session_id": None},
    ]
    assert sent[0].keys() == examples["a turn in a project"].body.keys()
    assert sent[1].keys() == examples["a turn with no project"].body.keys()
    assert sent[2] == examples["a turn with no browser"].body


def test_a_refused_scope_raises_with_the_routers_code(router):
    conflict = _fixture("browser_router", "error_answer.json").examples["a key scoped to another session"]
    router.answers["/browser/_scope"] = httpx.Response(conflict.status, json=conflict.body)
    config = svc.turn_block(SimpleNamespace(browser_enabled=True), user_id="u", conversation_id="c", project_id=None)
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=config))
    assert (err.value.status, err.value.detail.code) == (svc.UPSTREAM_FAILED, "scope_conflict")


def test_the_scope_key_is_read_on_every_scope_so_a_late_or_rotated_secret_lands(router, scope_key_file):
    """The kubelet fills the mounted Secret after a pod that started first, and
    rewrites it on rotation; neither needs a restart."""
    scope_key_file.unlink()
    asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=None))
    scope_key_file.write_text("rotated-" + "r" * 32)
    asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=None))
    # No file: no header at all, and the router answers scope_key_invalid.
    assert [c.scope_key for c in router.calls] == ["", "rotated-" + "r" * 32]


def test_the_scope_key_never_reaches_a_log_line(router, caplog):
    invalid = _fixture("browser_router", "error_answer.json").examples["a scope without the scope key"]
    router.answers["/browser/_scope"] = httpx.Response(invalid.status, json=invalid.body)
    with caplog.at_level(logging.DEBUG), pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.scope_turn(turn_key="mdb_turnkey", router=ROUTER, config=None))
    assert (err.value.status, err.value.detail.code) == (403, "scope_key_invalid")
    assert SCOPE_KEY not in caplog.text and "mdb_turnkey" not in caplog.text


# ── the router contract fixtures ───────────────────────────────────────────


def test_router_contract_fixtures(router):
    """Each request file is what this server sends, built with its own code,
    and each answer file parses into the model it is read with."""
    warm = _fixture("browser_router", "warm_request.json").examples
    asyncio.run(svc.warm("jwt", svc.router_prefs(timezone="America/Los_Angeles", locale="en-US")))
    asyncio.run(svc.warm("jwt", svc.router_prefs()))
    assert [c.body for c in router.calls] == [warm["time zone and language"].body, warm["nothing to change"].body]

    approve = _fixture("browser_router", "approve.json").examples
    asyncio.run(svc.approve("jwt", svc.ApproveRequest(session_id="t-abc", pending_id="pa-0123456789abcdef01234567")))
    assert router.calls[-1].body == approve["approve a held step"].body
    svc.BrowserApproval.model_validate(approve["approved"].body)
    decline = _fixture("browser_router", "decline.json").examples
    asyncio.run(svc.decline("jwt", svc.DeclineRequest(session_id="t-abc", pending_id="pa-0123456789abcdef01234567")))
    assert router.calls[-1].body == decline["decline a held step"].body
    svc.BrowserDecline.model_validate(decline["declined"].body)
    for example in _fixture("browser_router", "logins_answer.json").examples.values():
        svc.LoginsCleared.model_validate(example.body)
    for example in _fixture("browser_router", "status_answer.json").examples.values():
        svc.RouterStatus.model_validate(example.body)
    for example in _fixture("browser_router", "embed_answer.json").examples.values():
        parsed = svc.BrowserEmbed.model_validate(example.body)
        assert parsed.model_dump() == example.body
        router.answers["/browser/_embed"] = httpx.Response(200, json=example.body)
        assert asyncio.run(svc.embed("jwt", example.body["session_id"])) == parsed


# ── the event ──────────────────────────────────────────────────────────────


def _events(stream_events):
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def gen():
        for e in stream_events:
            yield e

    async def collect():
        return [chunk async for chunk in format_responses_stream(gen(), model="m")]

    return asyncio.run(collect())


def test_formatter_persists_only_the_session_id():
    from anton.core.llm.provider import StreamBrowserSession

    chunks = _events([StreamBrowserSession(session_id="t-abc")])
    opened = [c for c in chunks if c.startswith("event: response.browser_session_opened")]
    assert len(opened) == 1
    data = json.loads(opened[0].split("data: ", 1)[1])
    # at_ms is the time stamp every event carries.
    assert set(data) == {"type", "sequence_number", "session_id", "at_ms"}
    assert data["session_id"] == "t-abc"
    assert "et=" not in opened[0] and "view_url" not in opened[0]


def test_a_pod_browser_step_carries_only_the_session_id():
    from anton.core.llm.provider import StreamBrowserSession

    from cowork.turnqueue.producer import step_stream_events

    sid = "t-0123456789abcdef01234567"
    step = {"step": "browser", "session_id": sid, "view_url": f"{ROUTER}/sessions/{sid}/view?et=t", "expires_at": 7}
    assert step_stream_events(step) == [StreamBrowserSession(session_id=sid)]
    for bad in ("", "../x", "has space", "x" * 65, None, 7):
        assert step_stream_events({"step": "browser", "session_id": bad}) == [], bad
    assert step_stream_events({"step": "browser"}) == []


def test_the_cloud_turns_browser_line_becomes_the_event():
    """anton's fixture: the line its cloud turn writes, as the turn queue
    relays it (the line's kind becomes the step), is one session event."""
    from anton.core.llm.provider import StreamBrowserSession

    from cowork.turnqueue.producer import step_stream_events

    for example in _fixture("browser_contract", "session_event.json").examples.values():
        line = dict(example.body)
        step = {"step": line.pop("kind"), **line}
        assert step_stream_events(step) == [StreamBrowserSession(session_id=example.body["session_id"])]


def test_a_mindshub_401_asks_the_person_to_sign_in_again(web_client, router, mindshub):
    """A credential MindsHub refused says nothing about the browser."""
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    mindshub.replies["jwt-alice"] = MindsHubReply(MindsHubAnswer.REFUSED, status_code=401)
    status = web_client.get("/api/v1/browse/status", headers=_as("alice"))
    assert _detail(status) == (401, "unauthorized")
    assert router.calls == []


def test_auths_outage_answer_is_kept_only_as_long_as_a_failure(mindshub):
    """auth answers no permissions and no browser block when it can't read
    membership. Kept the good answer's 60 s, /browse/status would stay 503
    for a minute after auth recovers."""
    from cowork.services import mindshub_teams
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    mindshub.replies["jwt-alice"] = MindsHubReply(MindsHubAnswer.OK, payload={"permissions": [], "browser": None})
    first = asyncio.run(mindshub_teams.fetch_mindshub_permissions(bearer_token="jwt-alice", org_id="o", user_id="u"))
    assert first.is_outage_answer()
    key = mindshub_teams.CallerCacheKey.for_caller(org_id="o", user_id="u", bearer_token="jwt-alice")
    cache = mindshub_teams._permissions_cache
    entry = cache._entries[key]
    assert entry.fresh_until - entry.stored_at == pytest.approx(cache.ttl_fail_s)


def test_a_desktop_keeps_its_switch_answer_when_mindshub_refuses_the_credential(client, signed_in_desktop, monkeypatch):
    from cowork.services import mindshub_teams
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    replies = signed_in_desktop.replies
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(False, reason="organization_off"))
    assert asyncio.run(svc.desktop_organization_off()) is True
    mindshub_teams.reset_caches_for_tests()
    logged = []
    monkeypatch.setattr(svc.logger, "warning", lambda msg, *a, **kw: logged.append(kw.get("extra", {}).get("event")))
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.REFUSED, status_code=401)
    assert asyncio.run(svc.desktop_organization_off()) is True
    replies["desktop-jwt"] = MindsHubReply(MindsHubAnswer.UNREACHABLE)
    mindshub_teams.reset_caches_for_tests()
    assert asyncio.run(svc.desktop_organization_off()) is True
    assert logged == ["desktop_browser_switch_unread", "desktop_browser_switch_unread"]


def test_handing_over_the_desktop_browser_reads_the_organizations_switch(signed_in_desktop):
    from fastapi.testclient import TestClient

    from cowork.server import create_app
    from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

    signed_in_desktop.replies["desktop-jwt"] = MindsHubReply(
        MindsHubAnswer.OK, payload=_entitlements(False, reason="organization_off"),
    )
    # The app hands its browser over from the same machine.
    loopback = TestClient(create_app(), client=("127.0.0.1", 40000))
    answer = loopback.put("/api/v1/runtime-credential/browser", json={"url": APP_BROWSER.url, "token": APP_BROWSER.token.get_secret_value()})
    assert answer.status_code == 200
    assert svc._desktop_organization_off is True
