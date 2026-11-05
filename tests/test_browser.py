"""The user's MindsHub browser (ENG-3298): provisioning and status as the
caller, the viewer embed URL, the per-turn block anton reads, and the
`response.browser_session_opened` event on both the desktop and web paths."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.principal import HEADER_HUB_CREDENTIAL, get_principal
from cowork.services import browser as svc

ENDPOINT = "https://br-ab12cd34.4nton.ai"


@pytest.fixture
def mindshub(monkeypatch):
    """Fake MindsHub: the /instance lambda and the browser worker's /_embed."""
    calls: list[tuple[str, str, dict | None, str]] = []
    state = {"provisioned": False, "status": "running"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, str(request.url), body, request.headers.get("authorization", "")))
        if request.url.path == "/instance" and request.method == "GET":
            if not state["provisioned"]:
                return httpx.Response(200, json={"status": "none", "provisioned": False})
            return httpx.Response(200, json={"status": state["status"], "provisioned": True, "endpoint": ENDPOINT})
        if request.url.path == "/instance" and request.method == "POST":
            state["provisioned"] = True
            return httpx.Response(200, json={"status": "provisioning", "endpoint": ENDPOINT})
        if request.url.path == "/_embed":
            return httpx.Response(200, json={"view_url": f"{ENDPOINT}/sessions/{body['session_id']}/view?et=tok", "expires_at": 1999999999})
        return httpx.Response(404)

    real = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(svc, "instance_api_base", lambda: "https://api.dev.mindshub.ai")
    return SimpleNamespace(calls=calls, state=state)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: None
    yield TestClient(app)
    # Leave the global settings row as we found it.
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_engine
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.services.settings import SettingService
    from sqlmodel import Session

    with Session(get_engine(get_app_settings().database.uri)) as session:
        service = SettingService(session, LOCAL_SCOPE)
        for key in ("browser_url", "browser_enabled", "browser_mode", "browser_service_url", "browser_service_api_key"):
            try:
                service.delete_setting(key)
            except Exception:
                pass


HUB = {HEADER_HUB_CREDENTIAL: "Bearer user-jwt"}


# ── service ────────────────────────────────────────────────────────────────


def test_instance_api_base_follows_the_deployment(monkeypatch):
    import cowork.common.settings.app_settings as app_settings

    for host, expected in [
        ("https://api.mindshub.ai", "https://4nton.ai"),
        ("https://api.staging.mindshub.ai", "https://api.staging.mindshub.ai"),
        ("https://api-pr-cowork-12.dev.mindshub.ai", "https://api-pr-cowork-12.dev.mindshub.ai"),
    ]:
        monkeypatch.setattr(app_settings, "default_turn_minds_api_host", lambda h=host: h)
        assert svc.instance_api_base() == expected


def test_only_hosted_browser_instances_are_accepted():
    assert svc.valid_endpoint(ENDPOINT)
    assert svc.valid_endpoint("https://br-ab12cd34-alpha.dev.mindshub.ai/")
    for bad in ["", "http://br-x.4nton.ai", "https://hm-x.4nton.ai", "https://br-x.evil.com", "https://br-x.4nton.ai.evil.com"]:
        assert not svc.valid_endpoint(bad), bad


def test_turn_block_needs_the_toggle_and_a_valid_instance():
    on = SimpleNamespace(browser_enabled=True, browser_url=ENDPOINT + "/")
    assert svc.turn_block(on) == {"base_url": ENDPOINT, "profile": "main"}
    assert svc.turn_block(SimpleNamespace(browser_enabled=False, browser_url=ENDPOINT)) is None
    assert svc.turn_block(SimpleNamespace(browser_enabled=True, browser_url="https://elsewhere.com")) is None
    config = svc.anton_browser_config(on)
    assert (config.base_url, config.profile) == (ENDPOINT, "main")


def test_calls_go_out_as_the_caller(mindshub):
    view = asyncio.run(svc.provision("user-jwt"))
    assert view == {"provisioned": True, "status": "provisioning", "endpoint": ENDPOINT}
    method, url, body, auth = mindshub.calls[-1]
    assert (method, url, body, auth) == ("POST", "https://api.dev.mindshub.ai/instance", {"agent": "browser"}, "Bearer user-jwt")


def test_no_credential_is_a_401_before_any_call(mindshub):
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.fetch_status(""))
    assert err.value.status == 401
    assert mindshub.calls == []


def test_embed_rejects_a_foreign_viewer_url(monkeypatch):
    async def lying(*args, **kwargs):
        return {"view_url": "https://evil.example/x", "expires_at": 1}

    monkeypatch.setattr(svc, "_call", lying)
    with pytest.raises(svc.BrowserServiceError):
        asyncio.run(svc.embed("jwt", ENDPOINT))
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed("jwt", ""))
    assert err.value.status == 409


# ── routes ─────────────────────────────────────────────────────────────────


def test_status_before_provisioning(client, mindshub):
    res = client.get("/api/v1/browse/status", headers=HUB)
    assert res.status_code == 200
    assert res.json() == {
        "provisioned": False, "status": "none", "endpoint": "", "mode": "mindshub", "enabled": True, "available": False,
    }


def test_provision_remembers_the_instance_and_embed_uses_it(client, mindshub):
    res = client.post("/api/v1/browse/provision", headers=HUB)
    assert res.status_code == 200 and res.json()["endpoint"] == ENDPOINT

    from cowork.common.settings.user_settings import get_user_settings

    assert get_user_settings().browser_url == ENDPOINT

    client.put("/api/v1/settings/browser_enabled", json={"value": "true"})
    status = client.get("/api/v1/browse/status", headers=HUB).json()
    assert status["enabled"] is True and status["available"] is True

    embed = client.post("/api/v1/browse/embed", headers=HUB, json={"session_id": "main"})
    assert embed.status_code == 200
    assert embed.json()["view_url"] == f"{ENDPOINT}/sessions/main/view?et=tok"
    assert mindshub.calls[-1][1] == f"{ENDPOINT}/_embed"
    assert mindshub.calls[-1][3] == "Bearer user-jwt"


def test_embed_before_provisioning_is_a_409(client, mindshub):
    assert client.post("/api/v1/browse/embed", headers=HUB, json={}).status_code == 409


def test_mindshub_failures_say_what_failed(monkeypatch):
    def broken(request):
        return httpx.Response(500, json={"message": "Internal server error"})

    real = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(broken), **kw))
    monkeypatch.setattr(svc, "instance_api_base", lambda: "https://api-pr-x.dev.mindshub.ai")
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.fetch_status("jwt"))
    # 4xx, not 502: the ingress replaces 5xx bodies and the reason would be lost.
    assert err.value.status == 424
    assert err.value.detail == "api-pr-x.dev.mindshub.ai returned HTTP 500: Internal server error"


def test_mindshub_refusals_pass_through(client, monkeypatch):
    async def refused(*args, **kwargs):
        raise svc.BrowserServiceError(403, "Your plan doesn't include agents.")

    monkeypatch.setattr(svc, "_call", refused)
    res = client.post("/api/v1/browse/provision", headers=HUB)
    assert res.status_code == 403
    assert res.json()["detail"] == "Your plan doesn't include agents."


# ── standalone mode ────────────────────────────────────────────────────────

SERVICE = "https://browser.example.com"
SERVICE_KEY = "svc-secret-token-123"


def _standalone(**overrides):
    from pydantic import SecretStr

    values = dict(
        browser_enabled=True, browser_url="", browser_mode="standalone",
        browser_service_url=SERVICE + "/", browser_service_api_key=SecretStr(SERVICE_KEY),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def service(monkeypatch):
    """Fake standalone browser service; MindsHub's /instance must never be hit."""
    calls: list[tuple[str, str, dict | None, str]] = []
    state = {"health": httpx.Response(200, json={"ok": True, "sessions": 0, "max_sessions": 4}), "embed_url": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, str(request.url), body, request.headers.get("authorization", "")))
        if request.url.path == "/instance":
            raise AssertionError("standalone mode called MindsHub /instance")
        if request.url.path == "/healthz":
            if isinstance(state["health"], Exception):
                raise state["health"]
            return state["health"]
        if request.url.path == "/_embed":
            view_url = state["embed_url"] or f"{SERVICE}/sessions/{body['session_id']}/view?et=tok"
            return httpx.Response(200, json={"session_id": body["session_id"], "view_url": view_url,
                                             "events_url": f"{SERVICE}/x", "expires_at": 1999999999})
        return httpx.Response(404)

    real = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(svc, "instance_api_base", lambda: "https://api.dev.mindshub.ai")
    return SimpleNamespace(calls=calls, state=state)


def test_browser_settings_defaults_and_scope():
    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    defaults = UserSettings()
    assert defaults.browser_mode == "mindshub"
    assert defaults.browser_service_url == ""
    assert defaults.browser_service_api_key is None
    assert UserSettings.field_is_sensitive("browser_service_api_key")
    assert UserSettings.field_options("browser_mode") == ["mindshub", "standalone"]
    for key in ("browser_mode", "browser_service_url", "browser_service_api_key"):
        assert setting_is_org_scoped(key) is True, key
    # The person's own toggle and MindsHub instance stay per-user.
    for key in ("browser_enabled", "browser_url"):
        assert setting_is_org_scoped(key) is False, key


def test_standalone_service_url_must_be_https_or_loopback():
    for good, expected in [
        ("https://browser.example.com", "https://browser.example.com"),
        ("https://browser.example.com:8443/", "https://browser.example.com:8443"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://127.0.0.1:3000/", "http://127.0.0.1:3000"),
        ("http://[::1]:3000", "http://[::1]:3000"),
    ]:
        assert svc.normalize_service_url(good) == expected, good
    for bad in [
        "", "browser.example.com", "http://browser.example.com", "http://10.0.0.5:3000",
        "https://browser.example.com/api", "https://browser.example.com/?x=1", "https://browser.example.com/#f",
        "ftp://browser.example.com", "https://user:pw@browser.example.com",
    ]:
        assert svc.normalize_service_url(bad) == "", bad


def test_standalone_service_needs_the_mode_a_url_and_a_key():
    assert svc.standalone_service(_standalone()) == (SERVICE, SERVICE_KEY)
    assert svc.standalone_service(_standalone(browser_mode="mindshub")) is None
    assert svc.standalone_service(_standalone(browser_service_url="http://browser.example.com")) is None
    assert svc.standalone_service(_standalone(browser_service_api_key=None)) is None


def test_standalone_profile_is_per_user_and_stable():
    a = svc.standalone_profile("user-a")
    assert a == svc.standalone_profile("user-a")
    assert a != svc.standalone_profile("user-b")
    assert a.startswith("u-") and "user-a" not in a
    assert svc._SESSION_ID.match(a)
    assert svc.standalone_profile(None) == svc.standalone_profile("") == "main"


def test_standalone_status_reads_healthz_with_the_key(service):
    view = asyncio.run(svc.service_status(_standalone()))
    assert view == {"provisioned": True, "status": "running", "endpoint": SERVICE}
    assert service.calls == [("GET", f"{SERVICE}/healthz", None, f"Bearer {SERVICE_KEY}")]

    service.state["health"] = httpx.Response(200, json={"ok": False})
    assert asyncio.run(svc.service_status(_standalone()))["status"] == "unreachable"


def test_standalone_status_reports_an_unreachable_service(service):
    service.state["health"] = httpx.ConnectError("refused")
    view = asyncio.run(svc.service_status(_standalone()))
    assert view == {"provisioned": True, "status": "unreachable", "endpoint": SERVICE}

    service.state["health"] = httpx.Response(503, text="down")
    assert asyncio.run(svc.service_status(_standalone()))["status"] == "unreachable"


def test_standalone_rejected_key_is_an_error_without_the_key(service):
    service.state["health"] = httpx.Response(401, json={"error": f"bad token {SERVICE_KEY}"})
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.service_status(_standalone()))
    assert err.value.detail == "The browser service at browser.example.com rejected the API key."
    assert SERVICE_KEY not in err.value.detail and SERVICE_KEY not in str(err.value)


def test_standalone_errors_name_the_service_not_mindshub(service):
    service.state["health"] = httpx.ConnectError("refused")
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc._call("GET", f"{SERVICE}/healthz", SERVICE_KEY, standalone=True))
    assert "the browser service at browser.example.com" in err.value.detail.lower()
    assert "MindsHub" not in err.value.detail and SERVICE_KEY not in err.value.detail


def test_standalone_status_when_not_configured_calls_nothing(service):
    view = asyncio.run(svc.service_status(_standalone(browser_service_api_key=None)))
    assert view == {"provisioned": False, "status": "not_configured", "endpoint": ""}
    assert service.calls == []


def test_standalone_embed_uses_the_key_and_the_users_profile(service):
    out = asyncio.run(svc.embed_for(_standalone(), "user-jwt", "user-a"))
    profile = svc.standalone_profile("user-a")
    assert out == {"session_id": profile, "view_url": f"{SERVICE}/sessions/{profile}/view?et=tok", "expires_at": 1999999999}
    assert service.calls == [("POST", f"{SERVICE}/_embed", {"session_id": profile}, f"Bearer {SERVICE_KEY}")]


def test_standalone_embed_refuses_someone_elses_session_and_a_foreign_viewer(service):
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed_for(_standalone(), "", "user-a", svc.standalone_profile("user-b")))
    assert err.value.status == 403
    assert service.calls == []

    service.state["embed_url"] = "https://evil.example/view"
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed_for(_standalone(), "", "user-a"))
    assert SERVICE_KEY not in err.value.detail

    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed_for(_standalone(browser_service_url=""), "", "user-a"))
    assert err.value.status == 409


def test_turn_block_standalone_carries_the_token_and_the_users_profile():
    block = svc.turn_block(_standalone(), "user-a")
    assert block == {"base_url": SERVICE, "profile": svc.standalone_profile("user-a"), "token": SERVICE_KEY}
    assert svc.turn_block(_standalone(), None)["profile"] == "main"
    assert svc.turn_block(_standalone(browser_enabled=False), "user-a") is None
    assert svc.turn_block(_standalone(browser_service_api_key=None), "user-a") is None
    # mindshub mode is unchanged: no token, the shared profile, the user id ignored.
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url=ENDPOINT,
                         browser_service_url=SERVICE, browser_service_api_key=SERVICE_KEY)
    assert svc.turn_block(on, "user-a") == {"base_url": ENDPOINT, "profile": "main"}
    config = svc.anton_browser_config(_standalone(), "user-a")
    assert (config.base_url, config.profile) == (SERVICE, svc.standalone_profile("user-a"))
    # Every task keeps the user's one profile (the service has no login store
    # of its own); its project gets a key for saved tabs.
    assert svc.turn_block(_standalone(), "user-a", "conv-1", "proj-1") == {
        "base_url": SERVICE, "profile": svc.standalone_profile("user-a"), "token": SERVICE_KEY,
        "project": svc.project_key("user-a", "proj-1"),
    }
    assert "project" not in svc.turn_block(_standalone(), "user-a", "conv-1")
    # The per-user instance keeps its one profile and no project.
    assert svc.turn_block(on, "user-a", "conv-1", "proj-1") == {"base_url": ENDPOINT, "profile": "main"}


def test_the_harness_uses_the_turns_user_for_the_standalone_profile():
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.db.scoped import TenantScope
    from cowork.harnesses.anton_harness.harness import _browser_config

    with use_settings_scope(TenantScope(org_mode=True, org_id="o", user_id="user-a")):
        assert _browser_config(_standalone()).profile == svc.standalone_profile("user-a")
        assert _browser_config(_standalone(), conversation_id="c1").profile == svc.standalone_profile("user-a")
    assert _browser_config(_standalone()).profile == "main"


def _configure_standalone(client):
    for key, value in [
        ("browser_mode", "standalone"), ("browser_service_url", SERVICE), ("browser_service_api_key", SERVICE_KEY),
    ]:
        assert client.put(f"/api/v1/settings/{key}", json={"value": value}).status_code == 200, key


def test_standalone_routes_need_no_mindshub_credential(client, service):
    _configure_standalone(client)

    status = client.get("/api/v1/browse/status")
    assert status.status_code == 200
    assert status.json() == {"provisioned": True, "status": "running", "endpoint": SERVICE,
                             "mode": "standalone", "enabled": True, "available": True}

    provision = client.post("/api/v1/browse/provision")
    assert provision.status_code == 200 and provision.json()["mode"] == "standalone"
    from cowork.common.settings.user_settings import get_user_settings

    # Never MindsHub, and MindsHub's instance slot stays untouched.
    assert all("/instance" not in url for _, url, _, _ in service.calls)
    assert get_user_settings().browser_url == ""

    embed = client.post("/api/v1/browse/embed", json={})
    assert embed.status_code == 200 and embed.json()["session_id"] == "main"
    assert service.calls[-1][1:] == (f"{SERVICE}/_embed", {"session_id": "main"}, f"Bearer {SERVICE_KEY}")


# ── the event ──────────────────────────────────────────────────────────────


def _events(stream_events):
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def gen():
        for e in stream_events:
            yield e

    async def collect():
        return [chunk async for chunk in format_responses_stream(gen(), model="m")]

    return asyncio.run(collect())


def test_formatter_emits_browser_session_opened():
    from anton.core.llm.provider import StreamBrowserSession

    chunks = _events([StreamBrowserSession(session_id="main", view_url=f"{ENDPOINT}/sessions/main/view?et=t", expires_at=5)])
    opened = [c for c in chunks if c.startswith("event: response.browser_session_opened")]
    assert len(opened) == 1
    data = json.loads(opened[0].split("data: ", 1)[1])
    assert (data["session_id"], data["view_url"], data["expires_at"]) == ("main", f"{ENDPOINT}/sessions/main/view?et=t", 5)


def test_a_pod_browser_step_becomes_the_same_event():
    from anton.core.llm.provider import StreamBrowserSession
    from cowork.turnqueue.producer import step_stream_events

    [event] = step_stream_events({"step": "browser", "session_id": "main", "view_url": f"{ENDPOINT}/x", "expires_at": 7})
    assert event == StreamBrowserSession(session_id="main", view_url=f"{ENDPOINT}/x", expires_at=7)
    assert step_stream_events({"step": "browser", "view_url": "javascript:alert(1)"}) == []
    assert step_stream_events({"step": "browser", "view_url": f"{ENDPOINT}/x", "expires_at": True})[0].expires_at == 0


def test_the_harness_passes_anton_a_browser_only_when_enabled():
    from cowork.harnesses.anton_harness.harness import _browser_config

    assert _browser_config(SimpleNamespace(browser_enabled=False, browser_url=ENDPOINT)) is None
    assert _browser_config(SimpleNamespace(browser_enabled=True, browser_url=ENDPOINT)).base_url == ENDPOINT


# ── the shared router (ENG-3188) ───────────────────────────────────────────

ROUTER = "https://cowork-pr-cowork-1146.dev.mindshub.ai/browser"
#: What a turn's warm-up sends for someone who never touched the proxy settings.
DEFAULT_PREFS = {"proxy": True, "city": ""}


@pytest.fixture
def router(monkeypatch):
    """Fake router at ROUTER; MindsHub's /instance must never be hit."""
    calls: list[tuple[str, str, dict | None, str]] = []
    state = {"status": "none", "fail": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, str(request.url), body, request.headers.get("authorization", "")))
        if state["fail"] is not None:
            return state["fail"]
        path = request.url.path
        if path == "/instance":
            raise AssertionError("router mode called MindsHub /instance")
        if path == "/browser/_status":
            return httpx.Response(200, json={"status": state["status"]})
        if path == "/browser/_warm":
            state["status"] = "starting"
            return httpx.Response(200, json={"status": "starting"})
        if path == "/browser/_prefs":
            return httpx.Response(200, json=state.get("prefs_answer") or {
                "proxy_available": True, "prefs": {"proxy": True, "city": ""},
                "cities": [{"id": "us-los_angeles", "label": "Los Angeles, United States"}, {"id": "bad"}, "x"],
                "effective_city": "us-los_angeles", "extra": "dropped",
            })
        if path == "/browser/_embed":
            sid = body["session_id"]
            return httpx.Response(200, json={
                "session_id": sid, "view_url": f"{ROUTER}/sessions/{sid}/view?et=tok",
                "events_url": "wss://x/events", "expires_at": 1999999999,
            })
        return httpx.Response(404)

    real = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(svc, "instance_api_base", lambda: "https://api.dev.mindshub.ai")
    monkeypatch.setattr(svc, "router_url", lambda: ROUTER)
    return SimpleNamespace(calls=calls, state=state)


def test_router_url_setting_is_validated_and_normalized(monkeypatch):
    from pydantic import ValidationError

    from cowork.common.settings.app_settings import AppSettings

    for raw, expected in [
        ("", ""),
        (ROUTER + "/", ROUTER),
        ("https://cowork.mindshub.ai/browser", "https://cowork.mindshub.ai/browser"),
        ("https://browser.example.com:8443", "https://browser.example.com:8443"),
        ("http://localhost:8080/browser/", "http://localhost:8080/browser"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
    ]:
        monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", raw)
        assert AppSettings().browser_router_url == expected, raw
    for bad in [
        "cowork.mindshub.ai/browser", "http://cowork.mindshub.ai/browser", "https://cowork.mindshub.ai/browser?x=1",
        "https://cowork.mindshub.ai/browser#f", "https://u:p@cowork.mindshub.ai/browser", "ftp://cowork.mindshub.ai",
    ]:
        monkeypatch.setenv("COWORK_BROWSER_ROUTER_URL", bad)
        with pytest.raises(ValidationError):
            AppSettings()


def test_router_is_a_valid_endpoint_only_when_configured(monkeypatch):
    assert not svc.valid_endpoint(ROUTER)
    monkeypatch.setattr(svc, "router_url", lambda: ROUTER)
    assert svc.valid_endpoint(ROUTER) and svc.valid_endpoint(ROUTER + "/")
    assert svc.valid_endpoint(ENDPOINT)
    assert not svc.valid_endpoint(ROUTER + "/sessions")
    assert not svc.valid_endpoint("https://cowork-pr-cowork-1146.dev.mindshub.ai")


def test_router_sessions_and_project_keys_hash_the_ids():
    import hashlib

    sid = svc.router_session("user-a", "conv-1")
    assert sid == "t-" + hashlib.sha256(b"user-a:conv-1").hexdigest()[:24]
    assert sid != svc.router_session("user-a", "conv-2") and svc._SESSION_ID.match(sid)
    assert svc.project_key("user-a", "proj-1") == "p-" + hashlib.sha256(b"user-a:proj-1").hexdigest()[:24]
    assert svc.project_key("user-a", None) == "" and svc.project_key("user-a", "") == ""
    assert svc.router_session(None, "conv-1") == "t-" + hashlib.sha256(b":conv-1").hexdigest()[:24]


def test_router_status_warm_and_embed_go_to_the_router_as_the_caller(router):
    assert asyncio.run(svc.status_for("user-jwt")) == {"provisioned": False, "status": "none", "endpoint": ROUTER}
    assert asyncio.run(svc.provision_for("user-jwt")) == {"provisioned": True, "status": "starting", "endpoint": ROUTER}
    # "starting" reaches the renderer as is.
    assert asyncio.run(svc.status_for("user-jwt"))["status"] == "starting"

    sid = svc.router_session("user-a", "conv-1")
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url="")
    out = asyncio.run(svc.embed_for(on, "user-jwt", "user-a", sid))
    assert out == {"session_id": sid, "view_url": f"{ROUTER}/sessions/{sid}/view?et=tok", "expires_at": 1999999999}
    assert [(m, u, b, a) for m, u, b, a in router.calls] == [
        ("GET", f"{ROUTER}/_status", None, "Bearer user-jwt"),
        ("POST", f"{ROUTER}/_warm", None, "Bearer user-jwt"),
        ("GET", f"{ROUTER}/_status", None, "Bearer user-jwt"),
        ("POST", f"{ROUTER}/_embed", {"session_id": sid}, "Bearer user-jwt"),
    ]


def test_router_embed_checks_the_session_and_the_viewer_url(router, monkeypatch):
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url="")
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed_for(on, "user-jwt", "user-a", "../escape"))
    assert err.value.status == 400 and router.calls == []

    router.state["fail"] = httpx.Response(200, json={"view_url": "https://cowork-pr-cowork-1146.dev.mindshub.ai/x"})
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.embed_for(on, "user-jwt", "user-a", "t-abc"))
    assert err.value.status == svc.UPSTREAM_FAILED


def test_router_errors_pass_through(router):
    router.state["fail"] = httpx.Response(401, json={"error": "unauthorized", "message": "Sign in again."})
    with pytest.raises(svc.BrowserServiceError) as err:
        asyncio.run(svc.status_for("stale"))
    assert (err.value.status, err.value.detail) == (401, "Sign in again.")


def test_turn_block_with_the_router(monkeypatch):
    monkeypatch.setattr(svc, "router_url", lambda: ROUTER)
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url=ENDPOINT)
    assert svc.turn_block(on, "user-a", "conv-1", "proj-1") == {
        "base_url": ROUTER, "profile": svc.router_session("user-a", "conv-1"),
        "project": svc.project_key("user-a", "proj-1"),
    }
    assert svc.turn_block(on, "user-a", "conv-1") == {"base_url": ROUTER, "profile": svc.router_session("user-a", "conv-1")}
    # No stored instance needed: the router is the endpoint.
    assert svc.turn_block(SimpleNamespace(browser_enabled=True, browser_url=""), "u", "c")["base_url"] == ROUTER
    assert svc.turn_block(SimpleNamespace(browser_enabled=False, browser_url=""), "u", "c") is None
    # Standalone still wins when the org chose it.
    assert svc.turn_block(_standalone(), "user-a", "conv-1")["base_url"] == SERVICE
    config = svc.anton_browser_config(on, "user-a", "conv-1")
    assert (config.base_url, config.profile) == (ROUTER, svc.router_session("user-a", "conv-1"))


def test_start_early_warms_without_waiting_or_failing(router):
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url="")

    async def scenario():
        assert svc.start_early(on, "turn-key") is True
        # Not awaited by the caller: nothing has gone out yet.
        assert router.calls == []
        await asyncio.gather(*list(svc._WARMING))
        assert router.calls == [("POST", f"{ROUTER}/_warm", {"prefs": DEFAULT_PREFS}, "Bearer turn-key")]
        assert not svc._WARMING

        # A failing router is only logged.
        router.state["fail"] = httpx.Response(503, json={"error": "no_capacity"})
        assert svc.start_early(on, "turn-key") is True
        await asyncio.gather(*list(svc._WARMING))

        # Off, standalone, or no credential: nothing to warm.
        assert svc.start_early(SimpleNamespace(browser_enabled=False), "turn-key") is False
        assert svc.start_early(_standalone(), "turn-key") is False
        assert svc.start_early(on, "") is False
        assert svc.start_early(on, None) is False

    asyncio.run(scenario())
    assert len(router.calls) == 2
    # Outside a running loop it declines rather than raising.
    assert svc.start_early(on, "turn-key") is False


def test_start_early_needs_the_router(mindshub):
    async def scenario():
        on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url=ENDPOINT)
        assert svc.start_early(on, "turn-key") is False

    asyncio.run(scenario())
    assert mindshub.calls == []


def test_router_routes_answer_from_the_router_and_store_nothing(client, router):
    client.put("/api/v1/settings/browser_enabled", json={"value": "true"})
    status = client.get("/api/v1/browse/status", headers=HUB)
    assert status.status_code == 200
    assert status.json() == {
        "provisioned": False, "status": "none", "endpoint": ROUTER, "mode": "mindshub", "enabled": True,
        "available": False,
    }
    provision = client.post("/api/v1/browse/provision", headers=HUB).json()
    assert (provision["status"], provision["endpoint"], provision["available"]) == ("starting", ROUTER, False)
    router.state["status"] = "running"
    assert client.get("/api/v1/browse/status", headers=HUB).json()["available"] is True

    from cowork.common.settings.user_settings import get_user_settings

    assert get_user_settings().browser_url == ""

    sid = svc.router_session(None, "conv-1")
    embed = client.post("/api/v1/browse/embed", headers=HUB, json={"session_id": sid})
    assert embed.status_code == 200 and embed.json()["view_url"] == f"{ROUTER}/sessions/{sid}/view?et=tok"
    assert all(auth == "Bearer user-jwt" for _, _, _, auth in router.calls)


def test_the_harness_threads_the_task_and_warms_with_the_held_credential(router, monkeypatch):
    from cowork.common.settings import runtime_credential
    from cowork.harnesses.anton_harness.harness import _browser_config

    monkeypatch.setattr(runtime_credential, "get_minds_credential", lambda: "desktop-jwt")
    on = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_url="")

    async def scenario():
        config = _browser_config(on, conversation_id="conv-1", project_id="proj-1")
        assert (config.base_url, config.profile) == (ROUTER, svc.router_session(None, "conv-1"))
        await asyncio.gather(*list(svc._WARMING))

    asyncio.run(scenario())
    assert router.calls == [("POST", f"{ROUTER}/_warm", {"prefs": DEFAULT_PREFS}, "Bearer desktop-jwt")]


def test_proxy_settings_defaults_scope_and_city_check():
    from pydantic import ValidationError

    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    defaults = UserSettings()
    assert (defaults.browser_proxy, defaults.browser_city) == (True, "")
    for key in ("browser_proxy", "browser_city"):
        assert setting_is_org_scoped(key) is False, key
    assert UserSettings(browser_city=" us-new_york ").browser_city == "us-new_york"
    for bad in ("US-new_york", "us", "us-new york", "us-../x", "usa-la"):
        with pytest.raises(ValidationError):
            UserSettings(browser_city=bad)


def test_router_prefs_keep_only_sane_time_zones_and_locales():
    user = SimpleNamespace(browser_proxy=False, browser_city="gb-london")
    assert svc.router_prefs(user) == {"proxy": False, "city": "gb-london"}
    assert svc.router_prefs(user, "Europe/London", "en-GB") == {
        "proxy": False, "city": "gb-london", "timezone": "Europe/London", "locale": "en-GB",
    }
    assert svc.router_prefs(user, "Etc/GMT+5", "zh-Hant-TW")["timezone"] == "Etc/GMT+5"
    assert svc.router_prefs(user, "Europe/London; rm", "en_GB<script>") == {"proxy": False, "city": "gb-london"}
    # Settings objects from before the proxy existed read as the defaults.
    assert svc.router_prefs(SimpleNamespace()) == DEFAULT_PREFS


def test_start_early_sends_the_users_proxy_choice(router):
    off = SimpleNamespace(browser_enabled=True, browser_mode="mindshub", browser_proxy=False, browser_city="fr-paris")

    async def scenario():
        assert svc.start_early(off, "turn-key") is True
        await asyncio.gather(*list(svc._WARMING))

    asyncio.run(scenario())
    assert router.calls == [("POST", f"{ROUTER}/_warm", {"prefs": {"proxy": False, "city": "fr-paris"}}, "Bearer turn-key")]


def test_provision_hands_the_router_the_prefs_and_locations_list_cities(client, router):
    client.put("/api/v1/settings/browser_enabled", json={"value": "true"})
    client.put("/api/v1/settings/browser_city", json={"value": "us-los_angeles"})
    # No body still works, as the renderer that doesn't send one expects.
    assert client.post("/api/v1/browse/provision", headers=HUB).status_code == 200
    answer = client.post(
        "/api/v1/browse/provision", headers=HUB, json={"timezone": "America/Los_Angeles", "locale": "en-US"},
    )
    assert answer.status_code == 200
    warms = [body for method, url, body, _ in router.calls if url.endswith("/_warm")]
    assert warms == [
        {"prefs": {"proxy": True, "city": "us-los_angeles"}},
        {"prefs": {"proxy": True, "city": "us-los_angeles", "timezone": "America/Los_Angeles", "locale": "en-US"}},
    ]

    listed = client.get("/api/v1/browse/locations", headers=HUB)
    assert listed.status_code == 200
    assert listed.json() == {
        "proxy_available": True, "prefs": {"proxy": True, "city": ""},
        "cities": [
            # An id Settings couldn't save is dropped.
            {"id": "us-los_angeles", "label": "Los Angeles, United States"},
        ],
        "effective_city": "us-los_angeles",
    }
    assert ("GET", f"{ROUTER}/_prefs", None, "Bearer user-jwt") in router.calls


def test_locations_without_a_router_or_in_standalone(client, mindshub, monkeypatch):
    assert client.get("/api/v1/browse/locations", headers=HUB).json() == svc.NO_LOCATIONS
    assert mindshub.calls == []
    monkeypatch.setattr(svc, "router_url", lambda: ROUTER)
    client.put("/api/v1/settings/browser_mode", json={"value": "standalone"})
    assert client.get("/api/v1/browse/locations", headers=HUB).json() == svc.NO_LOCATIONS
    assert mindshub.calls == []
