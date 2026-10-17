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
        "provisioned": False, "status": "none", "endpoint": "", "mode": "mindshub", "enabled": False, "available": False,
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


def test_the_harness_uses_the_turns_user_for_the_standalone_profile():
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.db.scoped import TenantScope
    from cowork.harnesses.anton_harness.harness import _browser_config

    with use_settings_scope(TenantScope(org_mode=True, org_id="o", user_id="user-a")):
        assert _browser_config(_standalone()).profile == svc.standalone_profile("user-a")
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
                             "mode": "standalone", "enabled": False, "available": False}

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
