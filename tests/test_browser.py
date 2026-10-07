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
        for key in ("browser_url", "browser_enabled"):
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
    assert res.json() == {"provisioned": False, "status": "none", "endpoint": "", "enabled": False, "available": False}


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


def test_mindshub_refusals_pass_through(client, monkeypatch):
    async def refused(*args, **kwargs):
        raise svc.BrowserServiceError(403, "Your plan doesn't include agents.")

    monkeypatch.setattr(svc, "_call", refused)
    res = client.post("/api/v1/browse/provision", headers=HUB)
    assert res.status_code == 403
    assert res.json()["detail"] == "Your plan doesn't include agents."


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
