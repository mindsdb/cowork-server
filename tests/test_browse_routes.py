"""The task browser routes (ENG-4138, prototype) over HTTP: each acts on the
caller's own task sessions, named from the signed-in person, never from the body."""
from __future__ import annotations

import httpx
import pytest

import cowork.services.browser as browser
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.server import create_app

BEARER = "Bearer user-jwt"
SCOPE = TenantScope(org_mode=True, org_id="o1", user_id="u-scope")


@pytest.fixture
def service(monkeypatch):
    """The browser service: records each call's credential and arguments."""
    calls: list[tuple[str, str, dict]] = []

    def fake(name, answer):
        async def run(bearer, **kwargs):
            calls.append((name, bearer, kwargs))
            return answer
        return run

    monkeypatch.setattr(browser, "person_open", fake("open", browser.OpenedSession(session_id="t-x")))
    monkeypatch.setattr(browser, "task_browsers", fake("browsers", browser.TaskBrowsers(browsers=[])))
    monkeypatch.setattr(browser, "switch_tab", fake("switch", None))
    monkeypatch.setattr(browser, "close_tab", fake("close_tab", None))
    monkeypatch.setattr(browser, "close_task_browser", fake("close", None))
    return calls


async def _post(path: str, body: dict) -> httpx.Response:
    app = create_app()
    app.dependency_overrides[get_tenant_scope] = lambda: SCOPE
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(f"/api/v1/browse{path}", headers={"X-MindsHub-Authorization": BEARER}, json=body)


@pytest.mark.parametrize(
    ("path", "body", "name", "kwargs"),
    [
        ("/open", {"conversation_id": "c1", "url": "example.com"}, "open",
         {"conversation_id": "c1", "url": "https://example.com"}),
        ("/browsers", {"conversation_ids": ["c1", "c2"]}, "browsers", {"conversation_ids": ["c1", "c2"]}),
        ("/tabs/switch", {"conversation_id": "c1", "tab_id": "t2"}, "switch", {"conversation_id": "c1", "tab_id": "t2"}),
        ("/tabs/close", {"conversation_id": "c1", "tab_id": "t2", "park": False}, "close_tab",
         {"conversation_id": "c1", "tab_id": "t2", "park": False}),
        ("/close", {"conversation_id": "c1"}, "close", {"conversation_id": "c1"}),
    ],
)
async def test_each_route_acts_as_the_signed_in_person_not_a_body_user(service, path, body, name, kwargs):
    answered = await _post(path, {**body, "user_id": "someone-else"})

    assert answered.status_code == 200, answered.text
    assert service == [(name, "user-jwt", {"user_id": "u-scope", **kwargs})]


async def test_open_refuses_what_isnt_a_web_address(service):
    answered = await _post("/open", {"conversation_id": "c1", "url": "javascript:alert(1)"})

    assert answered.status_code == 400
    assert answered.json()["detail"]["code"] == "bad_url"
    assert service == []


@pytest.mark.parametrize("path", ["/tabs/switch", "/tabs/close"])
@pytest.mark.parametrize("tab_id", ["t1/../x", "t0", "../t1"])
async def test_a_tab_id_that_isnt_one_is_refused(service, path, tab_id):
    answered = await _post(path, {"conversation_id": "c1", "tab_id": tab_id})

    assert answered.status_code == 422
    assert service == []


async def test_a_router_refusal_keeps_its_status(service, monkeypatch):
    async def refused(bearer, **kwargs):
        raise browser.BrowserServiceError(401, "unauthorized", "Sign in again.")

    monkeypatch.setattr(browser, "close_task_browser", refused)

    answered = await _post("/close", {"conversation_id": "c1"})

    assert answered.status_code == 401
    assert answered.json()["detail"]["code"] == "unauthorized"
