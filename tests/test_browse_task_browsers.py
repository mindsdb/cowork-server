"""The project's browsers (ENG-4138, prototype): the open browsers among a
project's tasks, with their tabs, and the person's switch and close, as the
caller."""
from __future__ import annotations

import pytest

import cowork.services.browser as browser

ROUTER = "https://cowork.test/browser"


@pytest.fixture
def router(monkeypatch):
    calls: list[tuple[str, str]] = []
    mine = browser.task_session("u1", "c1")
    other = browser.task_session("u1", "c2")

    async def call(method, url, bearer, *, json=None, headers=None):
        calls.append((method, url))
        if url == f"{ROUTER}/sessions":
            return {"sessions": [
                {"session_id": "t-someone-elses", "running": True, "url": "https://x.test/"},
                {"session_id": other, "running": True, "url": "https://form.test/", "viewers": 0,
                 "control": {"holder": "human"}, "last_active_at": "2026-11-12T10:00:00.000Z"},
                {"session_id": mine, "running": True, "url": "https://shop.test/", "viewers": 1,
                 "control": {"holder": "agent"}, "last_active_at": "2026-11-12T11:00:00.000Z",
                 "parked_tabs": [{"url": "https://help.test/", "title": "Help"}]},
            ]}
        if url == f"{ROUTER}/sessions/{other}/tabs":
            raise browser.BrowserServiceError(409, "human_in_control", "The person is using the browser.")
        if url.endswith("/tabs"):
            return {"tabs": [{"id": "t1", "url": "https://shop.test/", "title": "Shop", "active": True}]}
        return {"ok": True}

    monkeypatch.setattr(browser, "_call", call)
    monkeypatch.setattr(browser, "router_url", lambda: ROUTER)
    return calls


async def test_lists_only_the_given_tasks_browsers_newest_first_with_their_tabs(router):
    listed = await browser.task_browsers("Bearer jwt", user_id="u1", conversation_ids=["c1", "c2", "c3"])

    assert [b.conversation_id for b in listed.browsers] == ["c1", "c2"]
    shop, form = listed.browsers
    assert [t.id for t in shop.tabs] == ["t1"] and shop.viewers == 1
    assert [t.url for t in shop.parked_tabs] == ["https://help.test/"]
    # Held by the person: listed, without tabs.
    assert form.holder == "human" and form.tabs == []


async def test_another_persons_ids_name_none_of_the_callers_browsers(router):
    listed = await browser.task_browsers("Bearer jwt", user_id="u2", conversation_ids=["c1", "c2"])
    assert listed.browsers == []


async def test_switch_close_and_close_all_act_on_the_tasks_own_session(router):
    session = browser.task_session("u1", "c1")
    await browser.switch_tab("Bearer jwt", user_id="u1", conversation_id="c1", tab_id="t2")
    await browser.close_tab("Bearer jwt", user_id="u1", conversation_id="c1", tab_id="t2", park=True)
    await browser.close_task_browser("Bearer jwt", user_id="u1", conversation_id="c1")
    assert router == [
        ("POST", f"{ROUTER}/sessions/{session}/tabs/t2/activate"),
        ("DELETE", f"{ROUTER}/sessions/{session}/tabs/t2?park=true"),
        ("DELETE", f"{ROUTER}/sessions/{session}"),
    ]
