"""The person opens an address in a task's browser from the pane (ENG-4138,
prototype): the task's own session opens at it, or goes there, as the caller,
with no agent."""
from __future__ import annotations

import pytest

import cowork.services.browser as browser


@pytest.mark.parametrize(
    ("typed", "url"),
    [
        ("example.com", "https://example.com"),
        (" https://news.ycombinator.com/item?id=1 ", "https://news.ycombinator.com/item?id=1"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("best tacos", None),
        ("javascript:alert(1)", None),
        ("file:///etc/passwd", None),
        ("", None),
    ],
)
def test_person_url(typed, url):
    assert browser.person_url(typed) == url


@pytest.fixture
def router(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []

    class Fake:
        resumed = False

    async def call(method, url, bearer, *, json=None, headers=None):
        calls.append((method, url, json))
        # The router's own answer (browser contract sessions.json).
        return {"session_id": "t-x", "resumed": Fake.resumed, "url": None} if url.endswith("/sessions") else {}

    monkeypatch.setattr(browser, "_call", call)
    monkeypatch.setattr(browser, "router_url", lambda: "https://cowork.test/browser")
    Fake.calls = calls
    return Fake


async def test_opens_the_tasks_own_session_at_the_address(router):
    opened = await browser.person_open("Bearer jwt", user_id="u1", conversation_id="c1", url="https://example.com")

    session = browser.task_session("u1", "c1")
    assert opened.session_id == session
    assert router.calls == [("POST", "https://cowork.test/browser/sessions", {"session_id": session, "url": "https://example.com"})]


async def test_an_open_session_goes_to_the_address(router):
    router.resumed = True

    await browser.person_open("Bearer jwt", user_id="u1", conversation_id="c1", url="https://example.com")

    session = browser.task_session("u1", "c1")
    assert router.calls[-1] == (
        "POST", f"https://cowork.test/browser/sessions/{session}/action", {"action": "navigate", "url": "https://example.com"},
    )


async def test_no_address_opens_a_blank_tab_and_leaves_an_open_one_be(router):
    router.resumed = True

    await browser.person_open("Bearer jwt", user_id="u1", conversation_id="c1", url=None)

    session = browser.task_session("u1", "c1")
    assert router.calls == [("POST", "https://cowork.test/browser/sessions", {"session_id": session})]
