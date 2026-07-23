from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import cowork.handlers.responses as responses_mod
import cowork.harnesses.anton_harness.browser_tools as bt
from cowork.db.session import get_open_session
from cowork.scheduler import execute_schedule
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleService


def _run_schedule(monkeypatch, bridge_up: bool):
    captured: list = []

    class FakeHandler:
        def __init__(self, session):
            pass

        async def handle(self, request):
            captured.append(request)

            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)
    monkeypatch.setattr(bt, "bridge_available", lambda: _ret(bridge_up))

    async def _noop():
        return None

    session = get_open_session()
    schedule = ScheduleService(session).create_schedule(
        title="surface test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    session.close()
    asyncio.run(execute_schedule(schedule_id, is_manual=False))
    return captured[0]


async def _ret(v: bool) -> bool:
    return v


def test_scheduled_turn_carries_browser_surface_when_bridge_up(monkeypatch):
    request = _run_schedule(monkeypatch, bridge_up=True)
    assert request.surface == "browser"


def test_scheduled_turn_degrades_when_bridge_down(monkeypatch):
    request = _run_schedule(monkeypatch, bridge_up=False)
    assert request.surface is None


def test_turn_context_marks_needs_auth_tabs(monkeypatch):
    async def _state(method, path, *, params=None, body=None, timeout=10.0):
        return {
            "activeTabId": "b",
            "tabs": [
                {"id": "a", "title": "Inbox", "url": "https://mail.google.com", "needsAuth": True},
                {"id": "b", "title": "Linear", "url": "https://linear.app"},
            ],
        }

    monkeypatch.setattr(bt, "_bridge_call", _state)
    import asyncio as aio

    context = aio.new_event_loop().run_until_complete(bt.build_browser_turn_context())
    assert "NEEDS SIGN-IN" in context
    assert "Inbox — https://mail.google.com — NEEDS SIGN-IN (ask the user, don't attempt)" in context
    assert "Linear — https://linear.app\n" in context or context.endswith("Linear — https://linear.app")
