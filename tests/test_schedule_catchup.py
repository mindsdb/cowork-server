from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import cowork.handlers.responses as responses_mod
import cowork.harnesses.anton_harness.browser_tools as bt
from cowork.db.session import get_open_session
from cowork.scheduler import _due_schedules, execute_schedule
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService


def _schedule(session, *, title, requires_browser, next_run_at=None, cadence="daily"):
    return ScheduleService(session).create_schedule(
        title=title,
        prompt="do browser things",
        cadence=cadence,
        next_run_at=next_run_at or (datetime.now(timezone.utc) - timedelta(minutes=2)),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
        requires_browser=requires_browser,
    )


def test_browser_schedule_defers_while_bridge_is_down():
    session = get_open_session()
    try:
        sched = _schedule(session, title="needs browser", requires_browser=True)
        plain = _schedule(session, title="plain", requires_browser=False)

        due_down = _due_schedules(session, datetime.now(timezone.utc), bridge_up=False)
        assert sched not in due_down  # deferred, not consumed
        assert plain in due_down  # non-browser schedules degrade and run

        due_up = _due_schedules(session, datetime.now(timezone.utc), bridge_up=True)
        assert sched in due_up  # catch-up fires the moment the bridge returns
    finally:
        session.close()


def test_late_fire_is_marked_catch_up(monkeypatch):
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
    monkeypatch.setattr(bt, "bridge_available", lambda: _true())

    session = get_open_session()
    sched = _schedule(
        session,
        title="Morning digest",
        requires_browser=True,
        next_run_at=datetime.now(timezone.utc) - timedelta(hours=2),  # slept through the slot
    )
    schedule_id = sched.id
    session.close()

    asyncio.run(execute_schedule(schedule_id, is_manual=False))

    from cowork.models.conversation import Conversation
    from sqlmodel import select

    check = get_open_session()
    conv = check.exec(select(Conversation).order_by(Conversation.created_at.desc())).first()  # type: ignore[attr-defined]
    assert conv.topic.startswith("Morning digest (catch-up — was due ")
    check.close()


async def _true() -> bool:
    return True
