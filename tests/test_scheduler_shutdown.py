"""Stopping the scheduler for a shutdown.

With remote turns the server cancels the poll loop before it ends the turns
in flight, so no run starts behind that, and then waits for the loop and for
the runs it already started, so each one records its turn's outcome and
finishes its run row before the process exits. A run whose turn the server
refuses because it is shutting down never started, and its schedule is left
as it was.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cowork import scheduler
from cowork.streaming.registry import registry


@pytest.fixture
async def one_due_schedule(monkeypatch):
    """A poll loop that finds one schedule due on its first poll and none
    after, and a run for it that holds until ``release`` is set."""
    schedule_id = uuid4()
    polls: list[int] = []
    release = asyncio.Event()
    finished: list[object] = []

    async def poll(fn, *, scope):
        polls.append(len(polls))
        return [schedule_id] if len(polls) == 1 else []

    async def execute_schedule(due_id, is_manual=False, conversation_id=None):
        await release.wait()
        finished.append(due_id)

    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(scheduler, "run_db", poll)
    monkeypatch.setattr(scheduler, "execute_schedule", execute_schedule)
    yield polls, release, finished, schedule_id
    release.set()
    task = scheduler._scheduler_task
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    scheduler._scheduler_task = None


async def _until(condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0)


async def test_stopping_twice_is_safe_and_the_runs_it_started_are_awaited(one_due_schedule):
    polls, release, finished, schedule_id = one_due_schedule
    scheduler.start_scheduler()
    loop_task = scheduler._scheduler_task
    await _until(lambda: scheduler._scheduled_runs)

    assert scheduler.stop_scheduler() is loop_task
    assert scheduler.stop_scheduler() is None
    await asyncio.wait_for(asyncio.gather(loop_task, return_exceptions=True), timeout=2)

    assert loop_task.cancelled() and scheduler._scheduler_task is None
    polled = len(polls)
    await asyncio.sleep(0.05)
    assert len(polls) == polled, "the loop polled after it was stopped"

    # The run holds, so a short wait reports it still going.
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=0.05) == 1
    release.set()
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=1) == 0
    assert finished == [schedule_id]
    assert not scheduler._scheduled_runs


async def test_stopping_a_scheduler_that_never_started_does_nothing():
    assert scheduler._scheduler_task is None
    assert scheduler.stop_scheduler() is None
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=1) == 0


@pytest.fixture
def refused_for_shutdown(monkeypatch, caplog):
    """A daily schedule, a closed registry, and a turn handler that starts its
    turn through that registry as ResponsesHandler.handle does. Yields the
    schedule's id and a function that reads back its state."""
    import cowork.handlers.responses as responses_mod
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.projects import GENERAL_PROJECT_ID
    from cowork.services.schedules import ScheduleService

    class ClosedRegistryHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            await registry.start(
                conversation_id=request.conversation, turn_id=0,
                open_buffer=None, produce=None,
            )

    monkeypatch.setattr(responses_mod, "ResponsesHandler", ClosedRegistryHandler)
    # The migration tests' alembic env calls fileConfig, which disables every
    # logger that existed before it for the rest of the session.
    monkeypatch.setattr(scheduler.logger, "disabled", False)
    caplog.set_level(logging.INFO, logger=scheduler.logger.name)
    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title="refused for shutdown",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id, next_run_at = schedule.id, schedule.next_run_at
    session.close()
    registry.close()

    def state():
        from sqlmodel import select

        from cowork.models.conversation import Conversation
        from cowork.services.schedules import ScheduleRunService

        check = get_open_session()
        try:
            fresh = ScheduleService(ScopedSession(check, SYSTEM_SCOPE)).get_schedule(schedule_id)
            (run,) = ScheduleRunService(ScopedSession(check, SYSTEM_SCOPE)).list_runs(schedule_id)
            conversations = list(check.exec(
                select(Conversation.id).where(Conversation.topic == "refused for shutdown")
            ).all())
            return fresh, run, conversations
        finally:
            check.close()

    yield schedule_id, next_run_at, state
    registry.reset()
    cleanup = get_open_session()
    ScheduleService(ScopedSession(cleanup, SYSTEM_SCOPE)).delete_schedule(schedule_id)
    cleanup.close()


def _scheduler_records(caplog) -> list[tuple[int, str]]:
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == scheduler.logger.name]


async def test_a_run_refused_for_shutdown_leaves_its_slot_due_and_no_empty_conversation(
    refused_for_shutdown, caplog,
):
    """The poll loop started this run just before the server closed its
    registry. The run did not start, so it is no failure of the schedule's:
    its last_error stays empty, its cron slot stays due for a replica still
    serving, the conversation made for it is deleted, and the log says so at
    INFO rather than as a failed run's ERROR."""
    from cowork.common.datetime_utils import ensure_utc
    from cowork.schemas.schedules import RunStatus

    schedule_id, next_run_at, state = refused_for_shutdown

    await scheduler.execute_schedule(schedule_id, is_manual=False)

    fresh, run, conversations = state()
    assert run.status == RunStatus.failed
    assert run.error == "Run did not start: the server was shutting down."
    assert run.conversation_id is None
    assert fresh.last_error is None
    assert ensure_utc(fresh.next_run_at) == ensure_utc(next_run_at)
    assert conversations == []
    assert _scheduler_records(caplog) == [
        (logging.INFO, "Schedule run did not start; the server is shutting down"),
    ]


async def test_a_manual_run_refused_for_shutdown_keeps_the_conversation_its_request_made(
    refused_for_shutdown,
):
    """Run now makes the conversation before the run and hands its id to the
    web UI, which opens it; only a conversation the run made itself is
    deleted."""
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.conversations import ConversationService

    schedule_id, _, state = refused_for_shutdown
    session = get_open_session()
    conversation_id = ConversationService(ScopedSession(session, SYSTEM_SCOPE)).create_conversation(
        topic="refused for shutdown",
    ).id
    session.close()

    await scheduler.execute_schedule(schedule_id, is_manual=True, conversation_id=conversation_id)

    fresh, run, conversations = state()
    assert run.error == "Run did not start: the server was shutting down."
    assert run.conversation_id == conversation_id
    assert fresh.last_error is None
    assert conversations == [conversation_id]
    cleanup = get_open_session()
    ConversationService(ScopedSession(cleanup, SYSTEM_SCOPE)).delete_conversation(conversation_id)
    cleanup.close()
