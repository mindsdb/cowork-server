"""Stopping the scheduler for a shutdown.

With remote turns the server cancels the poll loop before it ends the turns
in flight, so no run starts behind that, and then waits for the loop and for
the runs it already started, so each one records its turn's outcome and
finishes its run row before the process exits. A run whose turn the server
refuses because it is shutting down never started, and its schedule is left
as it was. A run a shutdown cancels mid-turn is saved as failed. The new
pods a rollout starts together begin polling at random points in the poll
interval.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cowork import scheduler
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.streaming.registry import registry


@pytest.fixture(autouse=True)
def _no_scheduler_left(monkeypatch):
    """A poll loop and runs of the scheduler's own, whatever an earlier test
    left in its module globals; monkeypatch puts those back afterwards."""
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())


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


async def test_a_new_poll_loop_waits_a_random_part_of_the_interval_before_its_first_poll(monkeypatch):
    """Pods a rollout starts together would otherwise poll in step, and two
    replicas polling together can both start one due schedule."""
    loop = asyncio.get_running_loop()
    drawn: list[tuple[float, float]] = []
    first_poll: list[float] = []

    def uniform(low: float, high: float) -> float:
        drawn.append((low, high))
        return 0.1

    async def poll(fn, *, scope):
        first_poll.append(loop.time())
        return []

    monkeypatch.setattr(scheduler, "random", SimpleNamespace(uniform=uniform))
    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "run_db", poll)
    started = loop.time()
    polling = asyncio.create_task(scheduler._scheduler_loop())
    try:
        await _until(lambda: first_poll)
    finally:
        polling.cancel()
        await asyncio.gather(polling, return_exceptions=True)

    assert drawn == [(0, 0.01)]
    assert first_poll[0] - started >= 0.1


async def test_stopping_a_scheduler_that_never_started_does_nothing():
    assert scheduler.stop_scheduler() is None
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=1) == 0


@dataclass(frozen=True)
class DueSchedule:
    """A daily schedule whose slot is due, and a reader for what a run left."""

    id: UUID
    next_run_at: datetime
    title: str
    state: Callable[[], tuple[Schedule, ScheduleRun, list[UUID]]]


@pytest.fixture
def due_schedule():
    """A daily schedule whose slot is due, under a title of its own. ``state``
    reads back the schedule, its one run and the conversations under its
    title. All three are removed afterwards."""
    from sqlmodel import select

    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService
    from cowork.services.projects import GENERAL_PROJECT_ID
    from cowork.services.schedules import ScheduleRunService, ScheduleService

    title = f"due daily {uuid4()}"
    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title=title,
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

    def conversations(check) -> list[UUID]:
        return list(check.exec(select(Conversation.id).where(Conversation.topic == title)).all())

    def state() -> tuple[Schedule, ScheduleRun, list[UUID]]:
        check = get_open_session()
        try:
            fresh = ScheduleService(ScopedSession(check, SYSTEM_SCOPE)).get_schedule(schedule_id)
            (run,) = ScheduleRunService(ScopedSession(check, SYSTEM_SCOPE)).list_runs(schedule_id)
            return fresh, run, conversations(check)
        finally:
            check.close()

    yield DueSchedule(id=schedule_id, next_run_at=next_run_at, title=title, state=state)
    cleanup = get_open_session()
    for conversation_id in conversations(cleanup):
        ConversationService(ScopedSession(cleanup, SYSTEM_SCOPE)).delete_conversation(conversation_id)
    ScheduleService(ScopedSession(cleanup, SYSTEM_SCOPE)).delete_schedule(schedule_id)
    cleanup.close()


@pytest.fixture
def refused_for_shutdown(monkeypatch, caplog, due_schedule):
    """A due schedule, a closed registry, and a turn handler that starts its
    turn through that registry as ResponsesHandler.handle does."""
    import cowork.handlers.responses as responses_mod

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
    registry.close()
    yield due_schedule
    registry.reset()


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

    schedule = refused_for_shutdown

    await scheduler.execute_schedule(schedule.id, is_manual=False)

    fresh, run, conversations = schedule.state()
    assert run.status == RunStatus.failed
    assert run.error == "Run did not start: the server was shutting down."
    assert run.conversation_id is None
    assert fresh.last_error is None
    assert ensure_utc(fresh.next_run_at) == ensure_utc(schedule.next_run_at)
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

    schedule = refused_for_shutdown
    session = get_open_session()
    conversation_id = ConversationService(ScopedSession(session, SYSTEM_SCOPE)).create_conversation(
        topic=schedule.title,
    ).id
    session.close()

    await scheduler.execute_schedule(schedule.id, is_manual=True, conversation_id=conversation_id)

    fresh, run, conversations = schedule.state()
    assert run.error == "Run did not start: the server was shutting down."
    assert run.conversation_id == conversation_id
    assert fresh.last_error is None
    assert conversations == [conversation_id]


@pytest.mark.parametrize("is_manual", [False, True], ids=["cron", "run-now"])
async def test_a_run_a_shutdown_cancels_mid_turn_is_saved_failed_and_its_slot_stays_due(
    monkeypatch, due_schedule, is_manual,
):
    """Uvicorn's request drain cancels a Run now request at its bound, and the
    event loop's close cancels a cron run still going. The run is saved as
    failed, not success, so the next poll still runs the slot: the freshness
    guard skips a slot only after a successful run."""
    import cowork.handlers.responses as responses_mod
    from cowork.common.datetime_utils import ensure_utc
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.schemas.schedules import RunStatus

    streaming = asyncio.Event()

    class MidTurnHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            async def stream():
                yield "event: response.output_text.delta\ndata: {}\n\n"
                streaming.set()
                await asyncio.Event().wait()

            return stream()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", MidTurnHandler)
    run = asyncio.create_task(scheduler.execute_schedule(due_schedule.id, is_manual=is_manual))
    await asyncio.wait_for(streaming.wait(), timeout=5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    fresh, finished, _ = due_schedule.state()
    assert finished.status == RunStatus.failed
    assert finished.error == "Run was interrupted: the server shut down before it finished."
    assert ensure_utc(fresh.next_run_at) == ensure_utc(due_schedule.next_run_at)
    poll = get_open_session()
    try:
        due = scheduler._due_schedules(ScopedSession(poll, SYSTEM_SCOPE), datetime.now(timezone.utc))
    finally:
        poll.close()
    assert due_schedule.id in [schedule.id for schedule in due]
