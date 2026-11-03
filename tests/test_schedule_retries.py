"""A cron run that failed before its agent acted is tried again.

Each attempt is its own run row for the same slot. The failed attempt gives
the slot back, and a later poll claims it again for the next attempt once the
pause for it is over: 60 seconds before the second attempt, 300 before the
third. A run whose agent may have acted is never tried again, a manual run
gets one attempt, and a run ended by a shutdown or a lost process is not an
attempt at all. Every run records what is known of its execution (`outcome`)
and when its turn was handed to the agent (`execution_started_at`). One
process runs at most three scheduled runs at once.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import create_engine, inspect, text

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import run_schema_migrations
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService
from cowork.streaming.buffer import FileStreamBuffer

# A slot that came due a minute ago, so the poll's missed-run pass leaves it due.
SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._MAX_CONCURRENT_RUNS))


def _make_schedule(*, cadence: str = "daily", next_run_at: datetime = SLOT) -> UUID:
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
            title=f"retries {uuid4()}", prompt="do the thing", cadence=cadence, next_run_at=next_run_at,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id
    finally:
        session.close()


def _delete_schedule(schedule_id: UUID) -> None:
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        title = session.get(Schedule, schedule_id).title
        for conversation_id in session.exec(select(Conversation.id).where(Conversation.topic == title)).all():
            ConversationService(scoped).delete_conversation(conversation_id)
        ScheduleService(scoped).delete_schedule(schedule_id)
    finally:
        session.close()


@pytest.fixture
def schedule_id():
    created = _make_schedule()
    yield created
    _delete_schedule(created)


@pytest.fixture
def turns(monkeypatch):
    """A turn handler that counts the turns it starts and ends each at once,
    and a switch for what the finished turn's records show."""
    import cowork.handlers.responses as responses_mod

    started: list[str] = []
    ending = SimpleNamespace(reason="completed", acted=True)

    class CountingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            started.append(request.conversation)

            async def stream():
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        return scheduler._TurnEvidence(reason=ending.reason, acted=ending.acted)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", CountingHandler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    return SimpleNamespace(started=started, ending=ending)


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    """The schedule's runs, oldest first."""
    session = get_open_session()
    try:
        return list(reversed(ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).list_runs(schedule_id)))
    finally:
        session.close()


def _schedule(schedule_id: UUID) -> Schedule:
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).get_schedule(schedule_id)
    finally:
        session.close()


def _poll(schedule_id: UUID) -> list[scheduler.DueSlot]:
    """One poll's due slots for this schedule. Other tests share the database
    and may leave schedules of their own due."""
    session = get_open_session()
    try:
        return [slot for slot in scheduler._poll(ScopedSession(session, SYSTEM_SCOPE)) if slot.schedule_id == schedule_id]
    finally:
        session.close()


def _retry_at(schedule_id: UUID) -> datetime | None:
    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        return scheduler.pending_retry_at(scoped, ScheduleService(scoped).get_schedule(schedule_id))
    finally:
        session.close()


def _age_last_run(schedule_id: UUID, seconds: float) -> None:
    """Move the latest run's finish back in time, as if that long had passed."""
    session = get_open_session()
    try:
        run = _runs(schedule_id)[-1]
        row = session.get(ScheduleRun, run.id)
        row.finished_at = ensure_utc(row.finished_at) - timedelta(seconds=seconds)
        session.add(row)
        session.commit()
    finally:
        session.close()


async def _run(slot: scheduler.DueSlot) -> None:
    await scheduler.execute_schedule(
        slot.schedule_id, slot_at=slot.slot_at, attempt=slot.attempt, retry_of=slot.retry_of,
    )


async def test_a_turn_that_failed_before_acting_is_tried_twice_more_after_60_then_300_seconds(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False

    (first,) = _poll(schedule_id)
    assert (first.attempt, first.retry_of) == (1, None)
    await _run(first)

    (run1,) = _runs(schedule_id)
    assert run1.status == RunStatus.failed
    assert run1.outcome == AttemptOutcome.failed_before_execution
    assert run1.attempt == 1 and run1.retry_of is None
    assert run1.slot_at is None and ensure_utc(run1.for_slot_at) == SLOT
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT, "the slot stays due for the next attempt"
    assert _retry_at(schedule_id) == ensure_utc(run1.finished_at) + timedelta(seconds=60)
    assert _poll(schedule_id) == [], "the next attempt waits out its pause"

    _age_last_run(schedule_id, 61)
    (second,) = _poll(schedule_id)
    assert (second.attempt, second.retry_of, second.slot_at) == (2, run1.id, SLOT)
    await _run(second)
    run2 = _runs(schedule_id)[-1]
    assert (run2.attempt, run2.retry_of) == (2, run1.id)
    assert _retry_at(schedule_id) == ensure_utc(run2.finished_at) + timedelta(seconds=300)

    _age_last_run(schedule_id, 120)
    assert _poll(schedule_id) == [], "the third attempt waits 300 seconds, not 60"
    _age_last_run(schedule_id, 181)
    (third,) = _poll(schedule_id)
    assert (third.attempt, third.retry_of) == (3, run1.id)
    await _run(third)

    assert len(turns.started) == 3
    assert _retry_at(schedule_id) is None
    assert ensure_utc(_schedule(schedule_id).next_run_at) > SLOT, "the last attempt consumes the slot"
    assert ensure_utc(_runs(schedule_id)[-1].slot_at) == SLOT
    assert _poll(schedule_id) == []


async def test_a_retry_that_succeeds_consumes_the_slot(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    (first,) = _poll(schedule_id)
    await _run(first)
    _age_last_run(schedule_id, 61)
    turns.ending.reason, turns.ending.acted = "completed", True
    (second,) = _poll(schedule_id)
    await _run(second)

    run2 = _runs(schedule_id)[-1]
    assert run2.status == RunStatus.success and run2.outcome == AttemptOutcome.completed
    assert ensure_utc(_schedule(schedule_id).next_run_at) > SLOT
    assert _retry_at(schedule_id) is None


async def test_a_turn_whose_agent_acted_before_failing_is_not_tried_again(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", True
    (first,) = _poll(schedule_id)
    await _run(first)

    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.failed_during_execution
    assert ensure_utc(_schedule(schedule_id).next_run_at) > SLOT
    assert _retry_at(schedule_id) is None


@pytest.mark.parametrize(
    ("reason", "outcome"),
    [
        ("completed", AttemptOutcome.completed),
        ("cancelled", AttemptOutcome.cancelled),
        (None, AttemptOutcome.uncertain),
    ],
)
async def test_a_turn_records_its_outcome_and_when_it_was_handed_to_the_agent(schedule_id, turns, reason, outcome):
    turns.ending.reason, turns.ending.acted = reason, True
    before = datetime.now(timezone.utc)
    (first,) = _poll(schedule_id)
    await _run(first)

    (run,) = _runs(schedule_id)
    assert run.outcome == outcome
    assert ensure_utc(run.execution_started_at) >= before
    assert ensure_utc(run.execution_started_at) <= ensure_utc(run.finished_at)
    assert _retry_at(schedule_id) is None


async def test_a_setup_failure_before_the_turn_is_tried_again(schedule_id, turns, monkeypatch):
    import cowork.services.product_permissions as permissions

    async def refused(*_args, **_kwargs):
        raise RuntimeError("could not read the task's settings")

    monkeypatch.setattr(permissions, "require_product_permission", refused)
    (first,) = _poll(schedule_id)
    await _run(first)

    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.failed_before_execution
    assert run.execution_started_at is None
    assert run.attempt == 1
    assert turns.started == []
    assert _retry_at(schedule_id) is not None


async def test_a_passing_outage_before_the_turn_is_not_an_attempt(schedule_id, turns, monkeypatch):
    """A permission service that is down leaves the slot due for the very
    next poll, as before retries, and uses up none of its attempts."""
    import cowork.services.product_permissions as permissions

    async def unavailable(*_args, **_kwargs):
        raise permissions.ProductPermissionUnavailable()

    monkeypatch.setattr(permissions, "require_product_permission", unavailable)
    (first,) = _poll(schedule_id)
    await _run(first)

    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.failed_before_execution
    assert run.attempt is None
    assert _retry_at(schedule_id) is None
    (again,) = _poll(schedule_id)
    assert (again.attempt, again.retry_of) == (1, None)


async def test_a_manual_run_gets_one_attempt(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    await scheduler.execute_schedule(schedule_id, is_manual=True)

    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.failed_before_execution
    assert run.attempt is None and run.for_slot_at is None
    assert _retry_at(schedule_id) is None


async def test_two_replicas_claim_a_retried_slot_once(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    (first,) = _poll(schedule_id)
    await _run(first)
    _age_last_run(schedule_id, 61)

    (seen_by_a,) = _poll(schedule_id)
    (seen_by_b,) = _poll(schedule_id)
    assert seen_by_a == seen_by_b
    turns.ending.reason, turns.ending.acted = "completed", True
    await asyncio.gather(_run(seen_by_a), _run(seen_by_b))

    assert len(turns.started) == 2, "one first attempt and one retry"
    assert [run.attempt for run in _runs(schedule_id)] == [1, 2]


async def test_a_run_whose_process_died_before_its_turn_is_not_an_attempt(schedule_id, turns):
    session = get_open_session()
    try:
        ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(
            schedule_id, slot_at=SLOT, lease_seconds=-60, attempt=1,
        )
    finally:
        session.close()

    (again,) = _poll(schedule_id)
    (lost,) = _runs(schedule_id)
    assert lost.outcome == AttemptOutcome.failed_before_execution
    assert lost.attempt is None
    assert (again.attempt, again.retry_of) == (1, None), "the slot fires again at once, as before"


def test_the_missed_run_pass_leaves_a_slot_that_waits_for_its_retry(turns):
    """An hourly slot 90 minutes old would be skipped as missed, but its
    attempts come first."""
    old_slot = (datetime.now(timezone.utc) - timedelta(minutes=90)).replace(microsecond=0)
    schedule_id = _make_schedule(cadence="hourly", next_run_at=old_slot)
    try:
        session = get_open_session()
        try:
            runs = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
            run = runs.create_run(schedule_id, slot_at=old_slot, attempt=1)
            runs.finish_run(
                run.id, error="failed", status=RunStatus.failed,
                outcome=AttemptOutcome.failed_before_execution, release_slot=True,
            )
        finally:
            session.close()

        assert _poll(schedule_id) == []
        assert ensure_utc(_schedule(schedule_id).next_run_at) == old_slot
    finally:
        _delete_schedule(schedule_id)


async def test_due_slots_over_the_process_limit_stay_due_for_a_later_poll(monkeypatch):
    """With every place taken, the poll loop claims nothing; the slot is still
    due on the next poll."""
    started: list[UUID] = []
    slot = scheduler.DueSlot(schedule_id=uuid4(), slot_at=SLOT)
    polls: list[int] = []

    async def poll(_fn, *, scope):
        polls.append(1)
        return [slot]

    async def execute_schedule(schedule_id, **_kwargs):
        started.append(schedule_id)

    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(scheduler, "run_db", poll)
    monkeypatch.setattr(scheduler, "execute_schedule", execute_schedule)
    for _ in range(scheduler._MAX_CONCURRENT_RUNS):
        assert scheduler.reserve_run_place()

    scheduler.start_scheduler()
    try:
        async with asyncio.timeout(2):
            while len(polls) < 3:
                await asyncio.sleep(0)
        assert started == []

        scheduler.release_run_place()
        async with asyncio.timeout(2):
            while not started:
                await asyncio.sleep(0)
        assert started[0] == slot.schedule_id
    finally:
        scheduler.stop_scheduler()


def test_a_run_place_is_given_back_when_its_run_ends():
    admission = scheduler._Admission(1)
    assert admission.reserve()
    assert not admission.reserve()
    admission.release()
    assert admission.reserve()


# --- Run now ---------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    from cowork.server import create_app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


async def _run_now(schedule_id: UUID) -> httpx.Response:
    async with _client() as client:
        return await client.post(f"/api/v1/schedules/{schedule_id}/run-now")


async def test_run_now_is_refused_while_the_task_is_running(schedule_id):
    session = get_open_session()
    try:
        ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(schedule_id, lease_seconds=60)
    finally:
        session.close()

    answer = await _run_now(schedule_id)
    assert answer.status_code == 409
    assert "already running" in answer.json()["detail"]


async def test_two_run_now_presses_that_both_pass_the_check_start_one_run(schedule_id, monkeypatch):
    """The check before the claim can pass for two presses at once; the run
    row is the claim, and the schedule allows one `running` run, so the press
    that loses the insert is refused."""
    session = get_open_session()
    try:
        held = scheduler.claim_manual_run(ScopedSession(session, SYSTEM_SCOPE), schedule_id)
    finally:
        session.close()
    monkeypatch.setattr(ScheduleRunService, "has_active_run", lambda self, schedule_id: False)

    answer = await _run_now(schedule_id)
    assert answer.status_code == 409
    assert "already running" in answer.json()["detail"]
    assert [run.id for run in _runs(schedule_id) if run.status == RunStatus.running] == [held]
    assert scheduler._admission.held == 0


def test_concurrent_claims_for_one_schedule_start_one_run(schedule_id):
    import threading

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim() -> None:
        session = get_open_session()
        try:
            barrier.wait()
            scheduler.claim_manual_run(ScopedSession(session, SYSTEM_SCOPE), schedule_id)
            outcomes.append("claimed")
        except scheduler.ScheduleBusy:
            outcomes.append("refused")
        finally:
            session.close()

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["claimed", "refused"]
    assert len([run for run in _runs(schedule_id) if run.status == RunStatus.running]) == 1


async def test_a_cron_claim_while_a_run_now_holds_the_schedule_leaves_its_slot_due(schedule_id, turns):
    session = get_open_session()
    try:
        scheduler.claim_manual_run(ScopedSession(session, SYSTEM_SCOPE), schedule_id)
    finally:
        session.close()

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)

    assert turns.started == []
    (run,) = _runs(schedule_id)
    assert run.is_manual
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT, "the slot stays due"


async def test_run_now_runs_the_run_its_request_claimed(schedule_id, turns):
    answer = await _run_now(schedule_id)
    assert answer.status_code == 202, answer.text
    (run,) = _runs(schedule_id)
    assert run.is_manual and run.status == RunStatus.success
    assert str(run.conversation_id) == answer.json()["conversation_id"]


async def test_run_now_is_refused_while_a_retry_waits_and_the_task_reports_it(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    (first,) = _poll(schedule_id)
    await _run(first)

    answer = await _run_now(schedule_id)
    assert answer.status_code == 409
    assert "tried again" in answer.json()["detail"]

    async with _client() as client:
        shown = (await client.get(f"/api/v1/schedules/{schedule_id}")).json()
    assert shown["retryAt"] is not None
    async with _client() as client:
        (run,) = (await client.get(f"/api/v1/schedules/{schedule_id}/runs")).json()["runs"]
    assert run["outcome"] == "failed_before_execution"
    assert run["attempt"] == 1
    assert run["executionStartedAt"] is not None


async def test_run_now_is_refused_when_the_process_is_at_its_limit(schedule_id):
    for _ in range(scheduler._MAX_CONCURRENT_RUNS):
        assert scheduler.reserve_run_place()

    answer = await _run_now(schedule_id)
    assert answer.status_code == 409
    assert "Too many" in answer.json()["detail"]
    assert _runs(schedule_id) == []


async def test_run_now_gives_its_place_back_when_the_run_ends(schedule_id, turns):
    answer = await _run_now(schedule_id)
    assert answer.status_code == 202, answer.text
    assert scheduler._admission.held == 0
    (run,) = _runs(schedule_id)
    assert run.is_manual and run.outcome == AttemptOutcome.completed


# --- What counts as the agent acting ----------------------------------------


def _sse(event: str, data: dict | None = None) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data or {})}\n\n"


@pytest.mark.parametrize(
    ("frames", "acted"),
    [
        ([_sse("response.created"), _sse("response.memory_read", {"read": True}), _sse("response.failed")], False),
        ([_sse("response.created"), _sse("response.in_progress", {"phase": "model_wait"}), _sse("response.failed")], False),
        ([_sse("response.created"), _sse("response.in_progress", {"thought_role": "thought.tool"})], True),
        ([_sse("response.created"), _sse("response.output_text.delta", {"delta": "Hi"})], True),
    ],
)
def test_a_turn_acted_when_its_stream_holds_more_than_its_start_and_failure(tmp_path, monkeypatch, frames, acted):
    buffer = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def fill():
        for frame in frames:
            await buffer.append("sse", {"sse": frame})
        await buffer.close("error")

    asyncio.run(fill())
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _cid: SimpleNamespace(buffer=buffer)))
    assert asyncio.run(scheduler._turn_acted("c1")) is acted


def test_a_turn_with_no_buffer_counts_as_having_acted(monkeypatch):
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _cid: None))
    assert asyncio.run(scheduler._turn_acted("c1")) is True


# --- Migration ---------------------------------------------------------------


def test_the_attempts_migration_upgrades_and_downgrades(tmp_path, monkeypatch):
    from alembic import command

    from cowork.db.migrations import _alembic_config

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = f"sqlite:///{tmp_path / 'attempts.db'}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    added = {"for_slot_at", "attempt", "retry_of", "outcome", "execution_started_at"}
    assert added <= {c["name"] for c in inspect(engine).get_columns("schedule_runs")}
    indexes = {i["name"] for i in inspect(engine).get_indexes("schedule_runs")}
    assert {"ix_schedule_runs_for_slot", "uq_schedule_runs_one_running"} <= indexes

    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "2a366c1f4b9f")
    assert not added & {c["name"] for c in inspect(engine).get_columns("schedule_runs")}

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM schedule_runs")).scalar_one() == 0
    assert added <= {c["name"] for c in inspect(engine).get_columns("schedule_runs")}


def test_the_migration_keeps_one_running_run_per_schedule(tmp_path, monkeypatch):
    """Older builds could leave two `running` runs for one schedule; the
    upgrade fails all but the newest, so the one-running index can be built."""
    from alembic import command

    from cowork.db.migrations import _alembic_config

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = f"sqlite:///{tmp_path / 'one-running.db'}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "2a366c1f4b9f")

    schedule_id = uuid4().hex
    older, newer = uuid4().hex, uuid4().hex
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO schedules (id, title, prompt, cadence, timezone, next_run_at, enabled, project_id, model) "
            "VALUES (:id, 't', 'p', 'daily', 'UTC', '2026-06-25 09:00:00', 1, :project, 'default')"
        ), {"id": schedule_id, "project": GENERAL_PROJECT_ID.hex})
        for run_id, started in ((older, "2026-06-25 09:00:00"), (newer, "2026-06-25 09:05:00")):
            connection.execute(text(
                "INSERT INTO schedule_runs (id, schedule_id, started_at, status, is_manual) "
                "VALUES (:id, :schedule, :started, 'running', 1)"
            ), {"id": run_id, "schedule": schedule_id, "started": started})

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    with engine.connect() as connection:
        statuses = dict(connection.execute(text("SELECT id, status FROM schedule_runs")).all())
    assert statuses == {older: "failed", newer: "running"}

