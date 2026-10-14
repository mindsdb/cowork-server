"""Occurrences: a firing is claimed once, and its retries are attempts within it.

Each test names the guarantee it pins. The claim tests run on a SQLite file
database so the two claimants hold separate connections, as two pollers (or
two replicas) do on Postgres.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleOccurrence, ScheduleRun
from cowork.schemas.schedules import OccurrenceStatus, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleOccurrenceService, ScheduleRunService

DUE = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'occurrences.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        session.commit()
    return engine


def _schedule(engine, *, cadence: str = "daily", next_run_at: datetime = DUE, title: str = "Daily report") -> UUID:
    with Session(engine) as session:
        schedule = Schedule(
            title=title,
            prompt="Summarize",
            cadence=cadence,
            next_run_at=next_run_at,
            model="default",
            project_id=GENERAL_PROJECT_ID,
        )
        session.add(schedule)
        session.commit()
        return schedule.id


def _scoped(session: Session) -> ScopedSession:
    return ScopedSession(session, SYSTEM_SCOPE)


def _claim(engine, schedule_id: UUID, key: str) -> UUID | None:
    with Session(engine) as session:
        occurrence = ScheduleOccurrenceService(_scoped(session)).claim(
            schedule_id,
            occurrence_key=key,
            trigger_source="cron",
            scheduled_for=DUE,
            max_attempts=3,
            lease_seconds=60,
        )
        return occurrence.id if occurrence else None


def _occurrences(engine, schedule_id: UUID) -> list[ScheduleOccurrence]:
    with Session(engine) as session:
        return list(
            session.exec(select(ScheduleOccurrence).where(ScheduleOccurrence.schedule_id == schedule_id)).all()
        )


def _runs(engine, schedule_id: UUID) -> list[ScheduleRun]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id).order_by(ScheduleRun.attempt)
            ).all()
        )


# --- The claim -------------------------------------------------------------


def test_the_same_slot_claimed_twice_yields_one_occurrence(tmp_path):
    """Two pollers deliver the same slot: the second insert loses."""
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)

    first = _claim(engine, schedule_id, "slot:2026-06-25T09:00:00+00:00")
    second = _claim(engine, schedule_id, "slot:2026-06-25T09:00:00+00:00")

    assert first is not None
    assert second is None
    assert len(_occurrences(engine, schedule_id)) == 1


def test_a_schedule_holds_one_active_occurrence_at_a_time(tmp_path):
    """A different slot (or a manual press) cannot start a second chain while
    one is active; once it settles, the next claim succeeds."""
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    first = _claim(engine, schedule_id, "slot:a")

    assert _claim(engine, schedule_id, "slot:b") is None

    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        occurrences.settle(first, OccurrenceStatus.success)
        session.commit()
    assert _claim(engine, schedule_id, "slot:b") is not None


def test_a_lost_claim_leaves_the_slot_where_the_winner_put_it(tmp_path):
    """The claim advances next_run_at in the same commit as the insert, so a
    losing claim rolls its advance back instead of skipping a slot."""
    from cowork.scheduler import _claim_slot

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    with Session(engine) as session:
        assert _claim_slot(_scoped(session), session.get(Schedule, schedule_id)) is not None
    with Session(engine) as session:
        advanced = session.get(Schedule, schedule_id).next_run_at

    # Rewind the slot so a second claim targets the same key.
    with Session(engine) as session:
        row = session.get(Schedule, schedule_id)
        row.next_run_at = DUE
        session.add(row)
        session.commit()
    with Session(engine) as session:
        assert _claim_slot(_scoped(session), session.get(Schedule, schedule_id)) is None
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).next_run_at.replace(tzinfo=timezone.utc) == DUE

    assert advanced.replace(tzinfo=timezone.utc) > DUE


def test_one_attempt_starts_even_when_two_pollers_try(tmp_path):
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")

    with Session(engine) as a, Session(engine) as b:
        first = ScheduleOccurrenceService(_scoped(a)).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        second = ScheduleOccurrenceService(_scoped(b)).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)

    assert first is not None and first.attempt == 1
    assert second is None
    assert [r.attempt for r in _runs(engine, schedule_id)] == [1]


# --- Retries ---------------------------------------------------------------


def test_a_failed_attempt_waits_then_retries_until_its_attempts_are_spent(tmp_path):
    from cowork.scheduler import _RETRY_DELAYS_SECONDS, _RunOutcome, _settle_occurrence

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    failed = _RunOutcome(status=RunStatus.failed, error="boom")

    delays = []
    for attempt in (1, 2, 3):
        with Session(engine) as session:
            scoped = _scoped(session)
            occurrences = ScheduleOccurrenceService(scoped)
            occurrence = occurrences.get(occurrence_id)
            # Let the retry come due now rather than waiting out the backoff.
            occurrence.next_attempt_at = datetime.now(timezone.utc)
            session.add(occurrence)
            session.commit()
            run = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
            assert run is not None and run.attempt == attempt
            before = datetime.now(timezone.utc)
            _settle_occurrence(scoped, occurrence_id=occurrence_id, attempt=attempt, outcome=failed, retryable=True)
            session.commit()
            occurrence = occurrences.get(occurrence_id)
            if occurrence.next_attempt_at is not None:
                delays.append((occurrence.next_attempt_at.replace(tzinfo=timezone.utc) - before).total_seconds())

    [settled] = _occurrences(engine, schedule_id)
    assert settled.status == OccurrenceStatus.failed
    assert settled.attempts == 3
    assert settled.error == "boom"
    assert [round(d) for d in delays] == list(_RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize(
    ("outcome_status", "retryable", "expected"),
    [
        (RunStatus.cancelled, True, OccurrenceStatus.cancelled),
        (RunStatus.failed, False, OccurrenceStatus.failed),
        (RunStatus.success, True, OccurrenceStatus.success),
    ],
)
def test_attempts_that_should_not_retry_settle_at_once(tmp_path, outcome_status, retryable, expected):
    """A run the user stopped, or one that will fail the same way every time,
    settles on its first attempt."""
    from cowork.scheduler import _RunOutcome, _settle_occurrence

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        scoped = _scoped(session)
        ScheduleOccurrenceService(scoped).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        _settle_occurrence(
            scoped,
            occurrence_id=occurrence_id,
            attempt=1,
            outcome=_RunOutcome(status=outcome_status, error="x" if outcome_status == RunStatus.failed else None),
            retryable=retryable,
        )
        session.commit()

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == expected
    assert occurrence.next_attempt_at is None


def test_permanent_errors_are_not_retried():
    from cowork.db.scoped import MissingTenantScopeError
    from cowork.scheduler import _is_retryable
    from cowork.services.product_permissions import ProductPermissionDenied

    assert _is_retryable(RuntimeError("Run exceeded max duration")) is True
    assert _is_retryable(ProductPermissionDenied()) is False
    assert _is_retryable(MissingTenantScopeError("no org")) is False
    assert _is_retryable(ValueError("Schedule not found")) is False


# --- The poll --------------------------------------------------------------


def test_the_poll_claims_no_more_than_its_capacity(tmp_path, monkeypatch):
    """Past the concurrency cap a due slot is left due, not consumed."""
    import cowork.scheduler as scheduler

    engine = _engine(tmp_path)
    due = datetime.now(timezone.utc) - timedelta(minutes=1)
    first = _schedule(engine, title="first", next_run_at=due)
    second = _schedule(engine, title="second", next_run_at=due)

    with Session(engine) as session:
        ready = scheduler._poll(_scoped(session), capacity=1)

    assert len(ready) == 1
    claimed = {o.schedule_id for sid in (first, second) for o in _occurrences(engine, sid)}
    [left_due] = {first, second} - claimed
    with Session(engine) as session:
        assert session.get(Schedule, left_due).next_run_at.replace(tzinfo=timezone.utc) == due


def test_the_poll_returns_a_retry_once_it_comes_due(tmp_path):
    from cowork.scheduler import _poll

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        occurrences.fail_attempt(occurrence_id, error="boom", retry_delay=60)
        session.commit()

    with Session(engine) as session:
        assert _poll(_scoped(session)) == []

    with Session(engine) as session:
        occurrence = session.get(ScheduleOccurrence, occurrence_id)
        occurrence.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(occurrence)
        session.commit()
    with Session(engine) as session:
        assert _poll(_scoped(session)) == [occurrence_id]


def test_the_poll_reaps_an_attempt_whose_lease_ran_out(tmp_path):
    """The process running an attempt died: its run is recorded failed and
    the occurrence waits for its next attempt instead of blocking the
    schedule until a restart."""
    from cowork.scheduler import _LEASE_EXPIRED_ERROR, _poll

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        run = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=-1
        )
        assert run is not None

    with Session(engine) as session:
        _poll(_scoped(session))

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == OccurrenceStatus.active
    assert occurrence.next_attempt_at is not None
    assert occurrence.lease_expires_at is None
    [reaped] = _runs(engine, schedule_id)
    assert reaped.status == RunStatus.failed
    assert reaped.error == _LEASE_EXPIRED_ERROR


def test_the_poll_leaves_an_attempt_this_process_is_running(tmp_path, monkeypatch):
    import cowork.scheduler as scheduler

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        ScheduleOccurrenceService(_scoped(session)).start_attempt(occurrence_id, is_manual=False, lease_seconds=-1)
    monkeypatch.setattr(scheduler, "_inflight", {occurrence_id})

    with Session(engine) as session:
        scheduler._poll(_scoped(session))

    [run] = _runs(engine, schedule_id)
    assert run.status == RunStatus.running


def test_boot_reaping_settles_the_occurrence_without_retrying(tmp_path):
    """A run cut off by a quit is not replayed on the next launch."""
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        ScheduleOccurrenceService(_scoped(session)).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)

    with Session(engine) as session:
        assert ScheduleRunService(_scoped(session)).reap_orphaned_runs() == 1

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == OccurrenceStatus.failed


# --- Through execute_schedule and the route ---------------------------------


def test_a_failing_cron_run_schedules_a_retry_and_a_later_attempt_succeeds(monkeypatch):
    """End to end through the executor: the first attempt's turn errors, the
    occurrence waits for its retry, and the second attempt's success settles
    it — two runs, numbered, under one occurrence."""
    import cowork.handlers.responses as responses_mod
    import cowork.scheduler as scheduler_mod
    from cowork.db.session import get_open_session
    from cowork.services.schedules import ScheduleService

    class FakeHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            async def _gen():
                if False:
                    yield

            return _gen()

    reasons = iter(["error", "completed"])

    async def _terminal(_conversation_id):
        return next(reasons)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)
    monkeypatch.setattr(scheduler_mod, "_turn_terminal_reason", _terminal)

    session = get_open_session()
    schedule_id = ScheduleService(_scoped(session)).create_schedule(
        title="retry test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=DUE,
        model="default",
        project_id=GENERAL_PROJECT_ID,
    ).id
    session.close()

    try:
        asyncio.run(scheduler_mod.execute_schedule(schedule_id, is_manual=False))
        s = get_open_session()
        [occurrence] = s.exec(select(ScheduleOccurrence).where(ScheduleOccurrence.schedule_id == schedule_id)).all()
        assert occurrence.status == OccurrenceStatus.active
        assert occurrence.next_attempt_at is not None
        occurrence.next_attempt_at = datetime.now(timezone.utc)
        s.add(occurrence)
        s.commit()
        occurrence_id = occurrence.id
        s.close()

        asyncio.run(scheduler_mod.run_occurrence(occurrence_id, is_manual=False))

        s = get_open_session()
        occurrence = s.get(ScheduleOccurrence, occurrence_id)
        runs = s.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id).order_by(ScheduleRun.attempt)
        ).all()
        assert occurrence.status == OccurrenceStatus.success
        assert [(r.attempt, r.status) for r in runs] == [(1, RunStatus.failed), (2, RunStatus.success)]
        assert {r.occurrence_id for r in runs} == {occurrence_id}
        s.close()
    finally:
        s = get_open_session()
        ScheduleService(_scoped(s)).delete_schedule(schedule_id)
        s.close()


async def test_run_now_refuses_while_the_schedule_has_an_occurrence_in_flight():
    import test_turn_pool_connections as turn_pool

    from cowork.db.session import get_open_session
    from cowork.scheduler import _claim_slot

    with turn_pool._a_daily_schedule(due=DUE) as schedule_id:
        session = get_open_session()
        scoped = _scoped(session)
        assert _claim_slot(scoped, session.get(Schedule, schedule_id)) is not None
        session.close()

        async with turn_pool._client() as client:
            answered = await client.post(f"/api/v1/schedules/{schedule_id}/run-now")

    assert answered.status_code == 409, answered.text


def test_a_schedule_waiting_to_retry_reports_when(tmp_path):
    from cowork.api.v1.endpoints.schedules import _serialize

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        scoped = _scoped(session)
        assert _serialize(session.get(Schedule, schedule_id), scoped)["retryAt"] is None
        occurrences = ScheduleOccurrenceService(scoped)
        run = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        assert _serialize(session.get(Schedule, schedule_id), scoped)["retryAt"] is None
        occurrences.fail_attempt(occurrence_id, error="boom", retry_delay=60)
        session.commit()
        ScheduleRunService(scoped).finish_run(run.id, error="boom")

        data = _serialize(session.get(Schedule, schedule_id), scoped)

    assert data["retryAt"] is not None
    assert data["running"] is False


async def test_run_now_says_so_when_the_schedule_is_waiting_to_retry():
    import test_turn_pool_connections as turn_pool

    from cowork.db.session import get_open_session
    from cowork.scheduler import _claim_slot

    with turn_pool._a_daily_schedule(due=DUE) as schedule_id:
        session = get_open_session()
        scoped = _scoped(session)
        occurrence_id = _claim_slot(scoped, session.get(Schedule, schedule_id))
        occurrences = ScheduleOccurrenceService(scoped)
        occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        occurrences.fail_attempt(occurrence_id, error="boom", retry_delay=60)
        session.commit()
        # The attempt's run would be finished by the executor; settle it here.
        for run in session.exec(select(ScheduleRun).where(ScheduleRun.occurrence_id == occurrence_id)).all():
            run.status = RunStatus.failed
            session.add(run)
        session.commit()
        session.close()

        async with turn_pool._client() as client:
            answered = await client.post(f"/api/v1/schedules/{schedule_id}/run-now")

    assert answered.status_code == 409
    assert "waiting to try again" in answered.json()["detail"]


# --- Missed runs (ENG-943) ---------------------------------------------------


def _set(engine, schedule_id: UUID, **fields) -> None:
    with Session(engine) as session:
        row = session.get(Schedule, schedule_id)
        for name, value in fields.items():
            setattr(row, name, value)
        session.add(row)
        session.commit()


def _skipped_slots(engine, schedule_id: UUID) -> list[datetime]:
    return sorted(
        o.scheduled_for.replace(tzinfo=timezone.utc)
        for o in _occurrences(engine, schedule_id)
        if o.status == OccurrenceStatus.skipped
    )


def _settle_claimed(engine, occurrence_id: UUID, outcome_status: RunStatus = RunStatus.success) -> None:
    """Run an attempt at the occurrence to the given end, as the executor would."""
    from cowork.scheduler import _RunOutcome, _settle_occurrence

    with Session(engine) as session:
        scoped = _scoped(session)
        run = ScheduleOccurrenceService(scoped).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        _settle_occurrence(
            scoped, occurrence_id=occurrence_id, attempt=run.attempt,
            outcome=_RunOutcome(status=outcome_status, error="boom" if outcome_status == RunStatus.failed else None),
            retryable=False,
        )
        session.commit()
        ScheduleRunService(scoped).finish_run(run.id, status=outcome_status)


def test_after_an_outage_the_default_runs_only_the_latest_missed_slot(tmp_path):
    """A weekly task three weeks behind runs once, for its latest slot. The
    three earlier slots are all older than the 7-day lookback, so they are
    counted in missed_runs without a skipped record each."""
    from cowork.scheduler import _poll

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    latest = now - timedelta(hours=1)
    schedule_id = _schedule(engine, cadence="weekly", next_run_at=latest - timedelta(weeks=3))

    with Session(engine) as session:
        ready = _poll(_scoped(session))

    [claimed] = ready
    with Session(engine) as session:
        occurrence = session.get(ScheduleOccurrence, claimed)
        schedule = session.get(Schedule, schedule_id)
        assert occurrence.scheduled_for.replace(tzinfo=timezone.utc) == latest
        assert schedule.next_run_at.replace(tzinfo=timezone.utc) == latest + timedelta(weeks=1)
        assert schedule.missed_runs == 3
    assert _skipped_slots(engine, schedule_id) == []


def test_applying_the_policy_twice_changes_nothing(tmp_path):
    from cowork.scheduler import _handle_missed_runs

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now - timedelta(days=3, hours=1))

    with Session(engine) as session:
        _handle_missed_runs(_scoped(session))
    with Session(engine) as session:
        first = session.get(Schedule, schedule_id)
        snapshot = (first.next_run_at, first.missed_runs)
    with Session(engine) as session:
        _handle_missed_runs(_scoped(session))
    with Session(engine) as session:
        again = session.get(Schedule, schedule_id)
        assert (again.next_run_at, again.missed_runs) == snapshot
    assert len(_skipped_slots(engine, schedule_id)) == 3


def test_bounded_catch_up_runs_at_most_ten_missed_slots_in_turn(tmp_path):
    from cowork.scheduler import _MAX_CATCH_UP, _poll

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    first_missed = now - timedelta(hours=14)
    schedule_id = _schedule(engine, cadence="hourly", next_run_at=first_missed)
    _set(engine, schedule_id, missed_run_policy="bounded_catch_up")

    ran: list[datetime] = []
    for _ in range(20):
        with Session(engine) as session:
            ready = _poll(_scoped(session))
        if not ready:
            break
        [occurrence_id] = ready  # single-flight: one slot at a time
        with Session(engine) as session:
            ran.append(session.get(ScheduleOccurrence, occurrence_id).scheduled_for.replace(tzinfo=timezone.utc))
        _settle_claimed(engine, occurrence_id)

    expected = [now - timedelta(hours=h) for h in range(_MAX_CATCH_UP - 1, -1, -1)]
    assert ran == expected
    assert _skipped_slots(engine, schedule_id) == [first_missed + timedelta(hours=h) for h in range(5)]


def test_skip_runs_nothing_the_app_was_closed_for(tmp_path):
    from cowork.scheduler import _poll

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now - timedelta(days=2, hours=3))
    _set(engine, schedule_id, missed_run_policy="skip")

    with Session(engine) as session:
        assert _poll(_scoped(session)) == []
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).next_run_at.replace(tzinfo=timezone.utc) > now
    assert len(_skipped_slots(engine, schedule_id)) == 3


def test_a_cron_success_does_not_suppress_the_next_slot(tmp_path):
    """The freshness guard is for a manual run just before a slot. A cron run
    that just finished is its own slot and must not skip the next one."""
    from cowork.scheduler import _due_schedules

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="hourly", next_run_at=now - timedelta(seconds=30))
    with Session(engine) as session:
        run_service = ScheduleRunService(_scoped(session))
        run = run_service.create_run(schedule_id, is_manual=False)
        run_service.finish_run(run.id)

    with Session(engine) as session:
        assert [s.id for s in _due_schedules(_scoped(session), now)] == [schedule_id]


# --- Edits --------------------------------------------------------------------


def test_editing_cadence_or_timezone_moves_the_next_slot_at_once(tmp_path):
    from cowork.services.schedules import ScheduleService

    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now + timedelta(hours=1))
    saturday = datetime(2030, 1, 5, 9, 0, tzinfo=timezone.utc)
    assert saturday.weekday() == 5

    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        edited = service.update_schedule(schedule_id, cadence="weekdays", next_run_at=saturday)
        assert edited.next_run_at.replace(tzinfo=timezone.utc) == saturday + timedelta(days=2)

        # A start already in the past moves to the next slot instead of
        # firing as a missed run on the next poll.
        stale = service.update_schedule(schedule_id, cadence="daily", next_run_at=now - timedelta(days=3))
        assert stale.next_run_at.replace(tzinfo=timezone.utc) > now


# --- Auto-pause -------------------------------------------------------------


def _fail_occurrences(engine, schedule_id: UUID, count: int, *, start: int = 0, manual: bool = False) -> None:
    for n in range(start, start + count):
        with Session(engine) as session:
            occurrence = ScheduleOccurrenceService(_scoped(session)).claim(
                schedule_id,
                occurrence_key=f"{'manual' if manual else 'slot'}:{n}",
                trigger_source="manual" if manual else "cron",
                scheduled_for=DUE,
                max_attempts=1,
                lease_seconds=60,
            )
        _settle_claimed(engine, occurrence.id, RunStatus.failed)


def test_three_failed_occurrences_in_a_row_pause_the_schedule(tmp_path):
    from cowork.scheduler import _MAX_FAILED_OCCURRENCES

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)

    _fail_occurrences(engine, schedule_id, _MAX_FAILED_OCCURRENCES - 1)
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).enabled is True

    _fail_occurrences(engine, schedule_id, 1, start=_MAX_FAILED_OCCURRENCES)
    with Session(engine) as session:
        schedule = session.get(Schedule, schedule_id)
        assert schedule.enabled is False
        assert schedule.auto_paused_at is not None
        assert schedule.consecutive_failures == _MAX_FAILED_OCCURRENCES


def test_retries_inside_one_occurrence_count_once(tmp_path):
    """Three failed attempts at one occurrence are one failure, not three."""
    from cowork.scheduler import _RunOutcome, _settle_occurrence

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    for attempt in (1, 2, 3):
        with Session(engine) as session:
            scoped = _scoped(session)
            occurrence = session.get(ScheduleOccurrence, occurrence_id)
            occurrence.next_attempt_at = datetime.now(timezone.utc)
            session.add(occurrence)
            session.commit()
            ScheduleOccurrenceService(scoped).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
            _settle_occurrence(
                scoped, occurrence_id=occurrence_id, attempt=attempt,
                outcome=_RunOutcome(status=RunStatus.failed, error="boom"), retryable=True,
            )
            session.commit()

    with Session(engine) as session:
        schedule = session.get(Schedule, schedule_id)
        assert schedule.consecutive_failures == 1
        assert schedule.enabled is True


def test_a_success_resets_the_streak_and_manual_failures_do_not_count(tmp_path):
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)

    _fail_occurrences(engine, schedule_id, 2)
    _fail_occurrences(engine, schedule_id, 5, start=10, manual=True)
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).consecutive_failures == 2

    ok = _claim(engine, schedule_id, "slot:ok")
    _settle_claimed(engine, ok)
    with Session(engine) as session:
        schedule = session.get(Schedule, schedule_id)
        assert schedule.consecutive_failures == 0
        assert schedule.enabled is True


def test_resume_clears_the_auto_pause_and_keeps_the_history(tmp_path):
    from cowork.scheduler import _MAX_FAILED_OCCURRENCES
    from cowork.services.schedules import ScheduleService

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    _fail_occurrences(engine, schedule_id, _MAX_FAILED_OCCURRENCES)

    with Session(engine) as session:
        resumed = ScheduleService(_scoped(session)).resume_schedule(schedule_id)
        assert resumed.enabled is True
        assert resumed.auto_paused_at is None
        assert resumed.consecutive_failures == 0
        assert resumed.next_run_at.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)

    assert len(_runs(engine, schedule_id)) == _MAX_FAILED_OCCURRENCES
    assert len(_occurrences(engine, schedule_id)) == _MAX_FAILED_OCCURRENCES


# --- Run now on a one-off -----------------------------------------------------


def test_running_a_one_off_now_uses_it_up(tmp_path):
    """Its scheduled slot must not fire a second time after a manual run."""
    from cowork.scheduler import _due_schedules, claim_manual_run

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, cadence="once", next_run_at=datetime.now(timezone.utc) + timedelta(days=1))

    with Session(engine) as session:
        assert claim_manual_run(_scoped(session), schedule_id) is not None
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).enabled is False
        later = datetime.now(timezone.utc) + timedelta(days=2)
        assert _due_schedules(_scoped(session), later) == []


def test_a_refused_run_now_leaves_the_one_off_enabled(tmp_path):
    from cowork.scheduler import claim_manual_run

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, cadence="once", next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    _claim(engine, schedule_id, "slot:busy")

    with Session(engine) as session:
        assert claim_manual_run(_scoped(session), schedule_id) is None
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).enabled is True


def test_running_a_recurring_schedule_now_keeps_its_slot(tmp_path):
    from cowork.scheduler import claim_manual_run

    engine = _engine(tmp_path)
    slot = datetime.now(timezone.utc) + timedelta(hours=3)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=slot)

    with Session(engine) as session:
        assert claim_manual_run(_scoped(session), schedule_id) is not None
    with Session(engine) as session:
        schedule = session.get(Schedule, schedule_id)
        assert schedule.enabled is True
        assert schedule.next_run_at.replace(tzinfo=timezone.utc) == slot
