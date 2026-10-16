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
from cowork.schemas.schedules import AttemptOutcome, OccurrenceStatus, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleOccurrenceService, ScheduleRunService, Settlement

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

    assert first is not None and first.number == 1
    assert second is None
    assert [r.attempt for r in _runs(engine, schedule_id)] == [1]


# --- Retries ---------------------------------------------------------------


def test_a_failed_attempt_waits_then_retries_until_its_attempts_are_spent(tmp_path):
    """Only a failure before execution is retried, and only as often as the
    occurrence allows."""
    from cowork.scheduler import _RETRY_DELAYS_SECONDS, _retry_delay

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")

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
            started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
            assert started is not None and started.number == attempt
            before = datetime.now(timezone.utc)
            settlement = occurrences.end_attempt(
                occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
                error="boom", retry_delay=_retry_delay(attempt),
            )
            session.commit()
            assert settlement == (Settlement.retry_scheduled if attempt < 3 else Settlement.settled)
        with Session(engine) as session:
            occurrence = session.get(ScheduleOccurrence, occurrence_id)
            if occurrence.next_attempt_at is not None:
                delays.append((occurrence.next_attempt_at.replace(tzinfo=timezone.utc) - before).total_seconds())

    [settled] = _occurrences(engine, schedule_id)
    assert settled.status == OccurrenceStatus.failed
    assert settled.attempts == 3
    assert settled.error == "boom"
    assert [round(d) for d in delays] == list(_RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize(
    ("outcome", "retry_delay", "expected"),
    [
        (AttemptOutcome.cancelled, 60, OccurrenceStatus.cancelled),
        (AttemptOutcome.failed_before_execution, None, OccurrenceStatus.failed),
        (AttemptOutcome.failed_during_execution, 60, OccurrenceStatus.failed),
        (AttemptOutcome.interrupted, 60, OccurrenceStatus.failed),
        (AttemptOutcome.uncertain, 60, OccurrenceStatus.failed),
        (AttemptOutcome.completed, 60, OccurrenceStatus.success),
    ],
)
def test_attempts_that_should_not_retry_settle_at_once(tmp_path, outcome, retry_delay, expected):
    """A run the user stopped, one that may have acted, or one that will fail
    the same way every time, settles on its first attempt."""
    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        assert occurrences.end_attempt(
            occurrence_id, started.token, outcome=outcome, error="x", retry_delay=retry_delay,
        ) == Settlement.settled
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
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        occurrences.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
            error="boom", retry_delay=60,
        )
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


@pytest.mark.parametrize("executed", [False, True])
def test_the_poll_reaps_an_attempt_whose_lease_ran_out(tmp_path, executed):
    """The process running an attempt died. Its run is recorded failed
    rather than blocking the schedule until a restart. An attempt that never
    reached the agent is tried again; one that did may have acted, so its
    occurrence settles without a replay."""
    from cowork.scheduler import _LEASE_EXPIRED_ERROR, _poll

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        if executed:
            assert occurrences.begin_execution(started, occurrence_id, lease_seconds=-1)
        else:
            occurrences.renew(occurrence_id, started.token, lease_seconds=-1)

    with Session(engine) as session:
        _poll(_scoped(session))

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.lease_expires_at is None
    if executed:
        assert occurrence.status == OccurrenceStatus.failed
        assert occurrence.next_attempt_at is None
    else:
        assert occurrence.status == OccurrenceStatus.active
        assert occurrence.next_attempt_at is not None
    [reaped] = _runs(engine, schedule_id)
    assert reaped.status == RunStatus.failed
    assert reaped.error == _LEASE_EXPIRED_ERROR
    assert reaped.outcome == (
        AttemptOutcome.interrupted if executed else AttemptOutcome.failed_before_execution
    )


def test_the_poll_leaves_an_attempt_this_process_is_running(tmp_path, monkeypatch):
    import cowork.scheduler as scheduler

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        ScheduleOccurrenceService(_scoped(session)).start_attempt(occurrence_id, is_manual=False, lease_seconds=-1)
    monkeypatch.setattr(scheduler, "admission", scheduler._Admission(3))
    scheduler.admission.admit(occurrence_id)

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

    # The first turn fails without having acted, so it may be tried again.
    reasons = iter(["error", "completed"])

    async def _evidence(_conversation_id):
        return scheduler_mod._TurnEvidence(reason=next(reasons), acted=False)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)
    monkeypatch.setattr(scheduler_mod, "_turn_evidence", _evidence)

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
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        assert _serialize(session.get(Schedule, schedule_id), scoped)["retryAt"] is None
        occurrences.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
            error="boom", retry_delay=60,
        )
        session.commit()
        ScheduleRunService(scoped).finish_run(started.run_id, error="boom")

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
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        occurrences.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
            error="boom", retry_delay=60,
        )
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


# --- Boot reaping by tenancy mode ---------------------------------------------


def test_boot_reaps_everything_on_the_desktop_but_only_stale_legacy_runs_in_org_mode(tmp_path):
    """On the desktop the starting server is the only process. In org mode
    another replica may still be running a run: only a run from before
    occurrences that is older than a lease is reaped. An occurrence's run is
    live for as long as its lease is renewed, so the lease reaper owns it."""
    from cowork.scheduler import _LEASE_SECONDS, boot_reap_cutoff

    now = datetime.now(timezone.utc)
    assert boot_reap_cutoff(False, now) is None
    assert boot_reap_cutoff(True, now) == now - timedelta(seconds=_LEASE_SECONDS)

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        runs = ScheduleRunService(_scoped(session))
        stale = runs.create_run(schedule_id)
        stale.started_at = now - timedelta(seconds=_LEASE_SECONDS + 60)
        session.add(stale)
        session.commit()
        live_id = runs.create_run(schedule_id).id
        stale_id = stale.id
        held = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
        old_attempt = session.get(ScheduleRun, held.run_id)
        old_attempt.started_at = now - timedelta(days=1)
        session.add(old_attempt)
        session.commit()

    with Session(engine) as session:
        assert ScheduleRunService(_scoped(session)).reap_orphaned_runs(
            started_before=boot_reap_cutoff(True, now)
        ) == 1
    with Session(engine) as session:
        assert session.get(ScheduleRun, stale_id).status == RunStatus.failed
        assert session.get(ScheduleRun, live_id).status == RunStatus.running
        assert session.get(ScheduleRun, held.run_id).status == RunStatus.running
        assert session.get(ScheduleOccurrence, occurrence_id).status == OccurrenceStatus.active


# --- Abandoned turns and the per-organization cap --------------------------


@pytest.mark.parametrize("executed", [False, True])
def test_the_poll_reports_a_reaped_turn_that_may_still_be_running(tmp_path, executed):
    """A reaped attempt that reached the agent may still be running in a pod
    whose replica died; the poll names its conversation so that pod is told
    to stop. One that never reached the agent has nothing to stop."""
    from uuid import uuid4

    from cowork.scheduler import _poll

    engine = _engine(tmp_path)
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    conversation_id = uuid4()
    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        started = occurrences.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        run = session.get(ScheduleRun, started.run_id)
        run.conversation_id = conversation_id
        session.add(run)
        session.commit()
        if executed:
            assert occurrences.begin_execution(started, occurrence_id, lease_seconds=-1)
        else:
            occurrences.renew(occurrence_id, started.token, lease_seconds=-1)

    abandoned: list[str] = []
    with Session(engine) as session:
        _poll(_scoped(session), abandoned=abandoned)

    assert abandoned == ([str(conversation_id)] if executed else [])


def test_cancelling_an_abandoned_turn_flags_its_pod_only_on_the_remote_backend(monkeypatch):
    import cowork.streaming.stop as stop

    cancelled: list[str] = []

    async def fake_cancel(correlation_id, *, stalled=False):
        cancelled.append(correlation_id)

    async def fake_get_turn(conversation_id):
        return {"correlation_id": f"corr-{conversation_id}"} if conversation_id == "known" else None

    monkeypatch.setattr(stop, "request_remote_cancel", fake_cancel)
    monkeypatch.setattr(stop, "get_turn", fake_get_turn)

    monkeypatch.setattr(stop, "remote_backend", lambda: False)
    assert asyncio.run(stop.cancel_conversation_turn("known")) is False

    monkeypatch.setattr(stop, "remote_backend", lambda: True)
    assert asyncio.run(stop.cancel_conversation_turn("unknown")) is False
    assert asyncio.run(stop.cancel_conversation_turn("known")) is True
    assert cancelled == ["corr-known"]


def _org_schedule(engine, org_id: str, title: str) -> UUID:
    with Session(engine) as session:
        schedule = Schedule(
            title=title, prompt="Summarize", cadence="daily",
            next_run_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            model="default", project_id=GENERAL_PROJECT_ID, org_id=org_id, created_by="user-1",
        )
        session.add(schedule)
        session.commit()
        return schedule.id


def test_an_organization_at_its_limit_claims_nothing_more_until_one_settles(tmp_path, monkeypatch):
    """The limit holds across replicas because it is counted in the
    database: past it a due slot stays due, a run-now is refused, and another
    organization is unaffected."""
    import cowork.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_MAX_ORG_ACTIVE_OCCURRENCES", 2)
    engine = _engine(tmp_path)
    busy = [_org_schedule(engine, "org-a", f"a{n}") for n in range(3)]
    other = _org_schedule(engine, "org-b", "b")

    def claim(schedule_id):
        with Session(engine) as session:
            return scheduler._claim_slot(_scoped(session), session.get(Schedule, schedule_id))

    first, second, third = (claim(s) for s in busy)
    assert first is not None and second is not None
    assert third is None
    with Session(engine) as session:
        assert session.get(Schedule, busy[2]).next_run_at < datetime.now(timezone.utc).replace(tzinfo=None)
        with pytest.raises(scheduler.RunCapReached):
            scheduler.claim_manual_run(_scoped(session), busy[2])
    assert claim(other) is not None

    with Session(engine) as session:
        ScheduleOccurrenceService(_scoped(session)).settle(first, OccurrenceStatus.success)
        session.commit()
    assert claim(busy[2]) is not None
