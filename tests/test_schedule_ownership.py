"""Who holds a scheduled occurrence, and what a run may be tried again after.

The race tests start their contenders together on separate connections, on a
SQLite file and, when COWORK_TEST_POSTGRES_URI points at a disposable
database, on Postgres, where concurrent writers really do interleave. The
execution tests drive the scheduler with a real detached registry producer.
"""
from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleOccurrence, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, OccurrenceStatus, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleOccurrenceService, Settlement

DUE = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")
CONTENDERS = 8


@pytest.fixture(params=["sqlite", "postgres"])
def engine(request, tmp_path):
    if request.param == "postgres":
        if not POSTGRES_URI:
            pytest.skip("set COWORK_TEST_POSTGRES_URI to a disposable database")
        engine = create_engine(POSTGRES_URI, pool_size=CONTENDERS + 2)
        SQLModel.metadata.drop_all(engine)
    else:
        engine = create_engine(
            f"sqlite:///{tmp_path / 'ownership.db'}", connect_args={"timeout": 30},
        )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        session.commit()
    yield engine
    if request.param == "postgres":
        SQLModel.metadata.drop_all(engine)
    engine.dispose()


def _scoped(session: Session) -> ScopedSession:
    return ScopedSession(session, SYSTEM_SCOPE)


def _schedule(engine) -> UUID:
    with Session(engine) as session:
        schedule = Schedule(
            title="Daily report", prompt="Summarize", cadence="daily", next_run_at=DUE,
            model="default", project_id=GENERAL_PROJECT_ID,
        )
        session.add(schedule)
        session.commit()
        return schedule.id


def _claim(engine, schedule_id: UUID, key: str = "slot:a", lease_seconds: float = 60) -> UUID | None:
    with Session(engine) as session:
        occurrence = ScheduleOccurrenceService(_scoped(session)).claim(
            schedule_id, occurrence_key=key, trigger_source="cron", scheduled_for=DUE,
            max_attempts=3, lease_seconds=lease_seconds,
        )
        return occurrence.id if occurrence else None


def _together(engine, act, contenders: int = CONTENDERS) -> list:
    """Run ``act(service)`` on ``contenders`` threads, each on its own
    session and connection, released at the same moment."""
    barrier = threading.Barrier(contenders)
    results: list = [None] * contenders
    errors: list[BaseException] = []

    def contend(i: int) -> None:
        try:
            with Session(engine) as session:
                service = ScheduleOccurrenceService(_scoped(session))
                session.connection()  # check out before the barrier, not after
                barrier.wait()
                results[i] = act(service)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return results


def _occurrence(engine, occurrence_id: UUID) -> ScheduleOccurrence:
    with Session(engine) as session:
        return session.get(ScheduleOccurrence, occurrence_id)


def _runs(engine, schedule_id: UUID) -> list[ScheduleRun]:
    with Session(engine) as session:
        return list(session.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id).order_by(ScheduleRun.attempt)
        ).all())


def _expire(engine, occurrence_id: UUID) -> None:
    with Session(engine) as session:
        row = session.get(ScheduleOccurrence, occurrence_id)
        row.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(row)
        session.commit()


# --- Races on separate connections ------------------------------------------


def test_concurrent_claims_of_one_slot_make_one_occurrence(engine):
    schedule_id = _schedule(engine)

    def claim(service):
        occurrence = service.claim(
            schedule_id, occurrence_key="slot:a", trigger_source="cron", scheduled_for=DUE,
            max_attempts=3, lease_seconds=60,
        )
        return occurrence.id if occurrence else None

    won = [r for r in _together(engine, claim) if r is not None]

    assert len(won) == 1
    with Session(engine) as session:
        assert len(session.exec(select(ScheduleOccurrence)).all()) == 1


def test_concurrent_claims_of_different_slots_make_one_active_occurrence(engine):
    """A manual press racing a cron slot, or two slots: one chain at a time."""
    schedule_id = _schedule(engine)
    keys = iter(f"slot:{i}" for i in range(CONTENDERS))
    lock = threading.Lock()

    def claim(service):
        with lock:
            key = next(keys)
        occurrence = service.claim(
            schedule_id, occurrence_key=key, trigger_source="cron", scheduled_for=DUE,
            max_attempts=3, lease_seconds=60,
        )
        return occurrence.id if occurrence else None

    assert len([r for r in _together(engine, claim) if r is not None]) == 1


def test_concurrent_starts_make_one_attempt(engine):
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)

    started = [
        r for r in _together(
            engine, lambda s: s.start_attempt(occurrence_id, is_manual=False, lease_seconds=60),
        ) if r is not None
    ]

    assert len(started) == 1
    assert [(r.attempt, r.status) for r in _runs(engine, schedule_id)] == [(1, RunStatus.running)]
    assert _occurrence(engine, occurrence_id).lease_token == started[0].token


def test_concurrent_settlements_of_one_attempt_land_once(engine):
    """The holder's own end racing a duplicate of it: one settlement."""
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        attempt = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )

    def end(service):
        settlement = service.end_attempt(
            occurrence_id, attempt.token, outcome=AttemptOutcome.failed_during_execution,
            error="boom", retry_delay=None,
        )
        service.session.commit()
        return settlement

    results = _together(engine, end)

    assert results.count(Settlement.settled) == 1
    assert results.count(Settlement.lost) == CONTENDERS - 1


def test_a_reaper_racing_a_renewal_leaves_one_consistent_holder(engine):
    """The holder renews at the moment a reaper takes back its lapsed lease.
    Exactly one wins. If the reaper did, the holder learns it before it hands
    its turn to the agent; if the holder did, its run is untouched."""
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        attempt = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
    _expire(engine, occurrence_id)
    with Session(engine) as session:
        [lease] = ScheduleOccurrenceService(_scoped(session)).expired_leases(datetime.now(timezone.utc))
    turn = iter(["reap", "renew"] * CONTENDERS)
    lock = threading.Lock()

    def contend(service):
        with lock:
            role = next(turn)
        if role == "reap":
            return role, service.reap(
                lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 60,
            )
        return role, service.renew(occurrence_id, attempt.token, lease_seconds=60)

    results = _together(engine, contend)
    reaped = [r for role, r in results if role == "reap" and r != Settlement.lost]
    renewed = [r for role, r in results if role == "renew" and r]

    assert len(reaped) + (1 if renewed else 0) == 1, results
    [run] = _runs(engine, schedule_id)
    with Session(engine) as session:
        still_holds = ScheduleOccurrenceService(_scoped(session)).begin_execution(
            attempt, occurrence_id, lease_seconds=60
        )
    if reaped:
        assert run.status == RunStatus.failed
        assert still_holds is False
    else:
        assert run.status == RunStatus.running
        assert still_holds is True


# --- Interleavings a stale reader must not win ------------------------------


def test_a_reaper_that_read_a_lapsed_claim_cannot_fail_the_attempt_started_since(engine):
    """The reaper lists a claim whose lease lapsed; before it acts, a poller
    starts the claim's first attempt. The reaper must neither fail that live
    attempt nor hand the occurrence on for a second one."""
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    _expire(engine, occurrence_id)
    with Session(engine) as reaper_session:
        reaper = ScheduleOccurrenceService(_scoped(reaper_session))
        [lease] = reaper.expired_leases(datetime.now(timezone.utc))

        with Session(engine) as owner_session:
            started = ScheduleOccurrenceService(_scoped(owner_session)).start_attempt(
                occurrence_id, is_manual=False, lease_seconds=60
            )
        assert started is not None

        assert reaper.reap(
            lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0,
        ) == Settlement.lost

    [run] = _runs(engine, schedule_id)
    assert run.status == RunStatus.running
    occurrence = _occurrence(engine, occurrence_id)
    assert occurrence.status == OccurrenceStatus.active
    assert occurrence.lease_token == started.token
    assert occurrence.next_attempt_at is None
    with Session(engine) as session:
        assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        ) is None


def test_a_reaper_that_read_a_lapsed_lease_cannot_take_it_after_a_renewal(engine):
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        attempt = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
    _expire(engine, occurrence_id)
    with Session(engine) as reaper_session:
        reaper = ScheduleOccurrenceService(_scoped(reaper_session))
        [lease] = reaper.expired_leases(datetime.now(timezone.utc))
        with Session(engine) as owner_session:
            assert ScheduleOccurrenceService(_scoped(owner_session)).renew(
                occurrence_id, attempt.token, lease_seconds=60
            )
        assert reaper.reap(
            lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0,
        ) == Settlement.lost

    [run] = _runs(engine, schedule_id)
    assert run.status == RunStatus.running


def test_a_holder_that_was_reaped_cannot_settle_its_successor(engine):
    """A holder out of touch past its lease is reaped before it acted, and a
    second attempt starts. When the first holder finally reports, nothing it
    writes lands: not on the occurrence, which the second holds, and not on
    its run, which the reaper already ended."""
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        first = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
    _expire(engine, occurrence_id)
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        [lease] = service.expired_leases(datetime.now(timezone.utc))
        assert service.reap(
            lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0,
        ) == Settlement.retry_scheduled
    with Session(engine) as session:
        second = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
    assert second is not None and second.number == 2

    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        assert service.begin_execution(first, occurrence_id, lease_seconds=60) is False
        assert service.end_attempt(
            occurrence_id, first.token, outcome=AttemptOutcome.completed, error=None, retry_delay=None,
        ) == Settlement.lost
        session.commit()

    occurrence = _occurrence(engine, occurrence_id)
    assert occurrence.status == OccurrenceStatus.active
    assert occurrence.lease_token == second.token
    runs = _runs(engine, schedule_id)
    assert [(r.attempt, r.status) for r in runs] == [(1, RunStatus.failed), (2, RunStatus.running)]


def test_an_attempt_settles_once_however_often_it_reports(engine):
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        attempt = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )
    for expected in (Settlement.settled, Settlement.lost):
        with Session(engine) as session:
            service = ScheduleOccurrenceService(_scoped(session))
            assert service.end_attempt(
                occurrence_id, attempt.token, outcome=AttemptOutcome.failed_during_execution,
                error="boom", retry_delay=60,
            ) == expected
            session.commit()
    assert _occurrence(engine, occurrence_id).status == OccurrenceStatus.failed


def test_a_reaped_run_that_had_started_work_is_not_replayed(engine):
    schedule_id = _schedule(engine)
    occurrence_id = _claim(engine, schedule_id)
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        attempt = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        assert service.begin_execution(attempt, occurrence_id, lease_seconds=60)
    _expire(engine, occurrence_id)
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        [lease] = service.expired_leases(datetime.now(timezone.utc))
        assert service.reap(
            lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0,
        ) == Settlement.settled

    occurrence = _occurrence(engine, occurrence_id)
    assert (occurrence.status, occurrence.next_attempt_at) == (OccurrenceStatus.failed, None)
    [run] = _runs(engine, schedule_id)
    assert run.outcome == AttemptOutcome.interrupted


def test_a_reaped_manual_claim_settles_rather_than_waiting_for_a_retry_no_one_starts(engine):
    schedule_id = _schedule(engine)
    with Session(engine) as session:
        occurrence_id = ScheduleOccurrenceService(_scoped(session)).claim(
            schedule_id, occurrence_key="manual:x", trigger_source="manual",
            scheduled_for=DUE, max_attempts=1, lease_seconds=-1,
        ).id
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        [lease] = service.expired_leases(datetime.now(timezone.utc))
        assert service.reap(
            lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0,
        ) == Settlement.settled
    assert _occurrence(engine, occurrence_id).status == OccurrenceStatus.failed


# --- What the turn's stream says about whether it acted ---------------------


def _sse(event: str) -> dict:
    return {"sse": f"event: {event}\ndata: {{}}\n\n"}


@pytest.mark.parametrize(
    ("events", "reason", "outcome"),
    [
        (["response.created", "response.in_progress", "response.failed"], "error",
         AttemptOutcome.failed_before_execution),
        (["response.created", "response.output_text.delta", "response.failed"], "error",
         AttemptOutcome.failed_during_execution),
        (["response.created", "response.output_item.added", "response.failed"], "error",
         AttemptOutcome.failed_during_execution),
        (["response.created", "response.output_text.delta", "response.completed"], "completed",
         AttemptOutcome.completed),
        (["response.created"], "cancelled", AttemptOutcome.cancelled),
        (["response.created"], "interrupted", AttemptOutcome.interrupted),
    ],
)
def test_a_turn_is_classified_by_what_its_stream_shows(tmp_path, monkeypatch, events, reason, outcome):
    from types import SimpleNamespace

    import cowork.scheduler as scheduler
    from cowork.streaming.buffer import FileStreamBuffer

    buffer = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def fill():
        for event in events:
            await buffer.append("sse", _sse(event))
        await buffer.close(reason)

    asyncio.run(fill())
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _cid: SimpleNamespace(buffer=buffer)))

    evidence = asyncio.run(scheduler._turn_evidence("c1"))
    end = scheduler._classify(evidence, timed_out=False)

    assert end.outcome == outcome
    assert end.retryable is (outcome == AttemptOutcome.failed_before_execution)


def test_a_turn_with_no_terminal_record_is_uncertain_and_not_retried():
    import cowork.scheduler as scheduler

    end = scheduler._classify(scheduler._TurnEvidence(reason=None, acted=False), timed_out=False)

    assert end.outcome == AttemptOutcome.uncertain
    assert end.retryable is False


def test_a_turn_stopped_for_its_duration_is_interrupted_whatever_it_shows():
    import cowork.scheduler as scheduler

    end = scheduler._classify(scheduler._TurnEvidence(reason="interrupted", acted=False), timed_out=True)

    assert end.outcome == AttemptOutcome.interrupted
    assert end.retryable is False


# --- Through the scheduler, with a real detached producer -------------------


class _Producer:
    """A turn behind the real registry: it writes ``events`` (and counts one
    external effect when it gets past its start), then ends with ``reason``.
    ``endless`` keeps it writing progress until it is stopped, and
    ``teardown`` is how long its stop takes."""

    def __init__(self, streams_dir, *, events=(), reason="completed", endless=False, teardown=0.0):
        self.streams_dir = streams_dir
        self.events = events
        self.reason = reason
        self.endless = endless
        self.teardown = teardown
        self.effects = 0
        self.handles = []
        self.last_write: datetime | None = None

    def handler(self):
        producer = self

        class Handler:
            def __init__(self, *, principal=None, interactive=True):
                pass

            async def handle(self, request):
                from _fakes import opens

                from cowork.handlers.responses import sse_from_buffer
                from cowork.streaming.buffer import FileStreamBuffer
                from cowork.streaming.registry import TurnLifecycle, registry

                buffer = FileStreamBuffer(producer.streams_dir / f"{request.conversation}.jsonl")
                lifecycle = TurnLifecycle()
                handle = await registry.start(
                    conversation_id=request.conversation, turn_id=0, open_buffer=opens(buffer),
                    produce=lambda b: producer.run(b, lifecycle), lifecycle=lifecycle,
                )
                producer.handles.append(handle)
                return sse_from_buffer(handle.buffer, 0)

        return Handler

    async def _write(self, buffer, event):
        await buffer.append("sse", _sse(event))
        self.last_write = datetime.now(timezone.utc)

    async def run(self, buffer, lifecycle):
        try:
            await self._write(buffer, "response.created")
            for event in self.events:
                if event == "response.output_text.delta":
                    self.effects += 1
                await self._write(buffer, event)
            while self.endless:
                await asyncio.sleep(0.02)
                await self._write(buffer, "response.output_text.delta")
            await buffer.close(self.reason)
        except asyncio.CancelledError:
            await asyncio.sleep(self.teardown)
            await buffer.close("interrupted" if lifecycle.timed_out else "cancelled")


def _app_schedule(*, cadence="daily") -> UUID:
    from cowork.db.units import unit_session
    from cowork.services.schedules import ScheduleService

    with unit_session(scope=SYSTEM_SCOPE) as session:
        return ScheduleService(session).create_schedule(
            title="ownership test", prompt="do the thing", cadence=cadence, next_run_at=DUE,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id


def _app_state(schedule_id: UUID):
    from cowork.db.session import get_open_session

    session = get_open_session()
    try:
        occurrences = session.exec(
            select(ScheduleOccurrence).where(ScheduleOccurrence.schedule_id == schedule_id)
        ).all()
        runs = session.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id).order_by(ScheduleRun.attempt)
        ).all()
        return list(occurrences), list(runs)
    finally:
        session.close()


def _make_due_now(occurrence_id: UUID) -> None:
    from cowork.db.session import get_open_session

    session = get_open_session()
    row = session.get(ScheduleOccurrence, occurrence_id)
    row.next_attempt_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()
    session.close()


@pytest.fixture
def fast_scheduler(monkeypatch):
    import cowork.scheduler as scheduler

    monkeypatch.setattr(scheduler, "admission", scheduler._Admission(3))
    monkeypatch.setattr(scheduler, "_LEASE_RENEW_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_STOP_GRACE_SECONDS", 2)
    monkeypatch.setattr(scheduler, "_REMOTE_STOP_GRACE_SECONDS", 0)
    return scheduler


def test_an_overrunning_turn_is_stopped_before_its_schedule_is_free_again(
    tmp_path, monkeypatch, fast_scheduler,
):
    """A turn that keeps making progress past the duration bound. The idle
    watchdog never fires; the scheduler stops the producer itself, saved as
    interrupted, and keeps the schedule held until the producer has stopped:
    a run-now in the meantime is refused, and no retry follows."""
    import cowork.handlers.responses as responses_mod

    scheduler = fast_scheduler
    producer = _Producer(tmp_path, endless=True, teardown=0.5)
    monkeypatch.setattr(responses_mod, "ResponsesHandler", producer.handler())
    monkeypatch.setattr(scheduler, "_MAX_RUN_DURATION_SECONDS", 0.3)
    schedule_id = _app_schedule()
    claims_while_stopping: list[UUID | None] = []

    async def main():
        run = asyncio.create_task(scheduler.execute_schedule(schedule_id))
        while not producer.handles or producer.handles[0].lifecycle.timed_out is False:
            await asyncio.sleep(0.01)
        # The producer was told to stop and is still tearing down.
        assert not producer.handles[0].task.done()
        from cowork.db.units import run_db

        claims_while_stopping.append(
            await run_db(lambda s: scheduler.claim_manual_run(s, schedule_id), scope=SYSTEM_SCOPE)
        )
        await run
        return datetime.now(timezone.utc)

    finished = asyncio.run(main())

    assert claims_while_stopping == [None]
    assert producer.handles[0].task.done()
    [occurrence], [run] = _app_state(schedule_id)
    assert occurrence.status == OccurrenceStatus.failed
    assert occurrence.next_attempt_at is None
    assert ensure_utc(occurrence.settled_at) >= producer.last_write
    assert ensure_utc(occurrence.settled_at) <= finished
    assert run.outcome == AttemptOutcome.interrupted
    assert run.stop_requested_at is not None
    assert "may have completed" in run.error


def test_a_turn_that_ignores_its_stop_holds_the_schedule_until_it_ends(tmp_path, monkeypatch, fast_scheduler):
    import cowork.handlers.responses as responses_mod

    scheduler = fast_scheduler
    producer = _Producer(tmp_path, endless=True, teardown=1.0)
    monkeypatch.setattr(responses_mod, "ResponsesHandler", producer.handler())
    monkeypatch.setattr(scheduler, "_MAX_RUN_DURATION_SECONDS", 0.2)
    monkeypatch.setattr(scheduler, "_STOP_GRACE_SECONDS", 0.1)
    schedule_id = _app_schedule()

    asyncio.run(scheduler.execute_schedule(schedule_id))

    [occurrence], [run] = _app_state(schedule_id)
    assert producer.handles[0].task.done()
    assert ensure_utc(occurrence.settled_at) >= producer.last_write
    assert run.outcome == AttemptOutcome.interrupted


def test_a_failure_before_the_agent_acts_is_tried_three_times(tmp_path, monkeypatch, fast_scheduler):
    import cowork.handlers.responses as responses_mod

    scheduler = fast_scheduler
    producer = _Producer(tmp_path, events=["response.in_progress", "response.failed"], reason="error")
    monkeypatch.setattr(responses_mod, "ResponsesHandler", producer.handler())
    schedule_id = _app_schedule()

    asyncio.run(scheduler.execute_schedule(schedule_id))
    for _ in range(2):
        [occurrence], _runs_so_far = _app_state(schedule_id)
        assert occurrence.status == OccurrenceStatus.active
        _make_due_now(occurrence.id)
        asyncio.run(scheduler.run_occurrence(occurrence.id))

    [occurrence], runs = _app_state(schedule_id)
    assert occurrence.status == OccurrenceStatus.failed
    assert [(r.attempt, r.outcome) for r in runs] == [
        (n, AttemptOutcome.failed_before_execution) for n in (1, 2, 3)
    ]
    assert producer.effects == 0


def test_a_failure_after_the_agent_acted_is_not_replayed(tmp_path, monkeypatch, fast_scheduler):
    """The turn did something with an effect outside the app, then failed.
    The effect happened once, the run's output is kept, and no retry runs it
    again."""
    import cowork.handlers.responses as responses_mod

    scheduler = fast_scheduler
    producer = _Producer(
        tmp_path, events=["response.output_text.delta", "response.failed"], reason="error",
    )
    monkeypatch.setattr(responses_mod, "ResponsesHandler", producer.handler())
    schedule_id = _app_schedule()

    asyncio.run(scheduler.execute_schedule(schedule_id))

    [occurrence], [run] = _app_state(schedule_id)
    assert producer.effects == 1
    assert occurrence.status == OccurrenceStatus.failed
    assert occurrence.next_attempt_at is None
    assert run.outcome == AttemptOutcome.failed_during_execution
    assert run.conversation_id is not None


def test_a_setup_failure_that_will_recur_is_not_retried(tmp_path, monkeypatch, fast_scheduler):
    import cowork.handlers.responses as responses_mod
    from cowork.services.product_permissions import ProductPermissionDenied

    scheduler = fast_scheduler

    class Refused:
        def __init__(self, **_kwargs):
            pass

        async def handle(self, request):
            raise ProductPermissionDenied()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", Refused)
    schedule_id = _app_schedule()

    asyncio.run(scheduler.execute_schedule(schedule_id))

    [occurrence], [run] = _app_state(schedule_id)
    assert occurrence.status == OccurrenceStatus.failed
    assert run.outcome == AttemptOutcome.failed_before_execution


async def test_run_now_is_refused_while_this_process_is_at_its_limit(monkeypatch, fast_scheduler):
    import test_turn_pool_connections as turn_pool

    scheduler = fast_scheduler
    for _ in range(3):
        scheduler.admission.admit(uuid4())

    with turn_pool._a_daily_schedule(due=DUE) as schedule_id:
        async with turn_pool._client() as client:
            answered = await client.post(f"/api/v1/schedules/{schedule_id}/run-now")
        occurrences, _ = _app_state(schedule_id)

    assert answered.status_code == 409
    assert "Too many scheduled tasks" in answered.json()["detail"]
    assert occurrences == []
    assert scheduler.admission.free() == 0


def ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def test_a_run_whose_occurrence_is_taken_back_stops_its_turn(tmp_path, monkeypatch, fast_scheduler):
    """The holder lost its lease while its turn was still running: whoever
    took the occurrence decides what runs now, so the turn is stopped rather
    than left running unwatched beside it."""
    import cowork.handlers.responses as responses_mod

    scheduler = fast_scheduler
    producer = _Producer(tmp_path, endless=True)
    monkeypatch.setattr(responses_mod, "ResponsesHandler", producer.handler())
    monkeypatch.setattr(scheduler, "_renew_lease", lambda session, **kwargs: False)
    schedule_id = _app_schedule()

    asyncio.run(asyncio.wait_for(scheduler.execute_schedule(schedule_id), timeout=10))

    [handle] = producer.handles
    assert handle.task.done()
    assert handle.lifecycle.timed_out
    [_occurrence], [run] = _app_state(schedule_id)
    assert run.outcome == AttemptOutcome.interrupted


def test_the_run_bound_clears_the_controllers_own_on_the_remote_backend(monkeypatch):
    """A turn in a pod is already bounded by the controller, start included.
    The run's bound sits above it so a turn that uses its budget ends on the
    controller's terms; in process, the bound is the run's own."""
    import cowork.scheduler as scheduler

    monkeypatch.setattr(scheduler, "remote_backend", lambda: False)
    assert scheduler._run_duration_limit() == scheduler._MAX_RUN_DURATION_SECONDS

    monkeypatch.setattr(scheduler, "remote_backend", lambda: True)
    assert scheduler._run_duration_limit() > scheduler._REMOTE_TURN_BUDGET_SECONDS
    assert "15-minute" in scheduler._duration_error()
