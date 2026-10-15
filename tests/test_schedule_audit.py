"""The schedule audit trail: immutable settings versions, lifecycle events,
kept history, and an indexed due-scan.

Each edit to what a schedule runs writes one version; an occurrence is
claimed against the current one and every attempt at it runs that version.
Run history is never deleted by the scheduler.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import (
    Schedule,
    ScheduleEvent,
    ScheduleOccurrence,
    ScheduleRun,
    ScheduleVersion,
)
from cowork.schemas.schedules import AttemptOutcome, OccurrenceStatus, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import (
    MAX_FAILED_OCCURRENCES,
    ScheduleOccurrenceService,
    ScheduleRunService,
    ScheduleService,
)

POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")


@pytest.fixture(params=["sqlite", "postgres"])
def engine(request, tmp_path):
    if request.param == "postgres":
        if not POSTGRES_URI:
            pytest.skip("set COWORK_TEST_POSTGRES_URI to a disposable database")
        engine = create_engine(POSTGRES_URI, pool_size=10)
        SQLModel.metadata.drop_all(engine)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}", connect_args={"timeout": 30})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        session.commit()
    yield engine
    if request.param == "postgres":
        SQLModel.metadata.drop_all(engine)
    engine.dispose()


def _scoped(session):
    return ScopedSession(session, SYSTEM_SCOPE)


def _create(engine, **overrides) -> UUID:
    with Session(engine) as session:
        fields = dict(
            title="Digest", prompt="Summarize", cadence="daily",
            next_run_at=datetime.now(timezone.utc) + timedelta(hours=1), model="default",
        )
        fields.update(overrides)
        return ScheduleService(_scoped(session)).create_schedule(**fields).id


def _versions(engine, schedule_id: UUID) -> list[ScheduleVersion]:
    with Session(engine) as session:
        return list(session.exec(
            select(ScheduleVersion).where(ScheduleVersion.schedule_id == schedule_id).order_by(ScheduleVersion.number)
        ).all())


def _kinds(engine, schedule_id: UUID) -> list[str]:
    with Session(engine) as session:
        return [
            e.kind for e in session.exec(
                select(ScheduleEvent).where(ScheduleEvent.schedule_id == schedule_id).order_by(ScheduleEvent.created_at)
            ).all()
        ]


def _claim(engine, schedule_id: UUID, key: str) -> UUID:
    import cowork.scheduler as scheduler

    with Session(engine) as session:
        taken = len(session.exec(
            select(ScheduleOccurrence).where(ScheduleOccurrence.schedule_id == schedule_id)
        ).all())
        schedule = session.get(Schedule, schedule_id)
        # Each claim its own slot, so its key is new.
        schedule.next_run_at = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc) + timedelta(days=taken)
        schedule.enabled = True
        session.add(schedule)
        session.commit()
        occurrence_id = scheduler._claim_slot(_scoped(session), session.get(Schedule, schedule_id))
        assert occurrence_id is not None, key
        return occurrence_id


# --- Versions ---------------------------------------------------------------


def test_a_new_schedule_starts_at_version_one(engine):
    schedule_id = _create(engine)

    [version] = _versions(engine, schedule_id)
    assert (version.number, version.prompt, version.model) == (1, "Summarize", "default")


def test_only_an_edit_to_what_runs_makes_a_version(engine):
    schedule_id = _create(engine)
    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        service.update_schedule(schedule_id, title="Renamed")
        service.update_schedule(schedule_id, next_run_at=datetime.now(timezone.utc) + timedelta(days=2))
        service.pause_schedule(schedule_id)
        service.resume_schedule(schedule_id)
        assert len(_versions(engine, schedule_id)) == 1
        service.update_schedule(schedule_id, prompt="Summarize tersely")
        # Saving the same value again is not a change.
        service.update_schedule(schedule_id, prompt="Summarize tersely")

    assert [(v.number, v.prompt) for v in _versions(engine, schedule_id)] == [
        (1, "Summarize"), (2, "Summarize tersely"),
    ]


def test_a_thousand_occurrences_without_an_edit_share_one_version_and_copy_nothing(engine):
    schedule_id = _create(engine)
    [version] = _versions(engine, schedule_id)
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        for n in range(1000):
            occurrence = service.claim(
                schedule_id, occurrence_key=f"slot:{n}", trigger_source="cron",
                scheduled_for=datetime.now(timezone.utc), max_attempts=1, lease_seconds=60,
                version_id=version.id,
            )
            service.settle(occurrence.id, OccurrenceStatus.success)
            session.commit()

    assert len(_versions(engine, schedule_id)) == 1
    with Session(engine) as session:
        assert {o.version_id for o in session.exec(select(ScheduleOccurrence)).all()} == {version.id}
    # Neither an occurrence nor an attempt has anywhere to copy the prompt to.
    for table in (ScheduleOccurrence, ScheduleRun):
        assert not {"prompt", "model", "project_id"} & set(table.__table__.columns.keys())


def test_an_edit_after_the_claim_does_not_change_what_the_occurrence_runs(engine):
    """The prompt is edited after the occurrence is claimed and after its
    first attempt failed before acting. The retry still runs the claimed
    version; the next occurrence runs the edit."""
    import cowork.scheduler as scheduler

    schedule_id = _create(engine)
    occurrence_id = _claim(engine, schedule_id, "first")
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        ScheduleService(_scoped(session)).update_schedule(schedule_id, prompt="Edited", model="pinned")
        service.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
            error="boom", retry_delay=0,
        )
        session.commit()

    with Session(engine) as session:
        retried = scheduler._read_schedule(_scoped(session), schedule_id=schedule_id, occurrence_id=occurrence_id)
    assert (retried.prompt, retried.model, retried.version) == ("Summarize", "default", 1)

    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        service.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.completed, error=None, retry_delay=None,
        )
        session.commit()
    following = _claim(engine, schedule_id, "second")
    with Session(engine) as session:
        nxt = scheduler._read_schedule(_scoped(session), schedule_id=schedule_id, occurrence_id=following)
    assert (nxt.prompt, nxt.model, nxt.version) == ("Edited", "pinned", 2)


def test_an_edit_racing_a_claim_yields_one_whole_version(engine):
    """Edits and claims on separate connections, released together: every
    occurrence references a version that existed whole, never a mix."""
    import cowork.scheduler as scheduler

    schedule_id = _create(engine)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    claimed: list[UUID] = []

    def edit():
        try:
            for n in range(10):
                with Session(engine) as session:
                    if n == 0:
                        barrier.wait()
                    ScheduleService(_scoped(session)).update_schedule(
                        schedule_id, prompt=f"Prompt {n}", model=f"model-{n}",
                    )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    def claim():
        try:
            for n in range(10):
                with Session(engine) as session:
                    if n == 0:
                        barrier.wait()
                    occurrence_id = scheduler.claim_manual_run(_scoped(session), schedule_id)
                    if occurrence_id is not None:
                        claimed.append(occurrence_id)
                        ScheduleOccurrenceService(_scoped(session)).settle(occurrence_id, OccurrenceStatus.success)
                        session.commit()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=edit), threading.Thread(target=claim)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert claimed
    versions = {v.id: v for v in _versions(engine, schedule_id)}
    with Session(engine) as session:
        for occurrence_id in claimed:
            version = versions[session.get(ScheduleOccurrence, occurrence_id).version_id]
            assert version.prompt.removeprefix("Prompt ") == version.model.removeprefix("model-") or (
                version.number == 1
            )


def test_run_history_reports_the_version_each_run_ran(engine):
    schedule_id = _create(engine)
    occurrence_id = _claim(engine, schedule_id, "first")
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        ScheduleService(_scoped(session)).update_schedule(schedule_id, prompt="Edited")
        runs = ScheduleRunService(_scoped(session))
        [run] = runs.list_runs(schedule_id)
        context = runs.run_context([run])

    assert started is not None
    assert context[occurrence_id]["version"] == 1
    assert context[occurrence_id]["trigger"] == "cron"
    with Session(engine) as session:
        assert ScheduleService(_scoped(session)).current_version_number(schedule_id) == 2


def test_versions_hold_no_secrets_only_the_settings(engine):
    assert set(ScheduleVersion.__table__.columns.keys()) == {
        "id", "created_at", "modified_at", "schedule_id", "number",
        "prompt", "model", "project_id", "cadence", "timezone",
    }


def test_deleting_a_schedule_removes_its_versions_occurrences_and_runs(engine):
    from sqlalchemy import event

    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", lambda conn, _r: conn.execute("PRAGMA foreign_keys=ON"))
        engine.dispose()
    schedule_id = _create(engine)
    occurrence_id = _claim(engine, schedule_id, "first")
    with Session(engine) as session:
        ScheduleService(_scoped(session)).update_schedule(schedule_id, prompt="Edited")
        ScheduleOccurrenceService(_scoped(session)).start_attempt(occurrence_id, is_manual=False, lease_seconds=60)

    with Session(engine) as session:
        assert ScheduleService(_scoped(session)).delete_schedule(schedule_id)

    with Session(engine) as session:
        for table in (ScheduleVersion, ScheduleOccurrence, ScheduleRun, ScheduleEvent):
            assert session.exec(select(table)).all() == [], table


# --- Events -----------------------------------------------------------------


def test_lifecycle_events_are_recorded(engine):
    schedule_id = _create(engine)
    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        service.update_schedule(schedule_id, prompt="New prompt")
        service.pause_schedule(schedule_id)
        service.resume_schedule(schedule_id)

    for n in range(MAX_FAILED_OCCURRENCES):
        occurrence_id = _claim(engine, schedule_id, f"slot:{n}")
        with Session(engine) as session:
            service = ScheduleOccurrenceService(_scoped(session))
            started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
            service.end_attempt(
                occurrence_id, started.token, outcome=AttemptOutcome.failed_during_execution,
                error="boom", retry_delay=None,
            )
            session.commit()

    assert _kinds(engine, schedule_id) == ["created", "edited", "paused", "resumed", "auto_paused"]
    with Session(engine) as session:
        events = ScheduleService(_scoped(session)).events(schedule_id)
    edited = next(e for e in events if e.kind == "edited")
    assert edited.detail == {"fields": ["prompt"], "version": 2}
    auto = next(e for e in events if e.kind == "auto_paused")
    assert auto.detail["failed_runs"] == MAX_FAILED_OCCURRENCES


# --- Kept history and the due scan -----------------------------------------


def test_history_older_than_ninety_days_survives_repeated_polls(engine, monkeypatch):
    import cowork.scheduler as scheduler

    schedule_id = _create(engine)
    old = datetime.now(timezone.utc) - timedelta(days=400)
    with Session(engine) as session:
        occurrence = ScheduleOccurrence(
            schedule_id=schedule_id, occurrence_key="slot:old", trigger_source="cron",
            scheduled_for=old, status=OccurrenceStatus.failed, settled_at=old,
        )
        session.add(occurrence)
        session.flush()
        session.add(ScheduleRun(
            schedule_id=schedule_id, started_at=old, finished_at=old, status=RunStatus.failed,
            occurrence_id=occurrence.id, attempt=1,
        ))
        session.add(ScheduleRun(schedule_id=schedule_id, started_at=old, finished_at=old, status=RunStatus.success))
        session.commit()

    for _ in range(5):
        with Session(engine) as session:
            scheduler._poll(_scoped(session))

    with Session(engine) as session:
        assert len(session.exec(select(ScheduleRun)).all()) == 2
        assert len(session.exec(select(ScheduleOccurrence)).all()) == 1
    assert not hasattr(scheduler, "_maybe_prune")


def test_the_due_scan_uses_the_enabled_next_run_index(engine):
    _create(engine)
    if engine.dialect.name == "sqlite":
        query = "EXPLAIN QUERY PLAN SELECT * FROM schedules WHERE enabled = 1 AND next_run_at <= '2030-01-01'"
    else:
        query = (
            "SET LOCAL enable_seqscan = off; "
            "EXPLAIN SELECT * FROM schedules WHERE enabled = true AND next_run_at <= '2030-01-01'"
        )
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            plan = conn.execute(text(query)).fetchall()
        else:
            conn.execute(text("SET LOCAL enable_seqscan = off"))
            plan = conn.execute(text(query.split("; ", 1)[1])).fetchall()
    assert any("ix_schedules_enabled_next_run_at" in str(row) for row in plan), plan


def test_only_due_schedules_are_returned(engine):
    now = datetime.now(timezone.utc)
    due = _create(engine, title="due", next_run_at=now + timedelta(minutes=5))
    _create(engine, title="later", next_run_at=now + timedelta(days=1))
    paused = _create(engine, title="paused", next_run_at=now + timedelta(minutes=5))
    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(paused)
        found = ScheduleService(_scoped(session)).due(now + timedelta(minutes=10))
    assert [s.id for s in found] == [due]


def test_a_schedule_a_year_behind_is_cheap_to_check():
    from cowork.schedule_timing import missed_slots

    now = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)
    start = now - timedelta(days=365)
    began = time.perf_counter()
    missed = missed_slots("hourly", start, "UTC", now=now, lookback=timedelta(days=7))
    elapsed = time.perf_counter() - began

    # The start and now are both slots: every slot but the last is passed over.
    assert missed.skipped == 365 * 24
    assert missed.run == now
    assert missed.first_skipped == start
    assert missed.last_skipped == now - timedelta(hours=1)
    assert elapsed < 0.05


def test_the_run_executes_the_version_the_occurrence_was_claimed_against(monkeypatch, tmp_path):
    """Through the executor: the prompt sent to the turn is the claimed
    version's, though the schedule was edited before the attempt started."""
    import cowork.handlers.responses as responses_mod
    import cowork.scheduler as scheduler
    from cowork.db.units import run_db, unit_session

    sent: list[str] = []

    class Handler:
        def __init__(self, **_kwargs):
            pass

        async def handle(self, request):
            sent.append(request.input)

            async def _gen():
                if False:
                    yield

            return _gen()

    async def _evidence(_cid):
        return scheduler._TurnEvidence(reason="completed", acted=True)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", Handler)
    monkeypatch.setattr(scheduler, "_turn_evidence", _evidence)
    monkeypatch.setattr(scheduler, "admission", scheduler._Admission(3))

    with unit_session(scope=SYSTEM_SCOPE) as session:
        schedule_id = ScheduleService(session).create_schedule(
            title="version test", prompt="Original", cadence="daily",
            next_run_at=datetime.now(timezone.utc) + timedelta(hours=1), model="default",
            project_id=GENERAL_PROJECT_ID,
        ).id

    async def main():
        occurrence_id = await run_db(lambda s: scheduler.claim_manual_run(s, schedule_id), scope=SYSTEM_SCOPE)
        await run_db(
            lambda s: ScheduleService(s).update_schedule(schedule_id, prompt="Edited"), scope=SYSTEM_SCOPE,
        )
        await scheduler.run_occurrence(occurrence_id, is_manual=True)

    try:
        asyncio.run(main())
    finally:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            ScheduleService(session).delete_schedule(schedule_id)

    assert sent == ["Original"]
