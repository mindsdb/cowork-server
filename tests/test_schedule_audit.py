"""The schedule audit trail: spec versions on runs, lifecycle events, pruning,
and an indexed due-scan."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import ScheduleEvent, ScheduleOccurrence, ScheduleRun
from cowork.schemas.schedules import OccurrenceStatus, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import (
    ScheduleOccurrenceService,
    ScheduleService,
    prune_schedule_history,
)


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        session.commit()
    return engine


def _scoped(session):
    return ScopedSession(session, SYSTEM_SCOPE)


def _create(engine, **overrides):
    with Session(engine) as session:
        fields = dict(
            title="Digest", prompt="Summarize", cadence="daily",
            next_run_at=datetime.now(timezone.utc) + timedelta(hours=1), model="default",
        )
        fields.update(overrides)
        return ScheduleService(_scoped(session)).create_schedule(**fields).id


def _kinds(engine, schedule_id):
    with Session(engine) as session:
        return [
            e.kind for e in session.exec(
                select(ScheduleEvent).where(ScheduleEvent.schedule_id == schedule_id).order_by(ScheduleEvent.created_at)
            ).all()
        ]


def test_edits_that_change_what_runs_bump_the_spec_version(tmp_path):
    engine = _engine(tmp_path)
    schedule_id = _create(engine)
    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        assert service.update_schedule(schedule_id, title="Renamed").spec_version == 1
        assert service.update_schedule(schedule_id, prompt="Summarize tersely").spec_version == 2
        # Saving the same value again is not a change.
        assert service.update_schedule(schedule_id, prompt="Summarize tersely").spec_version == 2


def test_each_attempt_records_the_spec_version_it_ran(tmp_path):
    engine = _engine(tmp_path)
    schedule_id = _create(engine)
    with Session(engine) as session:
        ScheduleService(_scoped(session)).update_schedule(schedule_id, model="pinned-model")
    with Session(engine) as session:
        occurrences = ScheduleOccurrenceService(_scoped(session))
        occurrence = occurrences.claim(
            schedule_id, occurrence_key="slot:a", trigger_source="cron",
            scheduled_for=datetime.now(timezone.utc), max_attempts=1, lease_seconds=60,
        )
        run = occurrences.start_attempt(occurrence.id, is_manual=False, lease_seconds=60)
        assert run.spec_version == 2


def test_lifecycle_events_are_recorded(tmp_path):
    from cowork.scheduler import _MAX_FAILED_OCCURRENCES, _RunOutcome, _settle_occurrence

    engine = _engine(tmp_path)
    schedule_id = _create(engine)
    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        service.update_schedule(schedule_id, prompt="New prompt")
        service.pause_schedule(schedule_id)
        service.resume_schedule(schedule_id)

    for n in range(_MAX_FAILED_OCCURRENCES):
        with Session(engine) as session:
            scoped = _scoped(session)
            occurrences = ScheduleOccurrenceService(scoped)
            occurrence = occurrences.claim(
                schedule_id, occurrence_key=f"slot:{n}", trigger_source="cron",
                scheduled_for=datetime.now(timezone.utc), max_attempts=1, lease_seconds=60,
            )
            run = occurrences.start_attempt(occurrence.id, is_manual=False, lease_seconds=60)
            _settle_occurrence(
                scoped, occurrence_id=occurrence.id, attempt=run.attempt,
                outcome=_RunOutcome(status=RunStatus.failed, error="boom"), retryable=False,
            )
            session.commit()

    assert _kinds(engine, schedule_id) == ["created", "edited", "paused", "resumed", "auto_paused"]
    with Session(engine) as session:
        events = ScheduleService(_scoped(session)).events(schedule_id)
        auto = next(e for e in events if e.kind == "auto_paused")
        assert auto.detail == {"failed_runs": _MAX_FAILED_OCCURRENCES, "error": "boom"}
        edited = next(e for e in events if e.kind == "edited")
        assert edited.detail == {"fields": ["prompt"], "spec_version": 2}


def test_pruning_drops_old_settled_history_and_keeps_the_rest(tmp_path):
    engine = _engine(tmp_path)
    schedule_id = _create(engine)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=120)
    with Session(engine) as session:
        for key, status, settled in (
            ("slot:old", OccurrenceStatus.failed, old),
            ("slot:recent", OccurrenceStatus.success, now - timedelta(days=1)),
        ):
            occurrence = ScheduleOccurrence(
                schedule_id=schedule_id, occurrence_key=key, trigger_source="cron",
                scheduled_for=settled, status=status, settled_at=settled,
            )
            session.add(occurrence)
            session.flush()
            session.add(ScheduleRun(
                schedule_id=schedule_id, started_at=settled, status=RunStatus.failed,
                occurrence_id=occurrence.id, attempt=1,
            ))
        # A run from before occurrences, finished long ago, and one still running.
        session.add(ScheduleRun(schedule_id=schedule_id, started_at=old, status=RunStatus.success))
        session.add(ScheduleRun(schedule_id=schedule_id, started_at=old, status=RunStatus.running))
        session.commit()

    with Session(engine) as session:
        assert prune_schedule_history(_scoped(session), now - timedelta(days=90)) == 2

    with Session(engine) as session:
        keys = {o.occurrence_key for o in session.exec(select(ScheduleOccurrence)).all()}
        statuses = sorted(r.status for r in session.exec(select(ScheduleRun)).all())
    assert keys == {"slot:recent"}
    assert statuses == [RunStatus.failed, RunStatus.running]


def test_the_due_scan_uses_the_enabled_next_run_index(tmp_path):
    engine = _engine(tmp_path)
    _create(engine)
    with engine.connect() as conn:
        plan = conn.execute(text(
            "EXPLAIN QUERY PLAN SELECT * FROM schedules WHERE enabled = 1 AND next_run_at <= '2030-01-01'"
        )).fetchall()
    assert any("ix_schedules_enabled_next_run_at" in str(row) for row in plan), plan


def test_only_due_schedules_are_returned(tmp_path):
    engine = _engine(tmp_path)
    now = datetime.now(timezone.utc)
    due = _create(engine, title="due", next_run_at=now + timedelta(minutes=5))
    _create(engine, title="later", next_run_at=now + timedelta(days=1))
    paused = _create(engine, title="paused", next_run_at=now + timedelta(minutes=5))
    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(paused)
        found = ScheduleService(_scoped(session)).due(now + timedelta(minutes=10))
    assert [s.id for s in found] == [due]


def test_a_schedule_a_year_behind_is_cheap_to_check():
    from cowork.schedule_timing import overdue_slots

    now = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)
    start = now - timedelta(days=365)
    began = time.perf_counter()
    older, recent, future = overdue_slots("hourly", start, "UTC", now=now, since=now - timedelta(days=7))
    elapsed = time.perf_counter() - began
    assert older + len(recent) == 365 * 24 + 1
    assert len(recent) == 7 * 24 + 1  # both ends of the window are slots
    assert future > now
    assert elapsed < 0.05
