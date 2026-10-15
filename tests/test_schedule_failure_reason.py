"""A failed scheduled run records the reason the turn gave, and a failure
that would repeat unchanged is not retried."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

import cowork.scheduler as scheduler_mod
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.handlers.turn_errors import response_failed_sse
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleOccurrence
from cowork.schemas.schedules import OccurrenceStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleOccurrenceService
from cowork.streaming.buffer import FileStreamBuffer


def _closed_stream(tmp_path, *frames: str, reason: str = "error") -> FileStreamBuffer:
    buf = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def _fill():
        await buf.append("sse", {"sse": "event: response.created"})
        for frame in frames:
            await buf.append("sse", {"sse": frame})
        await buf.close(reason)

    asyncio.run(_fill())
    return buf


def _use(monkeypatch, buf) -> None:
    monkeypatch.setattr(scheduler_mod, "registry", SimpleNamespace(get=lambda cid: SimpleNamespace(buffer=buf)))


def test_the_failure_frame_is_read_from_the_closed_stream(tmp_path, monkeypatch):
    frame = response_failed_sse("That model isn't available.", "model_not_found", request_id="r1")
    _use(monkeypatch, _closed_stream(tmp_path, frame))

    failure = asyncio.run(scheduler_mod._turn_failure("c1"))

    assert failure == scheduler_mod._TurnFailure(message="That model isn't available.", code="model_not_found")


def test_a_stream_without_a_failure_frame_reports_none(tmp_path, monkeypatch):
    _use(monkeypatch, _closed_stream(tmp_path))
    assert asyncio.run(scheduler_mod._turn_failure("c1")) is None


def _schedule_with_attempt(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reason.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        schedule = Schedule(
            title="Digest", prompt="p", cadence="daily", model="default",
            next_run_at=datetime.now(timezone.utc), project_id=GENERAL_PROJECT_ID,
        )
        session.add(schedule)
        session.commit()
        occurrences = ScheduleOccurrenceService(ScopedSession(session, SYSTEM_SCOPE))
        occurrence = occurrences.claim(
            schedule.id, occurrence_key="slot:a", trigger_source="cron",
            scheduled_for=datetime.now(timezone.utc), max_attempts=3, lease_seconds=60,
        )
        occurrences.start_attempt(occurrence.id, is_manual=False, lease_seconds=60)
        return engine, schedule.id, occurrence.id


def _record(engine, schedule_id, occurrence_id, failure):
    with Session(engine) as session:
        return scheduler_mod._record_turn_outcome(
            ScopedSession(session, SYSTEM_SCOPE), schedule_id=schedule_id, reason="error",
            conversation_id=None, occurrence_id=occurrence_id, attempt=1, failure=failure,
        )


def test_a_permanent_failure_is_recorded_with_its_reason_and_not_retried(tmp_path):
    engine, schedule_id, occurrence_id = _schedule_with_attempt(tmp_path)
    failure = scheduler_mod._TurnFailure(message="That model isn't available.", code="model_not_found")

    outcome = _record(engine, schedule_id, occurrence_id, failure)

    assert outcome.error == "That model isn't available."
    with Session(engine) as session:
        assert session.get(Schedule, schedule_id).last_error == "That model isn't available."
        occurrence = session.get(ScheduleOccurrence, occurrence_id)
        assert occurrence.status == OccurrenceStatus.failed
        assert occurrence.next_attempt_at is None


def test_a_transient_failure_keeps_its_reason_and_retries(tmp_path):
    engine, schedule_id, occurrence_id = _schedule_with_attempt(tmp_path)
    failure = scheduler_mod._TurnFailure(message="The provider is overloaded.", code="provider_overloaded")

    _record(engine, schedule_id, occurrence_id, failure)

    with Session(engine) as session:
        occurrence = session.get(ScheduleOccurrence, occurrence_id)
        assert occurrence.status == OccurrenceStatus.active
        assert occurrence.next_attempt_at is not None
        assert occurrence.error == "The provider is overloaded."


def test_without_a_failure_frame_the_generic_reason_is_kept(tmp_path):
    engine, schedule_id, occurrence_id = _schedule_with_attempt(tmp_path)

    outcome = _record(engine, schedule_id, occurrence_id, None)

    assert outcome.error == scheduler_mod._GENERIC_RUN_FAILURE
