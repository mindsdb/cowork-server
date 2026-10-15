"""A failed scheduled run records the reason the turn gave, and says what is
known of whether it acted. A failure code can rule a retry out; only the
turn's stream showing nothing beyond its start can rule one in."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

import cowork.scheduler as scheduler_mod
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.handlers.turn_errors import response_failed_sse
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleOccurrence, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, OccurrenceStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleOccurrenceService
from cowork.streaming.buffer import FileStreamBuffer


def _closed_stream(tmp_path, *frames: str, reason: str = "error") -> FileStreamBuffer:
    buf = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def _fill():
        await buf.append("sse", {"sse": "event: response.created\ndata: {}\n\n"})
        for frame in frames:
            await buf.append("sse", {"sse": frame})
        await buf.close(reason)

    asyncio.run(_fill())
    return buf


def _evidence(monkeypatch, buf):
    monkeypatch.setattr(scheduler_mod, "registry", SimpleNamespace(get=lambda cid: SimpleNamespace(buffer=buf)))
    return asyncio.run(scheduler_mod._turn_evidence("c1"))


def test_the_failure_frame_is_read_from_the_closed_stream(tmp_path, monkeypatch):
    frame = response_failed_sse("That model isn't available.", "model_not_found", request_id="r1")

    evidence = _evidence(monkeypatch, _closed_stream(tmp_path, frame))

    assert evidence.failure == scheduler_mod._TurnFailure(message="That model isn't available.", code="model_not_found")
    assert evidence.acted is False


def test_a_stream_without_a_failure_frame_reports_none(tmp_path, monkeypatch):
    assert _evidence(monkeypatch, _closed_stream(tmp_path)).failure is None


def test_a_permanent_failure_before_acting_keeps_its_reason_and_is_not_retried(tmp_path, monkeypatch):
    frame = response_failed_sse("That model isn't available.", "model_not_found", request_id="r1")

    end = scheduler_mod._classify(_evidence(monkeypatch, _closed_stream(tmp_path, frame)), timed_out=False)

    assert end.outcome == AttemptOutcome.failed_before_execution
    assert end.error == "That model isn't available."
    assert end.retryable is False


def test_a_transient_failure_before_acting_keeps_its_reason_and_may_retry(tmp_path, monkeypatch):
    frame = response_failed_sse("The provider is overloaded.", "provider_overloaded", request_id="r1")

    end = scheduler_mod._classify(_evidence(monkeypatch, _closed_stream(tmp_path, frame)), timed_out=False)

    assert (end.outcome, end.error, end.retryable) == (
        AttemptOutcome.failed_before_execution, "The provider is overloaded.", True,
    )


def test_a_transient_code_after_the_turn_acted_is_not_retried(tmp_path, monkeypatch):
    """The code says the provider was overloaded; the stream says the agent
    had already answered part of it. What it did may have had effects."""
    acted = "event: response.output_text.delta\ndata: {\"delta\": \"Sent the\"}\n\n"
    frame = response_failed_sse("The provider is overloaded.", "provider_overloaded", request_id="r1")

    end = scheduler_mod._classify(_evidence(monkeypatch, _closed_stream(tmp_path, acted, frame)), timed_out=False)

    assert (end.outcome, end.error, end.retryable) == (
        AttemptOutcome.failed_during_execution, "The provider is overloaded.", False,
    )


def test_without_a_failure_frame_the_generic_reason_is_kept(tmp_path, monkeypatch):
    end = scheduler_mod._classify(_evidence(monkeypatch, _closed_stream(tmp_path)), timed_out=False)

    assert end.error == scheduler_mod._FAILED_BEFORE_ERROR


def test_an_unreadable_stream_is_uncertain_not_a_success(monkeypatch):
    class Broken:
        is_closed = True

        async def tail(self, _from):
            raise OSError("gone")
            yield  # pragma: no cover

    evidence = _evidence(monkeypatch, Broken())
    end = scheduler_mod._classify(evidence, timed_out=False)

    assert end.outcome == AttemptOutcome.uncertain
    assert end.retryable is False


@pytest.mark.parametrize(
    ("outcome", "error_shown"),
    [
        (AttemptOutcome.failed_before_execution, "That model isn't available."),
        (AttemptOutcome.failed_during_execution, "The provider is overloaded."),
    ],
)
def test_the_schedule_reports_its_last_run_and_how_it_ended(tmp_path, outcome, error_shown):
    """Each failure shows on the schedule as it happens, before any pause,
    and a one-off's does too, though the claim disabled it."""
    from cowork.api.v1.endpoints.schedules import _serialize

    engine = create_engine(f"sqlite:///{tmp_path / 'reason.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        schedule = Schedule(
            title="Digest", prompt="p", cadence="once", model="default", enabled=False,
            next_run_at=datetime.now(timezone.utc) - timedelta(minutes=1), project_id=GENERAL_PROJECT_ID,
        )
        session.add(schedule)
        session.commit()
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        occurrences = ScheduleOccurrenceService(scoped)
        occurrence = occurrences.claim(
            schedule.id, occurrence_key="slot:a", trigger_source="cron",
            scheduled_for=datetime.now(timezone.utc), max_attempts=1, lease_seconds=60,
        )
        started = occurrences.start_attempt(occurrence.id, is_manual=False, lease_seconds=60)
        scheduler_mod._record_attempt_end(
            scoped, attempt=started, occurrence_id=occurrence.id,
            end=scheduler_mod._AttemptEnd(outcome=outcome, error=error_shown),
            retry_delay=None, conversation_id=None,
        )

        data = _serialize(session.get(Schedule, schedule.id), scoped)

        assert session.get(ScheduleOccurrence, occurrence.id).status == OccurrenceStatus.failed
        assert session.get(ScheduleRun, started.run_id).outcome == outcome
    assert data["lastError"] == error_shown
    assert data["lastRun"]["outcome"] == outcome
    assert data["lastRun"]["error"] == error_shown
    assert data["lastRun"]["trigger"] == "cron"
    assert data["state"] == "done"
