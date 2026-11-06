"""Why a scheduled run failed.

A failed turn's run records the text of its failure frame. A turn with no
record of how it ended is recorded as failed, and neither extends nor ends
the failure streak.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import MAX_FAILED_SLOTS, ScheduleRunService, ScheduleService
from cowork.streaming.buffer import FileStreamBuffer

SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._runs_per_process))


def _scoped():
    session = get_open_session()
    return session, ScopedSession(session, SYSTEM_SCOPE)


def _make_schedule(*, cadence: str = "daily", next_run_at: datetime = SLOT, prompt: str = "first prompt") -> UUID:
    session, scoped = _scoped()
    try:
        return ScheduleService(scoped).create_schedule(
            title=f"failures {uuid4()}", prompt=prompt, cadence=cadence, next_run_at=next_run_at,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id
    finally:
        session.close()


def _delete_schedule(schedule_id: UUID) -> None:
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    session, scoped = _scoped()
    try:
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
    """A turn handler that records each request and ends at once, and a switch
    for what the finished turn's records show."""
    import cowork.handlers.responses as responses_mod

    requests: list = []
    ending = SimpleNamespace(reason="completed", acted=True, failure=None)

    class RecordingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            requests.append(request)

            async def stream():
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        return scheduler._TurnEvidence(reason=ending.reason, acted=ending.acted, failure=ending.failure)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", RecordingHandler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    return SimpleNamespace(requests=requests, ending=ending)


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    session, scoped = _scoped()
    try:
        return list(reversed(ScheduleRunService(scoped).list_runs(schedule_id)))
    finally:
        session.close()


def _schedule(schedule_id: UUID) -> Schedule:
    session, scoped = _scoped()
    try:
        return ScheduleService(scoped).get_schedule(schedule_id)
    finally:
        session.close()


def _poll(schedule_id: UUID) -> list[scheduler.DueSlot]:
    session, scoped = _scoped()
    try:
        return [slot for slot in scheduler._poll(scoped) if slot.schedule_id == schedule_id]
    finally:
        session.close()


async def _run(slot: scheduler.DueSlot) -> None:
    await scheduler.execute_schedule(slot.schedule_id, slot_at=slot.slot_at)


def _client() -> httpx.AsyncClient:
    from cowork.server import create_app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


# --- Failure reasons ---------------------------------------------------------


def _sse(event: str, data: dict | None = None) -> str:
    return f"event: {event}\ndata: {json.dumps(data or {})}\n\n"


def _read_buffer(tmp_path, monkeypatch, frames: list[str], reason: str = "error"):
    buffer = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def fill():
        for frame in frames:
            await buffer.append("sse", {"sse": frame})
        await buffer.close(reason)

    asyncio.run(fill())
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _cid: SimpleNamespace(buffer=buffer)))


def test_the_failure_frame_is_read_with_whether_the_turn_acted(tmp_path, monkeypatch):
    _read_buffer(tmp_path, monkeypatch, [
        _sse("response.created"),
        _sse("response.failed", {"type": "response.failed", "error": "The model isn't available.", "code": "model_not_found"}),
    ])
    acted, failure = asyncio.run(scheduler._turn_activity("c1"))
    assert acted is False
    assert failure == "The model isn't available."


def test_a_turn_that_acted_still_reports_its_failure(tmp_path, monkeypatch):
    _read_buffer(tmp_path, monkeypatch, [
        _sse("response.created"),
        _sse("response.output_text.delta", {"delta": "Working"}),
        _sse("response.failed", {"type": "response.failed", "error": "The provider timed out.", "code": "provider_timeout"}),
    ])
    acted, failure = asyncio.run(scheduler._turn_activity("c1"))
    assert acted is True
    assert failure == "The provider timed out."


def test_a_blank_failure_text_falls_back_to_the_generic_one():
    end = scheduler._classify(scheduler._TurnEvidence(
        reason="error", acted=False, failure=scheduler._failure_from_frame({"error": "  "}),
    ))
    assert end.outcome == AttemptOutcome.failed_before_execution
    assert end.error == scheduler._FAILED_BEFORE_ERROR


async def test_a_run_records_the_turns_own_failure_text(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", True
    turns.ending.failure = "The Linear connection needs reconnecting."
    (slot,) = _poll(schedule_id)
    await _run(slot)

    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.outcome == AttemptOutcome.failed_during_execution
    assert run.error == "The Linear connection needs reconnecting."
    assert _schedule(schedule_id).last_error == "The Linear connection needs reconnecting."


async def test_a_failure_before_acting_uses_its_slot_up(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    turns.ending.failure = "That model isn't available."
    (slot,) = _poll(schedule_id)
    await _run(slot)

    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.failed_before_execution
    assert run.error == "That model isn't available."
    assert ensure_utc(_schedule(schedule_id).next_run_at) > SLOT
    assert _schedule(schedule_id).consecutive_failures == 1
    assert _poll(schedule_id) == []


# --- A turn with no record of how it ended -----------------------------------


async def test_an_uncertain_turn_is_failed_and_leaves_the_streak_alone(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = None, True
    (slot,) = _poll(schedule_id)
    await _run(slot)

    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.outcome == AttemptOutcome.uncertain
    assert run.error == scheduler._UNCERTAIN_ERROR
    schedule = _schedule(schedule_id)
    assert ensure_utc(schedule.next_run_at) > SLOT
    assert schedule.consecutive_failures == 0, "an unreadable record says nothing about the task"
    assert schedule.last_run_at is None, "an uncertain turn writes back no success"


async def test_an_uncertain_turn_neither_extends_nor_ends_a_failure_streak(schedule_id, turns):
    session, scoped = _scoped()
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
        schedule.consecutive_failures = MAX_FAILED_SLOTS - 1
        session.add(schedule)
        session.commit()
    finally:
        session.close()

    turns.ending.reason, turns.ending.acted = None, True
    (slot,) = _poll(schedule_id)
    await _run(slot)

    schedule = _schedule(schedule_id)
    assert schedule.consecutive_failures == MAX_FAILED_SLOTS - 1
    assert schedule.auto_paused_at is None


async def test_an_uncertain_manual_run_does_not_suppress_the_next_slot(schedule_id, turns):
    turns.ending.reason, turns.ending.acted = None, True
    await scheduler.execute_schedule(schedule_id, is_manual=True)
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed

    turns.ending.reason = "completed"
    assert [slot.slot_at for slot in _poll(schedule_id)] == [SLOT], "the freshness guard counts only successes"


# --- lastRun -----------------------------------------------------------------


async def test_a_schedule_reports_its_last_finished_run(schedule_id, turns):
    async with _client() as client:
        assert (await client.get(f"/api/v1/schedules/{schedule_id}")).json()["lastRun"] is None

    turns.ending.reason, turns.ending.acted = "error", True
    turns.ending.failure = "The sheet was not found."
    (slot,) = _poll(schedule_id)
    await _run(slot)

    async with _client() as client:
        last = (await client.get(f"/api/v1/schedules/{schedule_id}")).json()["lastRun"]
    assert last["status"] == "failed"
    assert last["outcome"] == "failed_during_execution"
    assert last["error"] == "The sheet was not found."
    assert last["trigger"] == "cron"
    assert last["conversationId"] is not None


async def test_published_terminal_is_read_before_the_producer_sets_closed(monkeypatch):
    from cowork.streaming.buffer import RedisStreamBuffer
    import cowork.streaming.buffer as buffers

    # Redis has committed the terminal record, but close() is still awaiting
    # its acknowledgement and has not set the local closed flag.
    class Redis:
        async def xrange(self, _key):
            return [("1-0", {"seq": "0", "ts": "", "type": "Done", "data": '{"reason":"completed"}'})]

    buffer = RedisStreamBuffer("race", 1)
    buffer._next_seq = 1
    monkeypatch.setattr(buffers, "get_redis", lambda: Redis())
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _: SimpleNamespace(buffer=buffer)))
    assert not buffer.is_closed
    end = scheduler._classify(await scheduler._turn_evidence("race"))
    assert end.outcome == AttemptOutcome.completed


async def test_missing_terminal_does_not_wait_for_a_stuck_producer(monkeypatch):
    class StuckBuffer:
        latest_seq = 0
        is_closed = False

        async def tail(self, _seq):
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(scheduler, "_EVIDENCE_READ_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "registry", SimpleNamespace(get=lambda _: SimpleNamespace(buffer=StuckBuffer())))
    evidence = await asyncio.wait_for(scheduler._turn_evidence("stuck"), timeout=1)
    assert scheduler._classify(evidence).outcome == AttemptOutcome.uncertain
    assert evidence.acted


@pytest.mark.parametrize("after_handoff", [False, True])
async def test_exception_details_never_reach_run_responses(schedule_id, turns, monkeypatch, after_handoff):
    import cowork.handlers.responses as responses
    from sqlalchemy.exc import OperationalError

    secret = "private-prompt-and-connection-details"

    async def broken_handle(self, request):
        async def stream():
            raise OperationalError("SELECT private_value", {"prompt": secret}, RuntimeError(secret))
            yield
        if after_handoff:
            return stream()
        raise OperationalError("SELECT private_value", {"prompt": secret}, RuntimeError(secret))

    monkeypatch.setattr(responses.ResponsesHandler, "handle", broken_handle)
    session, scoped = _scoped()
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
        schedule.consecutive_failures = 2
        session.add(schedule)
        session.commit()
    finally:
        session.close()
    await _run(_poll(schedule_id)[0])
    run = _runs(schedule_id)[0]
    expected = scheduler._UNCERTAIN_ERROR if after_handoff else scheduler._FAILED_BEFORE_ERROR
    assert run.error == expected
    assert _schedule(schedule_id).last_error == expected
    assert _schedule(schedule_id).consecutive_failures == (2 if after_handoff else 3)
    async with _client() as client:
        response = await client.get(f"/api/v1/schedules/{schedule_id}")
        assert secret not in response.text
        assert response.json()["lastRun"]["error"] == expected


@pytest.mark.parametrize("reason", ["error", "completed"])
async def test_bookkeeping_exception_preserves_confirmed_failures_only(schedule_id, turns, monkeypatch, reason):
    turns.ending.reason = reason
    turns.ending.failure = "The sheet was not found." if reason == "error" else None

    def broken_record(*args, **kwargs):
        raise RuntimeError("private database diagnostics")

    monkeypatch.setattr(scheduler, "_record_turn_outcome", broken_record)
    await _run(_poll(schedule_id)[0])
    run = _runs(schedule_id)[0]
    assert run.outcome == (AttemptOutcome.failed_during_execution if reason == "error" else AttemptOutcome.uncertain)
    assert run.error == (turns.ending.failure if reason == "error" else scheduler._UNCERTAIN_ERROR)
    assert _schedule(schedule_id).consecutive_failures == (1 if reason == "error" else 0)


def test_last_run_skips_released_cron_attempts_but_keeps_manual_failures(schedule_id):
    session, scoped = _scoped()
    try:
        service = ScheduleRunService(scoped)
        retained = ScheduleRun(schedule_id=schedule_id, started_at=SLOT, slot_at=SLOT, status="success", finished_at=SLOT)
        released = ScheduleRun(schedule_id=schedule_id, started_at=SLOT, status="failed", finished_at=SLOT + timedelta(seconds=1))
        session.add_all([retained, released])
        session.commit()
        assert service.last_finished_run(schedule_id).id == retained.id
        manual = ScheduleRun(schedule_id=schedule_id, started_at=SLOT, is_manual=True, status="failed", finished_at=SLOT + timedelta(seconds=2))
        session.add(manual)
        session.commit()
        assert service.last_finished_run(schedule_id).id == manual.id
        assert len(service.list_runs(schedule_id)) == 3
    finally:
        session.close()


def test_last_run_and_trigger_are_described_in_openapi():
    from cowork.server import create_app

    schemas = create_app().openapi()["components"]["schemas"]
    assert "ScheduleRunResponse" in str(schemas["ScheduleResponse"]["properties"]["lastRun"])
    assert schemas["RunTrigger"]["enum"] == ["trial", "manual", "cron"]
