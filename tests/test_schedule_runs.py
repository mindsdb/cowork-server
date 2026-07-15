from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService


def _session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    return session


def _schedule(session: Session) -> Schedule:
    schedule = Schedule(
        title="Daily report",
        prompt="Summarize",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        project_id=GENERAL_PROJECT_ID,
    )
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return schedule


def test_has_running_run_false_when_no_runs():
    session = _session()
    schedule = _schedule(session)
    assert ScheduleRunService(session).has_running_run(schedule.id) is False


def test_has_running_run_true_for_scheduled_run_in_progress():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    run = run_service.create_run(schedule.id, is_manual=False)

    assert run_service.has_running_run(schedule.id) is True

    run_service.finish_run(run.id)
    assert run_service.has_running_run(schedule.id) is False


def test_has_running_run_ignores_manual_run_in_progress():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    run = run_service.create_run(schedule.id, is_manual=True)

    assert run_service.has_running_run(schedule.id) is False

    run_service.finish_run(run.id)
    assert run_service.has_running_run(schedule.id) is False


# --- ENG-688: freshness guard — a due cron slot is skipped when a successful
# run (typically a manual "run now") finished within the cadence window.

def _finish_at(session: Session, run_id, when: datetime) -> None:
    from cowork.models.schedule import ScheduleRun

    run = session.get(ScheduleRun, run_id)
    run.finished_at = when
    session.add(run)
    session.commit()


def test_last_successful_finish_ignores_failures_and_running():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)

    assert run_service.last_successful_finish(schedule.id) is None

    failed = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(failed.id, error="boom")
    run_service.create_run(schedule.id, is_manual=False)  # still running
    assert run_service.last_successful_finish(schedule.id) is None

    ok = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(ok.id)
    finished = run_service.last_successful_finish(schedule.id)
    assert finished is not None and finished.tzinfo is not None


def test_due_slot_skipped_and_advanced_after_recent_manual_success():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)  # daily, due at 2026-06-25 09:00 UTC
    run_service = ScheduleRunService(session)

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id)
    _finish_at(session, run.id, now - timedelta(minutes=50))

    assert _due_schedules(session, now) == []
    session.refresh(schedule)
    # Slot consumed: advanced past the skipped occurrence to the next day.
    assert schedule.next_run_at.replace(tzinfo=timezone.utc) > now


def test_due_slot_runs_when_last_success_is_old():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id)
    _finish_at(session, run.id, now - timedelta(hours=2))

    assert [s.id for s in _due_schedules(session, now)] == [schedule.id]


def test_due_slot_deferred_while_manual_run_in_flight():
    """PR #181 review issue 3: a manual run still executing when the cron
    slot comes due must block the slot — otherwise both publish the same
    output. Deferred, not consumed: the slot stays due, and once the manual
    run finishes the freshness guard decides whether it still fires."""
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)  # daily, due at 2026-06-25 09:00 UTC
    run_service = ScheduleRunService(session)
    run_service.create_run(schedule.id, is_manual=True)  # still running

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    assert _due_schedules(session, now) == []
    session.refresh(schedule)
    assert schedule.next_run_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 6, 25, 9, 0, tzinfo=timezone.utc
    )


def test_due_slot_runs_when_recent_run_failed():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id, error="boom")
    _finish_at(session, run.id, now - timedelta(minutes=10))

    assert [s.id for s in _due_schedules(session, now)] == [schedule.id]


# --- ENG-688: cancelled-run status + the UI-facing "running" flag.

def test_finish_run_status_override_records_cancelled():
    from cowork.schemas.schedules import RunStatus

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)

    run = run_service.create_run(schedule.id, is_manual=False)
    finished = run_service.finish_run(run.id, status=RunStatus.cancelled)
    assert finished.status == RunStatus.cancelled
    assert finished.error is None
    # A cancelled run is not a success: it neither blocks via the freshness
    # guard nor counts as the last successful finish.
    assert run_service.last_successful_finish(schedule.id) is None


def test_has_active_run_counts_manual_runs():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)

    assert run_service.has_active_run(schedule.id) is False
    run = run_service.create_run(schedule.id, is_manual=True)
    # Manual runs are invisible to the cron-overlap guard but visible here.
    assert run_service.has_running_run(schedule.id) is False
    assert run_service.has_active_run(schedule.id) is True

    run_service.finish_run(run.id)
    assert run_service.has_active_run(schedule.id) is False


# --- ENG-688: schedule/run identity stamped on the turn's trace.

def test_execute_schedule_stamps_trace_identity(monkeypatch):
    import asyncio

    import cowork.handlers.responses as responses_mod
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    captured: list = []

    class FakeHandler:
        def __init__(self, session):
            pass

        async def handle(self, request):
            captured.append(request)

            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    session = get_open_session()
    schedule = ScheduleService(session).create_schedule(
        title="trace stamp test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    session.close()

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=False))
        asyncio.run(execute_schedule(schedule_id, is_manual=True))

        cron_req, manual_req = captured
        assert cron_req.trace_tags == ["scheduled_task", "trigger:cron"]
        assert manual_req.trace_tags == ["scheduled_task", "trigger:manual"]
        for req, trigger in ((cron_req, "cron"), (manual_req, "manual")):
            assert req.trace_metadata["schedule_id"] == str(schedule_id)
            assert req.trace_metadata["trigger_type"] == trigger
            assert req.trace_metadata["schedule_run_id"]
    finally:
        s = get_open_session()
        ScheduleService(s).delete_schedule(schedule_id)
        s.close()


# --- ENG-688: how the run actually ended comes from the stream buffer's
# terminal record. The producer runs detached and swallows its own
# cancellation (task.cancelled() stays False), so the terminal record is the
# only truthful signal — without it a cancelled or failed run is recorded as
# success.

def test_turn_terminal_reason_reads_the_terminal_record(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import cowork.scheduler as scheduler_mod
    from cowork.streaming.buffer import FileStreamBuffer

    buf = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def _fill():
        await buf.append("sse", {"sse": "event: response.created"})
        await buf.close("cancelled")

    asyncio.run(_fill())
    monkeypatch.setattr(
        scheduler_mod, "registry",
        SimpleNamespace(get=lambda cid: SimpleNamespace(buffer=buf)),
    )
    assert asyncio.run(scheduler_mod._turn_terminal_reason("c1")) == "cancelled"


def test_turn_terminal_reason_with_real_registry_cancel(tmp_path):
    """End-to-end through the real registry: cancel a producer that swallows
    its CancelledError the way handlers/responses._produce does. The task
    ends NOT-cancelled — which is exactly why task state can't be the
    signal — while the buffer terminal record says "cancelled"."""
    import asyncio

    import cowork.scheduler as scheduler_mod
    from cowork.streaming.buffer import FileStreamBuffer
    from cowork.streaming.registry import registry

    buf = FileStreamBuffer(tmp_path / "turn.jsonl")
    conversation_id = "eng688-real-cancel-test"

    async def main():
        started = asyncio.Event()

        async def producer():
            try:
                await buf.append("sse", {"sse": "event: response.created"})
                started.set()
                await asyncio.sleep(30)
                await buf.close("completed")
            except asyncio.CancelledError:
                await buf.close("cancelled")
                return

        handle = await registry.start(
            conversation_id=conversation_id,
            turn_id=0,
            buffer=buf,
            producer_coro=producer(),
        )
        await started.wait()
        await registry.cancel(conversation_id)
        assert handle.task.done()
        assert handle.task.cancelled() is False  # the swallowed cancel
        return await scheduler_mod._turn_terminal_reason(conversation_id)

    assert asyncio.run(main()) == "cancelled"


def test_turn_terminal_reason_none_without_handle(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import cowork.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "registry", SimpleNamespace(get=lambda cid: None))
    assert asyncio.run(scheduler_mod._turn_terminal_reason("c1")) is None


def _execute_with_terminal(monkeypatch, reason, *, is_manual=False):
    """Run execute_schedule with a no-op turn and a forced terminal reason;
    return the resulting run/schedule state as plain values."""
    import asyncio

    import cowork.handlers.responses as responses_mod
    import cowork.scheduler as scheduler_mod
    from cowork.common.datetime_utils import ensure_utc
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    class FakeHandler:
        def __init__(self, session):
            pass

        async def handle(self, request):
            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    async def _terminal(_conversation_id):
        return reason

    monkeypatch.setattr(scheduler_mod, "_turn_terminal_reason", _terminal)

    session = get_open_session()
    schedule = ScheduleService(session).create_schedule(
        title="terminal mapping test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    original_next = ensure_utc(schedule.next_run_at)
    session.close()

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=is_manual))
        check = get_open_session()
        fresh = ScheduleService(check).get_schedule(schedule_id)
        run = ScheduleRunService(check).list_runs(schedule_id)[0]
        state = {
            "run_status": run.status,
            "run_error": run.error,
            "run_conversation_id": run.conversation_id,
            "last_error": fresh.last_error,
            "last_run_at": fresh.last_run_at,
            "next_advanced": ensure_utc(fresh.next_run_at) > original_next,
        }
        check.close()
        return state
    finally:
        s = get_open_session()
        ScheduleService(s).delete_schedule(schedule_id)
        s.close()


def test_execute_schedule_records_cancelled_and_consumes_slot(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "cancelled")
    assert state["run_status"] == RunStatus.cancelled
    assert state["run_error"] is None
    assert state["last_error"] is None
    assert state["last_run_at"] is None
    # The slot is consumed — otherwise the next tick restarts the run the
    # user just killed (a cancelled run isn't a success, so the freshness
    # guard wouldn't block it).
    assert state["next_advanced"] is True


def test_execute_schedule_records_producer_error_as_failed(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "error")
    assert state["run_status"] == RunStatus.failed
    assert state["run_error"]
    assert state["last_error"]
    assert state["next_advanced"] is True


def test_execute_schedule_completed_is_success(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "completed")
    assert state["run_status"] == RunStatus.success
    assert state["run_error"] is None
    assert state["last_error"] is None
    assert state["last_run_at"] is not None
    assert state["next_advanced"] is True


def test_execute_schedule_links_conversation_before_turn_starts(monkeypatch):
    """The run's conversation is recorded as soon as it exists — not at
    finish — so the runs list can open a run that is still executing."""
    import asyncio

    import cowork.handlers.responses as responses_mod
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    session = get_open_session()
    schedule = ScheduleService(session).create_schedule(
        title="early link test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    session.close()

    seen: dict = {}

    class FakeHandler:
        def __init__(self, session):
            pass

        async def handle(self, request):
            check = get_open_session()
            run = ScheduleRunService(check).list_runs(schedule_id)[0]
            seen["conversation_id_during_turn"] = (
                str(run.conversation_id) if run.conversation_id else None
            )
            seen["request_conversation"] = request.conversation
            check.close()

            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=False))
        assert seen["conversation_id_during_turn"] == seen["request_conversation"]
    finally:
        s = get_open_session()
        ScheduleService(s).delete_schedule(schedule_id)
        s.close()


# --- ENG-769: reap orphaned `running` runs left by a crash/restart.

def test_reap_orphaned_runs_marks_running_as_failed():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    run = run_service.create_run(schedule.id, is_manual=False)

    # The stale `running` row would otherwise wedge the schedule forever.
    assert run_service.has_running_run(schedule.id) is True

    reaped = run_service.reap_orphaned_runs()

    assert reaped == 1
    assert run_service.has_running_run(schedule.id) is False

    session.refresh(run)
    assert run.status == RunStatus.failed
    assert run.error is not None
    assert run.finished_at is not None
    assert run.duration_ms is not None


def test_reap_orphaned_runs_reaps_manual_runs_too():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    manual = run_service.create_run(schedule.id, is_manual=True)

    assert run_service.reap_orphaned_runs() == 1

    session.refresh(manual)
    assert manual.status == RunStatus.failed


def test_reap_orphaned_runs_leaves_finished_runs_untouched():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    run = run_service.create_run(schedule.id, is_manual=False)
    run_service.finish_run(run.id)

    assert run_service.reap_orphaned_runs() == 0

    session.refresh(run)
    assert run.status == RunStatus.success
    assert run.error is None
