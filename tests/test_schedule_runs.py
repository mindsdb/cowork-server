from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.project import Project
from cowork.models.schedule import Schedule
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
