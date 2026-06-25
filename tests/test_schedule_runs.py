from __future__ import annotations

from datetime import datetime, timezone

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


def test_has_running_run_true_while_status_running():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(session)
    run = run_service.create_run(schedule.id)

    assert run_service.has_running_run(schedule.id) is True

    run_service.finish_run(run.id)
    assert run_service.has_running_run(schedule.id) is False
