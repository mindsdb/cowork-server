"""ENG-2357: deleting a project must take its schedules (and their runs) with it.

`delete_project` cascades to the project's conversations and nothing else, so on
Postgres the final `DELETE FROM projects` tripped `schedules_project_id_fkey`
and the endpoint returned HTTP 500 -- every time, for any project with a
scheduled task.

These run with `PRAGMA foreign_keys=ON`, following
`test_schedule_runs.py::test_delete_schedule_with_runs_under_enforced_foreign_keys`
(ENG-2356). SQLite leaves foreign keys unenforced by default and nothing turns
them on, so without the pragma the broken ordering passes here exactly as it did
in CI while failing in the cloud.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService
from cowork.services.schedules import ScheduleRunService


def _fk_enforced_session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    return session


def _project(session: Session, tmp_path, name: str = "doomed") -> Project:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    project = Project(name=name, path=str(path))
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def _schedule(session: Session, project_id, title: str = "Daily report") -> Schedule:
    schedule = Schedule(
        title=title,
        prompt="Summarize",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        project_id=project_id,
    )
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return schedule


def test_delete_project_removes_its_schedules(tmp_path, monkeypatch):
    """The reported case: a project with a scheduled task."""
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path))
    session = _fk_enforced_session()
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    project = _project(session, tmp_path)
    schedule = _schedule(session, project.id)

    assert ProjectService(scoped).delete_project(project.id) is True

    assert session.get(Project, project.id) is None
    assert session.get(Schedule, schedule.id) is None


def test_delete_project_removes_schedule_runs_too(tmp_path, monkeypatch):
    """The full subtree: project -> schedules -> runs.

    Runs are the level ENG-2356 fixed; this pins that a project delete reaches
    them, so neither foreign key can strand the other.
    """
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path))
    session = _fk_enforced_session()
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    project = _project(session, tmp_path)
    schedule = _schedule(session, project.id)
    run_service = ScheduleRunService(scoped)
    for _ in range(3):
        run_service.create_run(schedule.id, is_manual=False)

    assert ProjectService(scoped).delete_project(project.id) is True

    assert session.get(Project, project.id) is None
    assert session.get(Schedule, schedule.id) is None
    assert not session.exec(
        select(ScheduleRun).where(ScheduleRun.schedule_id == schedule.id)
    ).all()


def test_delete_project_without_schedules_still_works(tmp_path, monkeypatch):
    """Control. The bug needed a schedule to exist -- which is why the first
    eleven reproduction attempts, and everyone deleting ordinary projects on
    staging, saw nothing wrong."""
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path))
    session = _fk_enforced_session()
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    project = _project(session, tmp_path, name="plain")

    assert ProjectService(scoped).delete_project(project.id) is True
    assert session.get(Project, project.id) is None
