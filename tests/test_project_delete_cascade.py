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
from cowork.models.channel import ChannelBinding
from cowork.models.conversation import Conversation
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


def test_delete_project_when_a_schedule_points_at_a_deleted_conversation(
    tmp_path, monkeypatch
):
    """The two cascades meet.

    `delete_project` removes the project's conversations BEFORE the project
    row, and both `schedules.last_result_conversation_id` and
    `schedule_runs.conversation_id` are foreign keys into conversations with no
    `ondelete`. So a schedule that recorded a run in one of those conversations
    is pointing at a row that is about to disappear, while the schedule itself
    survives until the project row goes.

    `ScheduleService.release_conversation` nulls those pointers on the way past,
    which is what keeps this working -- but nothing pinned it, and the ordering
    is not obvious from either side.
    """
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path))
    session = _fk_enforced_session()
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    project = _project(session, tmp_path, name="entangled")

    conversation = Conversation(project_id=project.id, topic="run output")
    session.add(conversation)
    session.commit()
    session.refresh(conversation)

    schedule = _schedule(session, project.id)
    schedule.last_result_conversation_id = conversation.id
    session.add(schedule)
    session.commit()
    ScheduleRunService(scoped).create_run(schedule.id, is_manual=False)

    assert ProjectService(scoped).delete_project(project.id) is True

    assert session.get(Project, project.id) is None
    assert session.get(Schedule, schedule.id) is None
    assert session.get(Conversation, conversation.id) is None


def test_delete_project_releases_a_channel_binding_without_deleting_it(
    tmp_path, monkeypatch
):
    """The fourth foreign key into projects, found in review.

    `channel_bindings.anton_project_id` is nullable and had no `ondelete`, so a
    project that a Slack/Telegram route pointed at could not be deleted at all.

    SET NULL rather than CASCADE is the point of this test: deleting a project
    must NOT delete someone's channel route. The runtime reads the column as
    optional (`binding.anton_project_id or self._resolve_default_project_id(...)`
    at runtime.py:357 and :417), so a released binding keeps serving on the
    default project. Asserting only "the delete stopped throwing" would pass
    just as happily with CASCADE, which would silently destroy the route.
    """
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path))
    session = _fk_enforced_session()
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    project = _project(session, tmp_path, name="routed")

    binding = ChannelBinding(
        channel_type="slack",
        external_group_id="C123",
        anton_project_id=project.id,
    )
    session.add(binding)
    session.commit()
    session.refresh(binding)

    assert ProjectService(scoped).delete_project(project.id) is True

    assert session.get(Project, project.id) is None
    survivor = session.get(ChannelBinding, binding.id)
    assert survivor is not None, "the channel route must outlive the project"
    session.refresh(survivor)
    assert survivor.anton_project_id is None
    assert survivor.external_group_id == "C123"
