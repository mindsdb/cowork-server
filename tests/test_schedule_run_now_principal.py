"""Regression (ENG-289): manual "Run now" must forward the caller's principal
so the turn registers under their org and appears in /responses/in-flight-list;
otherwise the client shows no in-progress state.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import BackgroundTasks
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.api.v1.endpoints.schedules import run_schedule_now
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.principal import Principal
from cowork.scheduler import execute_schedule
from cowork.services.projects import GENERAL_PROJECT_ID


def _scoped() -> ScopedSession:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    return ScopedSession(session, LOCAL_SCOPE)


def _schedule(scoped: ScopedSession) -> Schedule:
    schedule = Schedule(
        title="Say hello",
        prompt="Say hello to me",
        cadence="daily",
        next_run_at=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
        model="default",
        project_id=GENERAL_PROJECT_ID,
    )
    scoped.add(schedule)
    scoped.commit()
    scoped.refresh(schedule)
    return schedule


class _CapturingBackgroundTasks(BackgroundTasks):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def add_task(self, func, *args, **kwargs) -> None:  # noqa: ANN001
        self.calls.append((func, args, kwargs))


def test_run_now_forwards_caller_principal_to_execute_schedule():
    scoped = _scoped()
    schedule = _schedule(scoped)
    principal = Principal(user_id="user-1", org_id="org-a")
    background = _CapturingBackgroundTasks()

    response = run_schedule_now(schedule.id, scoped, background, principal=principal)

    assert response["conversation_id"]
    assert len(background.calls) == 1
    func, _args, kwargs = background.calls[0]
    assert func is execute_schedule
    assert kwargs["principal"] is principal
    assert kwargs["is_manual"] is True
    assert kwargs["conversation_id"] is not None


def test_execute_schedule_accepts_a_principal():
    # Cron calls execute_schedule with no principal; guard the keyword default.
    import inspect

    params = inspect.signature(execute_schedule).parameters
    assert "principal" in params
    assert params["principal"].default is None
