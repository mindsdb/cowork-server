from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.scoped import ScopedSessionDep
from cowork.db.session import get_session
from cowork.schemas.schedules import (
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleRunResponse,
    ScheduleUpdateRequest,
)
from cowork.services.schedules import ScheduleRunService, ScheduleService

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


def _serialize(schedule, session: Session) -> dict:
    """ScheduleResponse plus the live `running` flag (any in-flight run)."""
    data = ScheduleResponse.serialize(schedule)
    data["running"] = ScheduleRunService(session).has_active_run(schedule.id)
    return data


@router.get("/")
def list_schedules(session: SessionDep, project_id: UUID | None = None):
    schedules = ScheduleService(session).list_schedules(project_id=project_id)
    return {"schedules": [_serialize(s, session) for s in schedules]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_schedule(body: ScheduleCreateRequest, session: SessionDep):
    schedule = ScheduleService(session).create_schedule(
        title=body.title,
        prompt=body.prompt,
        cadence=body.cadence,
        next_run_at=body.next_run_at,
        model=body.model or "default",
        timezone=body.timezone,
        project_id=body.project_id,
        enabled=body.enabled,
    )
    return _serialize(schedule, session)


@router.get("/{schedule_id}")
def get_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return _serialize(ScheduleService(session).get_schedule(schedule_id), session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{schedule_id}")
@router.patch("/{schedule_id}")
def update_schedule(schedule_id: UUID, body: ScheduleUpdateRequest, session: SessionDep):
    try:
        schedule = ScheduleService(session).update_schedule(
            schedule_id,
            **body.model_dump(exclude_none=True),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize(schedule, session)


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: UUID, session: SessionDep):
    deleted = ScheduleService(session).delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


@router.post("/{schedule_id}/pause")
def pause_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return _serialize(ScheduleService(session).pause_schedule(schedule_id), session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/resume")
def resume_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return _serialize(ScheduleService(session).resume_schedule(schedule_id), session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
def run_schedule_now(schedule_id: UUID, session: SessionDep, scoped: ScopedSessionDep, background_tasks: BackgroundTasks):
    try:
        schedule = ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from cowork.scheduler import execute_schedule
    from cowork.services.conversations import ConversationService

    conversation = ConversationService(scoped).create_conversation(
        topic=schedule.title,
        project_id=schedule.project_id,
    )

    background_tasks.add_task(
        execute_schedule, schedule_id, is_manual=True,
        conversation_id=conversation.id,
    )
    return {"detail": "Run triggered", "conversation_id": str(conversation.id)}


@router.get("/{schedule_id}/runs")
def list_schedule_runs(schedule_id: UUID, session: SessionDep, limit: int = 100):
    try:
        ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    runs = ScheduleRunService(session).list_runs(schedule_id, limit=limit)
    return {"runs": [ScheduleRunResponse.serialize(r) for r in runs]}
