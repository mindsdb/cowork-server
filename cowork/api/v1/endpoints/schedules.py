from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session

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


@router.get("/")
def list_schedules(session: SessionDep, project_id: UUID | None = None):
    schedules = ScheduleService(session).list_schedules(project_id=project_id)
    return {"schedules": [ScheduleResponse.serialize(s) for s in schedules]}


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
    return ScheduleResponse.serialize(schedule)


@router.get("/{schedule_id}")
def get_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleResponse.serialize(ScheduleService(session).get_schedule(schedule_id))
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
    return ScheduleResponse.serialize(schedule)


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: UUID, session: SessionDep):
    deleted = ScheduleService(session).delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


@router.post("/{schedule_id}/pause")
def pause_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleResponse.serialize(ScheduleService(session).pause_schedule(schedule_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/resume")
def resume_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleResponse.serialize(ScheduleService(session).resume_schedule(schedule_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
def run_schedule_now(schedule_id: UUID, session: SessionDep, background_tasks: BackgroundTasks):
    try:
        ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from cowork.scheduler import execute_schedule
    background_tasks.add_task(execute_schedule, schedule_id, is_manual=True)
    return {"detail": "Run triggered"}


@router.get("/{schedule_id}/runs")
def list_schedule_runs(schedule_id: UUID, session: SessionDep, limit: int = 100):
    try:
        ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    runs = ScheduleRunService(session).list_runs(schedule_id, limit=limit)
    return {"runs": [ScheduleRunResponse.serialize(r) for r in runs]}
