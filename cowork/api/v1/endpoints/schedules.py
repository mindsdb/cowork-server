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


@router.get("/", response_model=list[ScheduleResponse])
def list_schedules(session: SessionDep, project_id: UUID | None = None):
    return ScheduleService(session).list_schedules(project_id=project_id)


@router.post("/", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
def create_schedule(body: ScheduleCreateRequest, session: SessionDep):
    return ScheduleService(session).create_schedule(
        title=body.title,
        prompt=body.prompt,
        cadence=body.cadence,
        next_run_at=body.next_run_at,
        model=body.model,
        timezone=body.timezone,
        project_id=body.project_id,
        enabled=body.enabled,
    )


@router.get("/{schedule_id}", response_model=ScheduleResponse)
def get_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(schedule_id: UUID, body: ScheduleUpdateRequest, session: SessionDep):
    try:
        return ScheduleService(session).update_schedule(
            schedule_id,
            **body.model_dump(exclude_none=True),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: UUID, session: SessionDep):
    deleted = ScheduleService(session).delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


@router.post("/{schedule_id}/pause", response_model=ScheduleResponse)
def pause_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleService(session).pause_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/resume", response_model=ScheduleResponse)
def resume_schedule(schedule_id: UUID, session: SessionDep):
    try:
        return ScheduleService(session).resume_schedule(schedule_id)
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


@router.get("/{schedule_id}/runs", response_model=list[ScheduleRunResponse])
def list_schedule_runs(schedule_id: UUID, session: SessionDep, limit: int = 100):
    try:
        ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ScheduleRunService(session).list_runs(schedule_id, limit=limit)
