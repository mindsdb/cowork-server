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
        missed_run_policy=body.missed_run_policy,
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
        schedule = ScheduleService(session).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # Create the conversation eagerly so we can mark it in-flight
    # *before* the background task starts. This closes the race where
    # the client sees the new conversation, polls /in-flight-list,
    # doesn't find it yet, and injects a "got interrupted" prompt.
    from cowork.scheduler import execute_schedule
    from cowork.services.conversations import ConversationService

    try:
        from cowork.api.v1.endpoints.responses import mark_stream_active
    except ImportError:  # best-effort hint; streaming registry self-tracks
        def mark_stream_active(_conversation_id: str) -> None:
            pass

    conversation = ConversationService(session).create_conversation(
        topic=schedule.title,
        project_id=schedule.project_id,
    )
    mark_stream_active(str(conversation.id))

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
