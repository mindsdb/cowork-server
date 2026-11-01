from __future__ import annotations

from functools import partial
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.db.units import run_db, run_efs_unit
from cowork.schemas.schedules import (
    DEFAULT_MODEL_SENTINEL,
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleRunResponse,
    ScheduleUpdateRequest,
)
from cowork.services.schedules import ScheduleRunService, ScheduleService

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])



def _serialize(schedule, scoped) -> dict:
    """ScheduleResponse plus the live `running` flag (any in-flight run)."""
    data = ScheduleResponse.serialize(schedule)
    data["running"] = ScheduleRunService(scoped).has_active_run(schedule.id)
    return data


@router.get("/")
def list_schedules(scoped: ScopedSessionDep, project_id: UUID | None = None):
    schedules = ScheduleService(scoped).list_schedules(project_id=project_id)
    return {"schedules": [_serialize(s, scoped) for s in schedules]}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_schedule(body: ScheduleCreateRequest, scope: Annotated[TenantScope, Depends(get_tenant_scope)]):
    # A schedule with no project runs in the caller's default project, which
    # checks the general project's folder on the file store.
    unit = run_db if body.project_id is not None else run_efs_unit
    return await unit(partial(_create_schedule, body=body), scope=scope)


def _create_schedule(scoped: ScopedSession, *, body: ScheduleCreateRequest) -> dict:
    try:
        schedule = ScheduleService(scoped).create_schedule(
            title=body.title,
            prompt=body.prompt,
            cadence=body.cadence,
            next_run_at=body.next_run_at,
            model=body.model or DEFAULT_MODEL_SENTINEL,
            timezone=body.timezone,
            project_id=body.project_id,
            enabled=body.enabled,
        )
    except ValueError as e:
        # e.g. a project_id that isn't visible in this scope
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize(schedule, scoped)


@router.get("/{schedule_id}")
def get_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).get_schedule(schedule_id), scoped)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{schedule_id}")
@router.patch("/{schedule_id}")
def update_schedule(schedule_id: UUID, body: ScheduleUpdateRequest, scoped: ScopedSessionDep):
    try:
        schedule = ScheduleService(scoped).update_schedule(
            schedule_id,
            **body.model_dump(exclude_none=True),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize(schedule, scoped)


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    deleted = ScheduleService(scoped).delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


@router.post("/{schedule_id}/pause")
def pause_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).pause_schedule(schedule_id), scoped)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/resume")
def resume_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).resume_schedule(schedule_id), scoped)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
def run_schedule_now(schedule_id: UUID, scoped: ScopedSessionDep, background_tasks: BackgroundTasks):
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from cowork.scheduler import execute_schedule
    from cowork.services.conversations import ConversationService

    conversation = ConversationService(scoped).create_conversation(
        topic=schedule.title,
        project_id=schedule.project_id,
    )
    # FastAPI closes this session only after the background run ends, which
    # can take minutes. Give the request's connection back before it starts.
    scoped.close()

    background_tasks.add_task(
        execute_schedule, schedule_id, is_manual=True,
        conversation_id=conversation.id,
    )
    return {"detail": "Run triggered", "conversation_id": str(conversation.id)}


@router.get("/{schedule_id}/runs")
def list_schedule_runs(schedule_id: UUID, scoped: ScopedSessionDep, limit: int = 100):
    try:
        ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    runs = ScheduleRunService(scoped).list_runs(schedule_id, limit=limit)
    return {"runs": [ScheduleRunResponse.serialize(r) for r in runs]}
