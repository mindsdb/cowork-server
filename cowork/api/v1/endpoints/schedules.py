from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.schedules import (
    DEFAULT_MODEL_SENTINEL,
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleRunResponse,
    ScheduleUpdateRequest,
)
from cowork.services.schedules import ScheduleRunService, ScheduleService

logger = logging.getLogger(__name__)

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
def create_schedule(body: ScheduleCreateRequest, scoped: ScopedSessionDep):
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
    changes = body.model_dump(exclude_none=True)
    # "No project" arrives as an explicit null, which exclude_none drops.
    if "project_id" in body.model_fields_set and body.project_id is None:
        changes["project_id"] = None
    try:
        schedule = ScheduleService(scoped).update_schedule(schedule_id, **changes)
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

    from cowork.scheduler import (
        abandon_manual_run,
        claim_manual_run,
        release_run_place,
        reserve_run_place,
        run_admitted,
        schedule_conversation_topic,
    )
    from cowork.services.conversations import ConversationService
    from cowork.services.schedules import ScheduleBusy

    already_running = HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This task is already running.")

    # Refused with a reason rather than queued: the user is watching and can
    # press Run again.
    if ScheduleRunService(scoped).has_active_run(schedule_id):
        raise already_running
    if not reserve_run_place():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Too many scheduled tasks are running right now. Try again in a few minutes.",
        )
    # The run row is the claim: inserting it before answering means two
    # presses at once, or a press racing a cron claim, start one run (the
    # schedule allows one `running` run).
    try:
        run_id = claim_manual_run(scoped, schedule_id)
    except ScheduleBusy:
        release_run_place()
        raise already_running
    except BaseException:
        release_run_place()
        raise
    try:
        conversation = ConversationService(scoped).create_conversation(
            topic=schedule_conversation_topic(schedule.title),
            project_id=schedule.project_id,
        )
    except BaseException:
        release_run_place()
        try:
            scoped.rollback()
            abandon_manual_run(scoped, run_id, "Run did not start: its conversation could not be created.")
        except Exception:
            # Its lease lapses and a poll reaps it.
            logger.exception(
                "Could not abandon a scheduled run after conversation creation failed",
                extra={"event": "schedule_abandon_failed", "schedule_id": str(schedule_id)},
            )
        raise
    # FastAPI closes this session only after the background run ends, which
    # can take minutes. Give the request's connection back before it starts.
    scoped.close()

    background_tasks.add_task(
        run_admitted, schedule_id, is_manual=True,
        conversation_id=conversation.id, run_id=run_id,
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
