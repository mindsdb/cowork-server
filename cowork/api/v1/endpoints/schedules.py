from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.schedules import (
    DEFAULT_MODEL_SENTINEL,
    ScheduleCreateRequest,
    ScheduleEventResponse,
    ScheduleResponse,
    ScheduleRunResponse,
    ScheduleUpdateRequest,
    SkippedPeriodResponse,
    schedule_state,
)
from cowork.services.schedules import (
    ScheduleConflict,
    ScheduleOccurrenceService,
    ScheduleRunService,
    ScheduleService,
)

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])



def _serialize(schedule, scoped) -> dict:
    """ScheduleResponse plus the live state: `running` (any in-flight run),
    `retryAt` (a failed run waiting to be tried again), `state` (what it
    will do next) and `lastSkipped` (the latest slots it passed over)."""
    data = ScheduleResponse.serialize(schedule)
    data["running"] = ScheduleRunService(scoped).has_active_run(schedule.id)
    data["retryAt"] = ScheduleOccurrenceService(scoped).pending_retry_at(schedule.id)
    data["state"] = schedule_state(schedule).value
    skipped = ScheduleService(scoped).last_skipped(schedule.id)
    data["lastSkipped"] = SkippedPeriodResponse.serialize(skipped) if skipped else None
    data["version"] = ScheduleService(scoped).current_version_number(schedule.id)
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
    try:
        schedule = ScheduleService(scoped).update_schedule(
            schedule_id,
            **body.model_dump(exclude_none=True),
        )
    except ScheduleConflict as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
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
    except ScheduleConflict as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
def run_schedule_now(schedule_id: UUID, scoped: ScopedSessionDep, background_tasks: BackgroundTasks):
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from cowork.scheduler import admission, claim_manual_run, run_occurrence
    from cowork.services.conversations import ConversationService

    # Counted with the cron runs: a run-now past the cap would start one more
    # turn than this process runs at once.
    if not admission.reserve():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Too many scheduled tasks are running right now. Run this one when one of them finishes.",
        )
    try:
        # Claim before creating anything: a schedule already running (or
        # waiting to retry) refuses a second run rather than overlapping it.
        occurrence_id = claim_manual_run(scoped, schedule.id)
    except BaseException:
        admission.unreserve()
        raise
    if occurrence_id is None:
        admission.unreserve()
        retrying = ScheduleOccurrenceService(scoped).pending_retry_at(schedule.id) is not None
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This task's last run failed and is waiting to try again. Run it after that retry finishes."
                if retrying
                else "This task is already running. Wait for it to finish, then run it again."
            ),
        )
    admission.fill(occurrence_id)
    try:
        conversation = ConversationService(scoped).create_conversation(
            topic=schedule.title,
            project_id=schedule.project_id,
        )
    except BaseException:
        # Nothing will run the claim; its lease lapses and the reaper settles it.
        admission.release(occurrence_id)
        raise
    # FastAPI closes this session only after the background run ends, which
    # can take minutes. Give the request's connection back before it starts.
    scoped.close()

    background_tasks.add_task(
        run_occurrence, occurrence_id, is_manual=True,
        conversation_id=conversation.id,
    )
    return {"detail": "Run triggered", "conversation_id": str(conversation.id)}


@router.get("/{schedule_id}/runs")
def list_schedule_runs(schedule_id: UUID, scoped: ScopedSessionDep, limit: int = 100):
    try:
        ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    service = ScheduleRunService(scoped)
    runs = service.list_runs(schedule_id, limit=limit)
    context = service.run_context(runs)
    return {"runs": [
        {**ScheduleRunResponse.serialize(r), **_camel(context.get(r.occurrence_id, {}))} for r in runs
    ]}


def _camel(fields: dict) -> dict:
    return {
        "trigger": fields.get("trigger"),
        "scheduledFor": fields.get("scheduled_for"),
        "version": fields.get("version"),
    } if fields else {}


@router.get("/{schedule_id}/events")
def list_schedule_events(schedule_id: UUID, scoped: ScopedSessionDep, limit: int = 100):
    """The schedule's lifecycle: created, edited, paused, resumed, auto-paused."""
    service = ScheduleService(scoped)
    try:
        service.get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"events": [ScheduleEventResponse.serialize(e) for e in service.events(schedule_id, limit=limit)]}


@router.get("/{schedule_id}/events")
def list_schedule_events(schedule_id: UUID, scoped: ScopedSessionDep, limit: int = 100):
    """The schedule's lifecycle: created, edited, paused, resumed, auto-paused."""
    service = ScheduleService(scoped)
    try:
        service.get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"events": [ScheduleEventResponse.serialize(e) for e in service.events(schedule_id, limit=limit)]}
