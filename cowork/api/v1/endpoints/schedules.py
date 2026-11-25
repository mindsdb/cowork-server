from __future__ import annotations

import asyncio
import logging
from functools import partial
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, TenantScopeDep
from cowork.db.units import run_db, run_efs_unit
from cowork.schemas.connectors import DisabledConnection
from cowork.schemas.schedules import (
    DEFAULT_MODEL_SENTINEL,
    ScheduleCreateRequest,
    ScheduleResponse,
    ScheduleRunResponse,
    ScheduleUpdateRequest,
)
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.persist import turn_vault_for_scope
from cowork.services.schedules import ScheduleConflict, ScheduleRunService, ScheduleService

logger = logging.getLogger(__name__)

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])



def _serialize_run(run) -> ScheduleRunResponse:
    return ScheduleRunResponse.model_validate(run)


def _serialize(schedule, scoped, *, run_count: int | None = None) -> ScheduleResponse:
    run_service = ScheduleRunService(scoped)
    response = ScheduleResponse.model_validate(schedule)
    response.run_count = run_count if run_count is not None else run_service.run_counts([schedule.id]).get(schedule.id, 0)
    response.running = run_service.has_active_run(schedule.id)
    last = run_service.last_finished_run(schedule.id)
    response.last_run = _serialize_run(last) if last is not None else None
    return response



@router.get("/", response_model=dict[str, list[ScheduleResponse]])
def list_schedules(scoped: ScopedSessionDep, project_id: UUID | None = None):
    schedules = ScheduleService(scoped).list_schedules(project_id=project_id)
    # Which conversations are a schedule's runs comes with the conversations
    # (scheduled_id), so this poll does not grow with each schedule's history.
    counts = ScheduleRunService(scoped).run_counts([s.id for s in schedules])
    return {"schedules": [_serialize(s, scoped, run_count=counts.get(s.id, 0)) for s in schedules]}


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=ScheduleResponse)
async def create_schedule(body: ScheduleCreateRequest, scope: TenantScopeDep, request: Request):
    # A schedule with no project runs in the caller's default project, which
    # checks the general project's folder on the file store.
    unit = run_db if body.project_id is not None else run_efs_unit
    exclusions = await _resolve_exclusions(body.disabled_connections, scope, request)
    return await unit(partial(_create_schedule, body=body, exclusions=exclusions), scope=scope)


async def _resolve_exclusions(
    exclusions: list[DisabledConnection] | None, scope: TenantScope, request: Request,
    *, stored: list[dict] | None = None,
) -> list[dict]:
    """Validate against this member's visible connections and store stable datasource IDs.

    An entry the task already holds (``stored``) stays as it is when its
    connection is gone: the editor resends the whole list with every save and
    shows no box for a missing connection, so refusing it would refuse every
    later edit. Kept rather than dropped, so a connection that comes back
    under the same name or id is still left out.
    """
    if not exclusions:
        return []
    kept = {(e.get("engine"), e.get("name"), e.get("datasource_id")): e for e in stored or [] if isinstance(e, dict)}
    if scope.org_mode:
        settings = OAuthSettings()
        catalogue, datasources, custom = await asyncio.gather(
            auth_proxy.proxy_catalogue(request, settings),
            auth_proxy.proxy_datasource_list(request, settings),
            auth_proxy.proxy_custom_connection_list(request, settings),
        )
        known = {(c["engine"], c["name"]) for item in catalogue.get("items", []) for c in item.get("connections", [])}
        known.update((c["connector_id"], c["name"]) for c in custom)
        by_id = {(c["connector_id"], c["id"]): c for c in datasources}
        by_name = {(c["connector_id"], c["name"]): c for c in datasources}
    else:
        rows = await run_efs_unit(lambda _: turn_vault_for_scope(scope).list_connections(), scope=scope)
        known = {(c["engine"], c["name"]) for c in rows}
        by_id, by_name = {}, {}

    canonical = []
    for item in exclusions:
        key = (item.engine, item.name)
        if item.datasource_id is not None:
            source = by_id.get((item.engine, item.datasource_id))
        else:
            source = by_name.get(key)
            if key in known:
                if source is not None:
                    raise HTTPException(status_code=422, detail="Two connectors match this exclusion. Select it again.")
                canonical.append(item.model_dump())
                continue
        if source is None:
            already = kept.get((item.engine, item.name, item.datasource_id))
            if already is not None:
                canonical.append(already)
                continue
            raise HTTPException(status_code=422, detail="An excluded connector could not be found in your connections.")
        canonical.append(DisabledConnection(
            engine=item.engine, name=source["name"], datasource_id=source["id"],
        ).model_dump())
    return canonical


def _create_schedule(scoped: ScopedSession, *, body: ScheduleCreateRequest, exclusions: list[dict]) -> ScheduleResponse:
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
            disabled_connections=exclusions,
        )
    except ValueError as e:
        # e.g. a project_id that isn't visible in this scope
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize(schedule, scoped)


@router.get("/{schedule_id}", response_model=ScheduleResponse)
def get_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).get_schedule(schedule_id), scoped)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{schedule_id}", response_model=ScheduleResponse)
@router.patch("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: UUID, body: ScheduleUpdateRequest, scope: TenantScopeDep, request: Request):
    changes = body.model_dump(exclude_none=True)
    # "No project" arrives as an explicit null, which exclude_none drops.
    if "project_id" in body.model_fields_set and body.project_id is None:
        changes["project_id"] = None
    if body.disabled_connections is not None:
        stored = await run_db(partial(_stored_exclusions, schedule_id=schedule_id), scope=scope) if body.disabled_connections else None
        changes["disabled_connections"] = await _resolve_exclusions(
            body.disabled_connections, scope, request, stored=stored,
        )
    unit = run_efs_unit if "project_id" in changes and changes["project_id"] is None else run_db
    return await unit(partial(_update_schedule, schedule_id=schedule_id, changes=changes), scope=scope)


def _stored_exclusions(scoped: ScopedSession, *, schedule_id: UUID) -> list[dict]:
    try:
        return list(ScheduleService(scoped).get_schedule(schedule_id).disabled_connections or [])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


def _update_schedule(scoped: ScopedSession, *, schedule_id: UUID, changes: dict) -> ScheduleResponse:
    try:
        schedule = ScheduleService(scoped).update_schedule(schedule_id, **changes)
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


@router.post("/{schedule_id}/pause", response_model=ScheduleResponse)
def pause_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).pause_schedule(schedule_id), scoped)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/resume", response_model=ScheduleResponse)
def resume_schedule(schedule_id: UUID, scoped: ScopedSessionDep):
    try:
        return _serialize(ScheduleService(scoped).resume_schedule(schedule_id), scoped)
    except ScheduleConflict as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{schedule_id}/run-now", status_code=status.HTTP_202_ACCEPTED)
def run_schedule_now(
    schedule_id: UUID, scoped: ScopedSessionDep, background_tasks: BackgroundTasks, trial: bool = False,
):
    """Start an additional run. Any upcoming occurrence keeps its time.
    A trial leaves the failure streak unchanged."""
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from cowork.scheduler import (
        abandon_manual_run,
        claim_manual_run,
        org_runs_full,
        release_run_place,
        reserve_run_place,
        run_admitted,
        schedule_conversation_topic,
    )
    from cowork.services.conversations import ConversationService
    from cowork.services.schedules import OrgRunsAtLimit, ScheduleBusy

    already_running = HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This task is already running.")
    org_full = HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Too many scheduled tasks are running for this account right now. Try again in a few minutes.",
    )

    # Refused with a reason rather than queued: the user is watching and can
    # press Run again.
    if ScheduleRunService(scoped).has_active_run(schedule_id):
        raise already_running
    if org_runs_full(scoped, schedule):
        raise org_full
    if not reserve_run_place():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Too many scheduled tasks are running right now. Try again in a few minutes.",
        )
    # The run row is the claim: inserting it before answering means two
    # presses at once, or a press racing a cron claim, start one run (the
    # schedule allows one `running` run).
    try:
        run_id = claim_manual_run(scoped, schedule_id, trial=trial)
    except ScheduleBusy:
        release_run_place()
        raise already_running
    except OrgRunsAtLimit:
        # Counted in the claim's transaction, under the organization's lock, so
        # two presses can't both take its last place.
        release_run_place()
        raise org_full
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
        conversation_id=conversation.id, run_id=run_id, trial=trial,
    )
    return {"detail": "Run triggered", "conversation_id": str(conversation.id)}


@router.get("/{schedule_id}/runs", response_model=dict[str, list[ScheduleRunResponse]])
def list_schedule_runs(schedule_id: UUID, scoped: ScopedSessionDep, limit: int = 100):
    try:
        ScheduleService(scoped).get_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    runs = ScheduleRunService(scoped).list_runs(schedule_id, limit=limit)
    return {"runs": [_serialize_run(r) for r in runs]}
