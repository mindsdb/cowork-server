from datetime import datetime
from functools import partial
from typing import Annotated, NamedTuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from cowork.api.v1.endpoints.responses import _shared_turn
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.db.units import run_db, run_efs_unit
from cowork.models.comparison import Comparison
from cowork.principal import mindshub_credential
from cowork.schemas.comparisons import (
    ComparisonContinueRequest,
    ComparisonCreateRequest,
    ComparisonResponse,
    ComparisonVerdictRequest,
    ComparisonUsageResponse,
)
from cowork.services.comparisons import (
    ComparisonConflictError,
    ComparisonNotFoundError,
    ComparisonService,
    ProjectTooLargeToCopyError,
    SideSpec,
)
from cowork.services.comparison_usage import comparison_usage
from cowork.services.projects import ProjectNameLockBusyError, ProjectNotFoundError

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

_SIDE_LABEL = r"^[ab]$"


async def _running_elsewhere(conversation_ids: list[str], scope: TenantScope) -> frozenset[str]:
    """The conversations among these whose turn another replica is running.

    The service's own check sees only this process's runs. With two replicas,
    the one answering may not be the one running the turn, so a side with no
    running local handle is looked up in the shared turn index. A finished
    handle stays registered after its turn, so it says nothing about a newer
    turn the other replica may have started. Not knowing is a refusal: moving
    or deleting a live side's workspace is not undone by a retry.
    """
    from cowork.streaming.registry import registry

    running = set()
    for conversation_id in conversation_ids:
        handle = registry.get(conversation_id)
        if handle is not None and handle.is_running:
            continue
        try:
            found = await _shared_turn(conversation_id, scope)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Couldn't check whether this comparison's sides are still working. Try again in a moment.",
            ) from None
        if found is not None and found.in_flight:
            running.add(conversation_id)
    return frozenset(running)


def _response(service: ComparisonService, comparison: Comparison, continued: dict | None = None) -> dict:
    """`continued`: `service.continued_sides` for a page that holds this
    comparison, so a list reads it once rather than per comparison."""
    if continued is None:
        continued = service.continued_sides([comparison])
    verdicts = sorted(comparison.verdicts, key=lambda v: v.turn_index)
    return ComparisonResponse.serialize({
        "id": comparison.id,
        "title": comparison.title,
        "created_at": comparison.created_at,
        "source_project_id": comparison.source_project_id,
        "source_project_label": comparison.source_project_label,
        "sides": [
            {
                "label": side.label,
                "model": side.model,
                "reasoning_effort": side.reasoning_effort,
                "project_id": side.project_id,
                "conversation_id": side.conversation_id,
                "turn_count": service.turn_count(side),
                "continued_at": side.continued_at,
                "continued_turn_count": side.continued_turn_count,
                "carry_incomplete": continued[side.id].carry_incomplete if side.id in continued else False,
                "continued_project_id": continued[side.id].project_id if side.id in continued else None,
                "usage": side.usage_snapshot,
            }
            for side in sorted(comparison.sides, key=lambda s: s.label)
        ],
        "verdicts": verdicts,
        "verdict": verdicts[-1].winner if verdicts else None,
    })


def _get(service: ComparisonService, comparison_id: UUID) -> Comparison:
    try:
        return service.get_comparison(comparison_id)
    except ComparisonNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/")
def list_comparisons(scoped: ScopedSessionDep, limit: int = 50, offset: int = 0):
    service = ComparisonService(scoped)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    # One past the page says whether there is another, without a count.
    page = service.list_comparisons(limit=limit + 1, offset=offset)
    shown = page[:limit]
    continued = service.continued_sides(shown)
    return {
        "comparisons": [_response(service, c, continued) for c in shown],
        "hasMore": len(page) > limit,
    }


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_comparison(body: ComparisonCreateRequest, scope: ScopeDep):
    """Copies the source project's files into each side's sandbox, so it runs
    as one unit on the file store's threads (run_efs_unit)."""
    return await run_efs_unit(partial(_create_comparison, body=body), scope=scope)


def _create_comparison(session: ScopedSession, *, body: ComparisonCreateRequest) -> dict:
    """ComparisonService.create_comparison, as one unit."""
    service = ComparisonService(session)
    try:
        comparison = service.create_comparison(
            title=body.title,
            sides=[SideSpec(model=s.model, reasoning_effort=s.reasoning_effort) for s in body.sides],
            source_project_id=body.source_project_id,
        )
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ProjectTooLargeToCopyError as e:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(e))
    except ProjectNameLockBusyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _response(service, comparison)


@router.get("/{comparison_id}")
def get_comparison(comparison_id: UUID, scoped: ScopedSessionDep):
    service = ComparisonService(scoped)
    return _response(service, _get(service, comparison_id))


class _SideTurns(NamedTuple):
    """What comparison_usage asks the gateway about one side, in its order."""

    label: str
    conversation_id: str
    turn_starts: list[datetime]
    turn_limit: int | None


def _usage_sides(session: ScopedSession, *, comparison_id: UUID) -> list[_SideTurns]:
    """Each side's conversation and turn starts, as one unit."""
    service = ComparisonService(session)
    comparison = _get(service, comparison_id)
    return [
        _SideTurns(side.label, str(side.conversation_id), service.turn_starts(side), side.continued_turn_count)
        for side in sorted(comparison.sides, key=lambda s: s.label)
    ]


def _record_usage(session: ScopedSession, *, comparison_id: UUID, usage: dict) -> None:
    """Keep each side's cost as just read, as one unit."""
    service = ComparisonService(session)
    service.record_usage(_get(service, comparison_id), usage)


@router.get("/{comparison_id}/usage")
async def get_comparison_usage(comparison_id: UUID, request: Request, scope: ScopeDep):
    """Tokens and list-price cost per turn for each side, read from the gateway as the caller.

    The comparison is read in one unit and the cost kept in another, so no
    pooled connection is held while the gateway answers."""
    sides = await run_db(partial(_usage_sides, comparison_id=comparison_id), scope=scope)
    usage = await comparison_usage(sides, bearer_token=mindshub_credential(request))
    await run_db(partial(_record_usage, comparison_id=comparison_id, usage=usage.sides), scope=scope)
    return usage.model_dump(by_alias=True)


@router.delete("/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comparison(comparison_id: UUID, scope: ScopeDep):
    # The sides are read in one unit and the delete runs in another, so no
    # pooled connection is held while another replica is asked about them.
    # The delete removes the sandbox folders, so it runs on the file store's
    # threads (run_efs_unit).
    conversation_ids = await run_db(
        lambda session: [
            str(side.conversation_id)
            for side in _get(ComparisonService(session), comparison_id).sides
            if side.continued_at is None
        ],
        scope=scope,
    )
    running = await _running_elsewhere(conversation_ids, scope)
    try:
        await run_efs_unit(
            lambda session: ComparisonService(session).delete_comparison(
                comparison_id, running_elsewhere=running
            ),
            scope=scope,
        )
    except ComparisonNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ComparisonConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.put("/{comparison_id}/verdicts/{turn_index}")
def record_verdict(comparison_id: UUID, turn_index: int, body: ComparisonVerdictRequest, scoped: ScopedSessionDep):
    service = ComparisonService(scoped)
    try:
        service.record_verdict(comparison_id, turn_index=turn_index, winner=body.winner)
    except ComparisonNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _response(service, _get(service, comparison_id))


class _Continued(NamedTuple):
    """A continued side's task, read before its unit ends."""

    conversation_id: str
    project_id: str
    carried_all: bool


def _continue_side(
    session: ScopedSession,
    *,
    comparison_id: UUID,
    label: str,
    body: ComparisonContinueRequest,
    running: frozenset[str],
) -> _Continued:
    """ComparisonService.continue_side, as one unit."""
    continued = ComparisonService(session).continue_side(
        comparison_id,
        label,
        body.project_id,
        model_label=body.model_label,
        running_elsewhere=running,
    )
    conversation = continued.conversation
    return _Continued(str(conversation.id), str(conversation.project_id), continued.carried_all)


@router.post("/{comparison_id}/sides/{label}/continue")
async def continue_side(
    comparison_id: UUID,
    body: ComparisonContinueRequest,
    scope: ScopeDep,
    label: str = Path(pattern=_SIDE_LABEL),
):
    """Continue with one side: its task and the work it made move out of its
    sandbox into the project the body names (ComparisonService.continue_side).
    ``carriedAll`` is false while some of the side's work stays in the
    sandbox; continuing again into the same project carries the rest."""
    # The side is read in one unit and continued in another, so no pooled
    # connection is held while another replica is asked about it. Continuing
    # moves the side's work out of its sandbox, so it runs on the file
    # store's threads (run_efs_unit).
    conversation_ids = await run_db(
        lambda session: [
            str(side.conversation_id)
            for side in _get(ComparisonService(session), comparison_id).sides
            if side.label == label
        ],
        scope=scope,
    )
    running = await _running_elsewhere(conversation_ids, scope)
    try:
        continued = await run_efs_unit(
            partial(
                _continue_side, comparison_id=comparison_id, label=label, body=body, running=running
            ),
            scope=scope,
        )
    except (ComparisonNotFoundError, ProjectNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ComparisonConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    # carriedAll false: some of the side's work stayed in the comparison;
    # continuing again into the same project carries the rest.
    return {
        "conversationId": continued.conversation_id,
        "projectId": continued.project_id,
        "carriedAll": continued.carried_all,
    }
