from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.models.comparison import Comparison
from cowork.schemas.comparisons import (
    ComparisonContinueRequest,
    ComparisonCreateRequest,
    ComparisonResponse,
    ComparisonVerdictRequest,
)
from cowork.services.comparisons import (
    ComparisonConflictError,
    ComparisonNotFoundError,
    ComparisonService,
    ProjectTooLargeToCopyError,
    SideSpec,
)
from cowork.services.projects import ProjectNameLockBusyError, ProjectNotFoundError

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

_SIDE_LABEL = r"^[ab]$"


def _response(service: ComparisonService, comparison: Comparison) -> dict:
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
def list_comparisons(scoped: ScopedSessionDep, limit: int = 50):
    service = ComparisonService(scoped)
    limit = max(1, min(limit, 200))
    return {"comparisons": [_response(service, c) for c in service.list_comparisons(limit=limit)]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_comparison(body: ComparisonCreateRequest, scoped: ScopedSessionDep):
    service = ComparisonService(scoped)
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


@router.delete("/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comparison(comparison_id: UUID, scoped: ScopedSessionDep):
    service = ComparisonService(scoped)
    try:
        service.delete_comparison(comparison_id)
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


@router.post("/{comparison_id}/sides/{label}/continue")
def continue_side(
    comparison_id: UUID,
    body: ComparisonContinueRequest,
    scoped: ScopedSessionDep,
    label: str = Path(pattern=_SIDE_LABEL),
):
    service = ComparisonService(scoped)
    try:
        conversation = service.continue_side(comparison_id, label, body.project_id)
    except (ComparisonNotFoundError, ProjectNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ComparisonConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"conversationId": str(conversation.id), "projectId": str(conversation.project_id)}
