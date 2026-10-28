from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.session import get_open_session
from cowork.principal import Principal, get_principal
from cowork.schemas.memory import (
    MemoryCapabilities,
    MemoryDeleteRequest,
    MemoryReset,
    MemoryResponse,
    MemoryUpdateRequest,
)
from cowork.services.memory import (
    PERSON_EDIT_REFUSED,
    SHARED_DELETE_REFUSED,
    MemoryService,
    is_person,
    person_may_reset_memory,
    person_owns_memory,
)

# AuthenticatedInOrgMode, declared explicitly: MemoryService already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py) —
# Project through ScopedSession, GlobalMemoryStore through
# scoped_user_storage_root — whenever org mode has no resolved org/user.
# Declaring it too makes the requirement visible to a route walker instead
# of something only discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])


@contextmanager
def _memory_service(
    scope: TenantScope,
    principal: Principal | None,
) -> Iterator[MemoryService]:
    raw_session = get_open_session()
    try:
        yield MemoryService(ScopedSession(raw_session, scope), principal)
    finally:
        raw_session.close()


@router.get("/", response_model=list[MemoryResponse])
def list_memory(
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    project_id: UUID | None = Query(default=None),
    principal: Principal | None = Depends(get_principal),
):
    with _memory_service(scope, principal) as service:
        return service.list_memory_sync(project_id=project_id)


@router.get("/capabilities", response_model=MemoryCapabilities)
def memory_capabilities(
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
) -> MemoryCapabilities:
    """What the caller may do to their memory as a whole, so the Memories page
    offers a reset only where DELETE /all would carry it out."""
    return MemoryCapabilities(can_reset=person_may_reset_memory(scope))


@router.put("/", response_model=MemoryResponse)
def update_memory(
    body: MemoryUpdateRequest,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
):
    # A person on a deployment that trusts identity headers never edits:
    # their agent saves memory on its own (MemoryService._capabilities).
    if is_person(scope):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=PERSON_EDIT_REFUSED)
    try:
        with _memory_service(scope, principal) as service:
            return service.update_memory_sync(
                scope=body.scope,
                category=body.category,
                content=body.content,
                project_id=body.project_id,
            )
    except UnicodeError as e:
        # A slot whose stored bytes do not decode is a broken shared resource,
        # not a bad request. UnicodeDecodeError is a ValueError, so without this
        # the unknown-project mapping below would answer 400 and name the
        # decoder's offset as if the caller had sent something wrong.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This memory slot exists but its stored bytes cannot be read",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/")
def delete_memory(
    body: MemoryDeleteRequest,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
):
    # Where memory is shared a person's slots are the deployment's one store,
    # so a person's delete there would remove everyone's.
    if is_person(scope) and not person_owns_memory(scope):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SHARED_DELETE_REFUSED)
    try:
        with _memory_service(scope, principal) as service:
            service.delete_memory_sync(
                scope=body.scope,
                category=body.category,
                project_id=body.project_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"ok": True}


@router.delete("/all", response_model=MemoryReset)
def reset_memory(
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
) -> MemoryReset:
    """Remove all of the caller's learned memory: their global tier and each
    project tier of theirs (MemoryService.reset_own_memory_sync)."""
    with _memory_service(scope, principal) as service:
        return MemoryReset(removed=service.reset_own_memory_sync().removed)
