from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.session import get_open_session
from cowork.principal import Principal, get_principal
from cowork.schemas.memory import (
    MemoryDeleteRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)
from cowork.services.memory import MemoryService

router = APIRouter()


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


@router.put("/", response_model=MemoryResponse)
def update_memory(
    body: MemoryUpdateRequest,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
):
    try:
        with _memory_service(scope, principal) as service:
            return service.update_memory_sync(
                scope=body.scope,
                category=body.category,
                content=body.content,
                project_id=body.project_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/")
def delete_memory(
    body: MemoryDeleteRequest,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
):
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
