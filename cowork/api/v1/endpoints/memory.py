from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cowork.db.scoped import ScopedSessionDep
from cowork.principal import Principal, get_principal
from cowork.schemas.memory import (
    MemoryDeleteRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)
from cowork.services.memory import MemoryService

router = APIRouter()

@router.get("/", response_model=list[MemoryResponse])
async def list_memory(
    scoped: ScopedSessionDep,
    project_id: UUID | None = Query(default=None),
    principal: Principal | None = Depends(get_principal),
):
    return await MemoryService(scoped, principal).list_memory(project_id=project_id)


@router.put("/", response_model=MemoryResponse)
async def update_memory(
    body: MemoryUpdateRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    try:
        return await MemoryService(scoped, principal).update_memory(
            scope=body.scope,
            category=body.category,
            content=body.content,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/")
async def delete_memory(
    body: MemoryDeleteRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    try:
        await MemoryService(scoped, principal).delete_memory(
            scope=body.scope,
            category=body.category,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"ok": True}
