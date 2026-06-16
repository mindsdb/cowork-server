from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.memory import (
    MemoryDeleteRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)
from cowork.services.memory import MemoryService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[MemoryResponse])
async def list_memory(
    session: SessionDep,
    project_id: UUID | None = Query(default=None),
):
    return await MemoryService(session).list_memory(project_id=project_id)


@router.put("/", response_model=MemoryResponse)
async def update_memory(body: MemoryUpdateRequest, session: SessionDep):
    try:
        return await MemoryService(session).update_memory(
            scope=body.scope,
            category=body.category,
            content=body.content,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/")
async def delete_memory(body: MemoryDeleteRequest, session: SessionDep):
    try:
        await MemoryService(session).delete_memory(
            scope=body.scope,
            category=body.category,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"ok": True}

