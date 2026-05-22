from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.memory import MemoryDeleteRequest, MemoryResponse, MemoryScope, MemoryUpdateRequest
from cowork.services.memory import MemoryService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


# TODO: Refine these two endpoints. Is there a need to list one memory item?
@router.get("/list", response_model=list[MemoryResponse])
async def list_memory(session: SessionDep):
    return await MemoryService(session).list_memory()


@router.get("/", response_model=MemoryResponse)
async def get_memory(
    session: SessionDep,
    scope: MemoryScope,
    category: str,
    project_id: UUID | None = None,
):
    try:
        return await MemoryService(session).get_memory(scope, category, project_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


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


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(body: MemoryDeleteRequest, session: SessionDep):
    try:
        await MemoryService(session).delete_memory(
            scope=body.scope,
            category=body.category,
            project_id=body.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))