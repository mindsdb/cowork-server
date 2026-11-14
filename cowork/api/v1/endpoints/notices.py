"""The caller's in-app notification inbox (cowork.services.notices)."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.notices import NoticeListResponse, NoticeResponse, UnreadCountResponse
from cowork.services.notices import NoticeService

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])


@router.get("/", response_model=NoticeListResponse)
def list_notices(scoped: ScopedSessionDep, limit: int = Query(default=50, ge=1, le=200)):
    service = NoticeService(scoped)
    return NoticeListResponse(
        notices=[NoticeResponse.model_validate(n) for n in service.list_notices(limit=limit)],
        unread_count=service.unread_count(),
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
def unread_count(scoped: ScopedSessionDep):
    """What the app polls for its bell: one count, no rows."""
    return UnreadCountResponse(unread_count=NoticeService(scoped).unread_count())


@router.post("/{notice_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_read(notice_id: UUID, scoped: ScopedSessionDep) -> Response:
    if not NoticeService(scoped).mark_read(notice_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notice not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/read-all", response_model=UnreadCountResponse)
def mark_all_read(scoped: ScopedSessionDep):
    NoticeService(scoped).mark_all_read()
    return UnreadCountResponse(unread_count=0)
