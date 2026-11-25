"""The caller's notification inbox (cowork.services.notices).

Works in every deployment: the desktop, self-hosted and MindsHub. Clients
poll the unread count; there is no push channel for the inbox.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.notices import NoticeListResponse, NoticeResponse, UnreadCountResponse
from cowork.services.notices import KEEP_PER_PERSON, NoticeService

# AuthenticatedInOrgMode, declared explicitly as pins does: ScopedSessionDep
# already fails closed in org mode without an org in scope.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])


@router.get("/")
def list_notices(
    scoped: ScopedSessionDep, limit: int = Query(default=50, ge=1, le=KEEP_PER_PERSON),
) -> NoticeListResponse:
    """The caller's notices, newest first, with their unread count."""
    service = NoticeService(scoped)
    return NoticeListResponse(
        notices=[NoticeResponse.model_validate(n, from_attributes=True) for n in service.list_notices(limit)],
        unread_count=service.unread_count(),
    )


@router.get("/unread-count")
def unread_count(scoped: ScopedSessionDep) -> UnreadCountResponse:
    """How many of the caller's notices are unread. Cheap enough to poll."""
    return UnreadCountResponse(unread_count=NoticeService(scoped).unread_count())


@router.post("/read-all")
def mark_all_read(scoped: ScopedSessionDep) -> dict[str, int]:
    """Mark every one of the caller's notices read."""
    return {"updated": NoticeService(scoped).mark_all_read()}


@router.post("/{notice_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_read(notice_id: UUID, scoped: ScopedSessionDep) -> Response:
    """Mark one of the caller's notices read. 404 for anyone else's."""
    if not NoticeService(scoped).mark_read(notice_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notice not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
