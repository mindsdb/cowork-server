"""
Approvals endpoint — the approve-before-act queue.

List/get the parked proposals; resolve them (approve / edit / skip). On
resolution the harness executes the approved descriptor deterministically
(see services/approvals.py) — exactly what was on the card, once.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.approvals import ApprovalResolveRequest, ApprovalResponse
from cowork.services.approvals import ApprovalService

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_approvals(
    session: SessionDep,
    status: str | None = None,
    conversation_id: UUID | None = None,
) -> dict:
    service = ApprovalService(session)
    approvals = service.list(status=status, conversation_id=conversation_id)
    return {"approvals": [ApprovalResponse.serialize(a) for a in approvals]}


@router.get("/{approval_id}")
def get_approval(approval_id: UUID, session: SessionDep) -> dict:
    try:
        approval = ApprovalService(session).get(approval_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    return {"approval": ApprovalResponse.serialize(approval)}


@router.post("/{approval_id}/resolve")
def resolve_approval(approval_id: UUID, body: ApprovalResolveRequest, session: SessionDep) -> dict:
    try:
        approval, executed_now = ApprovalService(session).resolve(
            approval_id,
            resolution=body.resolution,
            edited_draft=body.edited_draft,
        )
    except ValueError as e:
        detail = str(e)
        raise HTTPException(404 if "not found" in detail.lower() else 400, detail=detail)
    return {
        "approval": ApprovalResponse.serialize(approval),
        # Idempotent double-resolve reports itself: the client can tell a
        # re-click from a fresh execution (which never happens twice).
        "alreadyResolved": not executed_now,
    }
