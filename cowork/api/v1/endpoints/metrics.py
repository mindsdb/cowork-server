"""Metrics endpoint — the board's reliability claim, measured."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.services.metrics import approval_metrics

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/approvals")
def get_approval_metrics(session: SessionDep) -> dict:
    return approval_metrics(session)
