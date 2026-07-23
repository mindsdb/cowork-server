"""Standing rules endpoint — list active rules, one-click revoke.

The Memories shelf (R2) renders from GET /rules/; revocation is checked at
act time by the gate's exact-match lookup, so a revoke takes effect on the
very next action.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.rules import StandingRuleResponse
from cowork.services.rules import RuleService

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_rules(session: SessionDep, include_revoked: bool = False) -> dict:
    rules = RuleService(session).list(include_revoked=include_revoked)
    return {"rules": [StandingRuleResponse.serialize(r) for r in rules]}


@router.post("/{rule_id}/revoke")
def revoke_rule(rule_id: UUID, session: SessionDep) -> dict:
    try:
        rule = RuleService(session).revoke(rule_id)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    return {"rule": StandingRuleResponse.serialize(rule)}
