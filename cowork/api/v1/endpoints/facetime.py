"""/facetime: video calls with the agent (prototype).

GET  /facetime/status  whether this deployment offers video calls
POST /facetime/token   a session token for one call, which the web app starts within 60 seconds

cowork.services.facetime says how a call runs. A refusal keeps Nuance's 403,
409 or 429, with its code and a sentence for the person in ``detail``; any
other failure is 424.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.services import facetime

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/status", response_model=facetime.FacetimeStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def facetime_status() -> facetime.FacetimeStatus:
    return facetime.FacetimeStatus(available=facetime.available())


@router.post("/token", response_model=facetime.FacetimeToken, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def facetime_token(scope: ScopeDep) -> facetime.FacetimeToken:
    try:
        return await facetime.mint(facetime.end_user_id(scope.org_id, scope.user_id))
    except facetime.FacetimeError as exc:
        detail = {"code": exc.code, "message": exc.message}
        if exc.retry_after is not None:
            detail["retry_after_seconds"] = exc.retry_after
        raise HTTPException(status_code=exc.status, detail=detail) from exc
