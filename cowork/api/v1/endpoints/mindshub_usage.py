"""GET /hub/usage/: the caller's free tokens, balance, and auto top up state.

Read-only. Adding funds and changing auto top up stay in the MindsHub console;
the desktop deep-links there.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import mindshub_credential
from cowork.schemas.mindshub_usage import MindsHubUsageView
from cowork.services.mindshub_usage import fetch_mindshub_usage

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode, declared explicitly: this route has no local check
# of its own. It relies on fetch_mindshub_usage's MindsHub calls (entitlements,
# wallet, usage-summary) rejecting an absent or invalid MindsHub credential
# (cowork/services/mindshub_usage.py), same shape as mindshub_workspaces.py.
@router.get("/", response_model=MindsHubUsageView, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def get_mindshub_usage(request: Request, scope: ScopeDep) -> MindsHubUsageView:
    return await fetch_mindshub_usage(
        bearer_token=mindshub_credential(request),
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
