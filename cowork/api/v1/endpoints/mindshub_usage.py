"""GET /hub/usage/: the caller's free tokens, balance, and auto top up state.

Read-only. Adding funds and changing auto top up stay in the MindsHub console;
the desktop deep-links there.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import run_db
from cowork.principal import mindshub_credential, minds_hub_configured
from cowork.schemas.mindshub_usage import MindsHubUsageView
from cowork.services.mindshub_usage import fetch_mindshub_usage
from cowork.services.settings import unit_setting_service

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode, declared explicitly: this route has no local check
# of its own. It relies on fetch_mindshub_usage's MindsHub calls (entitlements,
# wallet, usage-summary) rejecting an absent or invalid MindsHub credential
# (cowork/services/mindshub_usage.py), same shape as mindshub_workspaces.py.
@router.get("/", response_model=MindsHubUsageView, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def get_mindshub_usage(request: Request, scope: ScopeDep) -> MindsHubUsageView:
    settings = await run_db(lambda session: unit_setting_service(session).load(), scope=scope)
    # Local mode only: a self-hosted install with no MindsHub key never sends
    # the credential onward, whatever the desktop attached to the header.
    bearer = mindshub_credential(request) if scope.org_mode or minds_hub_configured(settings) else ""
    return await fetch_mindshub_usage(
        bearer_token=bearer,
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
