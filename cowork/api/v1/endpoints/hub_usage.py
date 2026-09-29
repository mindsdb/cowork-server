"""GET /hub/usage/ — the caller's free tokens, balance, and auto top up state.

Read-only. Adding funds and changing auto top up stay in the MindsHub console;
the desktop deep-links there.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.principal import hub_credential, minds_hub_configured
from cowork.schemas.hub_usage import HubUsageView
from cowork.services.hub_usage import fetch_hub_usage
from cowork.services.settings import SettingService

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode, declared explicitly: this route has no local check
# of its own — it relies on fetch_hub_usage's auth calls (entitlements,
# wallet, usage-summary) rejecting an absent or invalid hub credential
# (cowork/services/hub_usage.py), same shape as hub_workspaces.py.
@router.get("/", response_model=HubUsageView, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def get_hub_usage(request: Request, session: SessionDep, scope: ScopeDep) -> HubUsageView:
    settings = SettingService(session, scope).load()
    # Local mode only: a self-hosted install with no MindsHub key never sends
    # the credential onward, whatever the desktop attached to the header.
    if scope.org_mode or minds_hub_configured(settings):
        bearer = hub_credential(request)
    else:
        bearer = ""
    return await fetch_hub_usage(
        bearer_token=bearer,
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
