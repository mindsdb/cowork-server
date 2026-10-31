"""GET /hub/teams/ — the MindsHub teams the caller belongs to.

What the skill sharing picker offers (ENG-3787): a skill can be shared only
with a team its sharer is in. Read from auth's ``GET /v1/me/teams/`` with the
caller's own bearer, cached per caller (cowork.services.hub_teams).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import hub_credential
from cowork.schemas.hub_teams import HubTeamRow, HubTeamsView
from cowork.services.hub_teams import fetch_hub_teams

# Gated as the hub workspaces router is: auth answers per caller, and refuses
# an absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=HubTeamsView)
async def get_hub_teams(request: Request, scope: ScopeDep) -> HubTeamsView:
    listing = await fetch_hub_teams(
        bearer_token=hub_credential(request),
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
    return HubTeamsView(
        reachable=listing.reachable,
        teams=[
            HubTeamRow(id=team.id, display_name=team.display_name, is_default=team.is_default)
            for team in listing.teams
        ],
    )
