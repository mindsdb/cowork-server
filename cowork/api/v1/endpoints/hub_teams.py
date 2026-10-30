"""GET /hub/teams/: the MindsHub teams the caller belongs to.

What the skill sharing picker offers (ENG-3787): a skill can be shared only
with a team its sharer is in. Read from the auth service's
``GET /v1/me/teams/`` with the Authorization the gateway validated (cowork.principal.caller_bearer), never
the client-set X-MindsHub-Authorization, which could list another org's
teams; cached per caller (cowork.services.hub_teams). Outside org mode there
are no teams to share with, so the answer is empty and reachable, as
GET /hub/people/ answers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import caller_bearer
from cowork.schemas.hub_teams import HubTeamRow, HubTeamsView
from cowork.services.hub_teams import fetch_hub_teams

# Gated as the hub workspaces router is: the auth service answers per caller,
# and refuses an absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=HubTeamsView)
async def get_hub_teams(request: Request, scope: ScopeDep) -> HubTeamsView:
    if scope is None or not scope.org_mode:
        return HubTeamsView(reachable=True, teams=[])
    listing = await fetch_hub_teams(
        bearer_token=caller_bearer(request),
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
