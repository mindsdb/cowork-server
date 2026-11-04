"""GET /hub/teams/: the MindsHub teams the caller belongs to.

What the sharing picker offers: a skill can be shared only with a team its
sharer is in. Read from MindsHub's ``GET /v1/me/teams/`` with the
Authorization the gateway validated (cowork.principal.caller_bearer), never
the client-set X-MindsHub-Authorization, which could list another org's
teams; cached per caller (cowork.services.mindshub_teams). Outside MindsHub
mode there are no teams to share with, so the answer is empty and reachable,
as GET /hub/people/ answers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import caller_bearer
from cowork.schemas.mindshub_teams import MindsHubTeamRow, MindsHubTeamsView
from cowork.services.mindshub_teams import fetch_mindshub_teams

# Gated as the workspaces router is: MindsHub answers per caller, and refuses
# an absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=MindsHubTeamsView)
async def get_mindshub_teams(request: Request, scope: ScopeDep) -> MindsHubTeamsView:
    if scope is None or not scope.org_mode:
        return MindsHubTeamsView(reachable=True, teams=[])
    listing = await fetch_mindshub_teams(
        bearer_token=caller_bearer(request),
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
    return MindsHubTeamsView(
        reachable=listing.reachable,
        teams=[MindsHubTeamRow.model_validate(team, from_attributes=True) for team in listing.teams],
    )
