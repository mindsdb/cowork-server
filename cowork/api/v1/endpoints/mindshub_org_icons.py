"""GET /hub/org-icons: the icons of the caller's MindsHub organizations, for
the org switcher (ENG-4020).

Read from auth's ``GET /v1/organizations/available/`` with the caller's own
credential (cowork.principal.hub_credential_for: the gateway's Authorization
in hosted org mode, the desktop app's X-MindsHub-Authorization on a local
server) and cached per caller for 60 s (cowork.services.mindshub_org_icons).
Answers ``{"reachable", "orgs": [{"id", "iconUrl"}]}``: each org by its
Keycloak id, its icon an absolute https URL or null. Without a MindsHub
credential (a local server signed out, or no org in scope) the answer is
empty and reachable, as GET /hub/teams/ answers outside MindsHub mode.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import hub_credential_for
from cowork.schemas.mindshub_org_icons import MindsHubOrgIconRow, MindsHubOrgIconsView
from cowork.services.mindshub_org_icons import fetch_org_icons

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("", response_model=MindsHubOrgIconsView)
async def get_org_icons(request: Request, scope: ScopeDep) -> MindsHubOrgIconsView:
    icons = await fetch_org_icons(
        bearer_token=hub_credential_for(request, org_mode=scope.org_mode, org_id=scope.org_id),
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
    )
    return MindsHubOrgIconsView(
        reachable=icons.reachable,
        orgs=[MindsHubOrgIconRow(id=org.id, icon_url=org.icon_url) for org in icons.orgs],
    )
