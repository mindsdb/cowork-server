"""GET /hub/people/ — search the people in the caller's MindsHub organization.

What the skill sharing picker offers when sharing with people (ENG-3787): a
skill can be shared only with members of the sharer's org. Read from auth's
``GET /v1/me/org-members/?q=`` with the caller's own bearer, cached briefly
per caller per query (cowork.services.hub_people). Outside org mode there is
no organization to search, so the answer is empty and reachable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import hub_credential
from cowork.schemas.hub_people import HubPeopleView, HubPersonRow
from cowork.services.hub_people import DEFAULT_LIMIT, search_org_members

# Gated as the hub teams router is: auth answers per caller, and refuses an
# absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=HubPeopleView)
async def get_hub_people(
    request: Request,
    scope: ScopeDep,
    q: Annotated[str, Query(max_length=200)] = "",
    # Clamped to auth's bound (hub_people.MAX_LIMIT) rather than refused.
    limit: Annotated[int, Query(ge=1)] = DEFAULT_LIMIT,
) -> HubPeopleView:
    if scope is None or not scope.org_mode:
        return HubPeopleView(reachable=True)
    found = await search_org_members(
        bearer_token=hub_credential(request),
        org_id=scope.org_id or "",
        user_id=scope.user_id or "",
        q=q,
        limit=limit,
    )
    return HubPeopleView(
        reachable=found.reachable,
        people=[
            HubPersonRow(id=p.id, name=p.name, email=p.email) for p in found.people
        ],
    )
