"""GET /hub/people/: the people in the caller's MindsHub organization.

Two reads (ENG-3787), both of the auth service's ``GET /v1/me/org-members/``
with the Authorization the gateway validated (cowork.principal.caller_bearer),
never the client-set X-MindsHub-Authorization, which could name another org's
members (cowork.services.hub_people):

- no params: the whole org directory (capped by the auth service,
  ``truncated`` passed through), the caller excluded, read uncached. What the share dialog loads
  once, keeps and filters locally.
- ``?ids=a,b``: the members among those Keycloak user ids, with names and
  addresses, so the app can name the ids a skill's access stores. Uncached.

Outside org mode there is no organization to read, so the answer is empty
and reachable. A caller the auth service refuses (not a current member of
the active org) is answered ``reachable: false``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import caller_bearer
from cowork.schemas.hub_people import HubPeopleView, HubPersonRow
from cowork.services.hub_people import (
    MAX_IDS_PER_LOOKUP,
    list_org_directory,
    org_members_by_ids,
)

# Gated as the hub teams router is: the auth service answers per caller, and
# refuses an absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=HubPeopleView)
async def get_hub_people(
    request: Request,
    scope: ScopeDep,
    # Comma-separated Keycloak user ids; at most four lookups' worth.
    ids: Annotated[str, Query(max_length=40 * 4 * MAX_IDS_PER_LOOKUP)] = "",
) -> HubPeopleView:
    if scope is None or not scope.org_mode:
        return HubPeopleView(reachable=True, truncated=False)
    wanted = [part for part in (p.strip() for p in ids.split(",")) if part]
    if wanted:
        members = await org_members_by_ids(bearer_token=caller_bearer(request), ids=wanted)
        return HubPeopleView(
            reachable=members is not None,
            people=[
                HubPersonRow(id=p.id, name=p.name, email=p.email) for p in members or ()
            ],
        )
    directory = await list_org_directory(bearer_token=caller_bearer(request))
    return HubPeopleView(
        reachable=directory.reachable,
        people=[
            HubPersonRow(id=p.id, name=p.name, email=p.email) for p in directory.people
        ],
        truncated=directory.truncated,
    )
