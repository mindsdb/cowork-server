"""GET /hub/people/ — the people in the caller's MindsHub organization.

Two reads (ENG-3787), both of auth's ``GET /v1/me/org-members/`` with the
caller's own bearer (cowork.services.hub_people):

- ``?q=<text>``: what the skill sharing picker offers when sharing with
  people (a skill can be shared only with members of the sharer's org),
  cached briefly per caller per query.
- ``?ids=a,b``: the members among those Keycloak user ids, with names and
  addresses, so the app can name the ids a skill's access stores. Uncached.
- ``?all=1``: the whole org directory (capped by auth, ``truncated`` passed
  through), the caller excluded, cached per caller for five minutes.

Precedence: ``ids``, then ``all``, then ``q``.

Outside org mode there is no organization to search, so the answer is empty
and reachable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import hub_credential
from cowork.schemas.hub_people import HubPeopleView, HubPersonRow
from cowork.services.hub_people import (
    DEFAULT_LIMIT,
    MAX_IDS_PER_LOOKUP,
    list_org_directory,
    org_members_by_ids,
    search_org_members,
)

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
    # Comma-separated Keycloak user ids; at most four lookups' worth.
    ids: Annotated[str, Query(max_length=40 * 4 * MAX_IDS_PER_LOOKUP)] = "",
    # The whole org directory: ``all=1`` (any truthy form).
    everyone: Annotated[bool, Query(alias="all")] = False,
) -> HubPeopleView:
    if scope is None or not scope.org_mode:
        return HubPeopleView(reachable=True, truncated=False)
    wanted = [part for part in (p.strip() for p in ids.split(",")) if part]
    if wanted:
        members = await org_members_by_ids(bearer_token=hub_credential(request), ids=wanted)
        return HubPeopleView(
            reachable=members is not None,
            people=[
                HubPersonRow(id=p.id, name=p.name, email=p.email) for p in members or ()
            ],
        )
    if everyone:
        directory = await list_org_directory(
            bearer_token=hub_credential(request),
            org_id=scope.org_id or "",
            user_id=scope.user_id or "",
        )
        return HubPeopleView(
            reachable=directory.reachable,
            people=[
                HubPersonRow(id=p.id, name=p.name, email=p.email) for p in directory.people
            ],
            truncated=directory.truncated,
        )
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
