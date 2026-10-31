"""GET /hub/people/: the people in the caller's MindsHub organization.

Three reads, each of MindsHub's ``GET /v1/me/org-members/`` with the
Authorization the gateway validated (cowork.principal.caller_bearer), never
the client-set X-MindsHub-Authorization, which could name another org's
members (cowork.services.mindshub_people):

- no params: the whole org directory (capped by MindsHub at MAX_DIRECTORY_ROWS,
  ``truncated`` passed through), the caller excluded, read uncached. What
  the share dialog loads once, keeps and filters locally.
- ``?ids=a,b``: the members among those user ids, with names and addresses,
  so the app can name the ids a skill's audience stores. Uncached.
- ``?emails=a,b``: the members who hold those addresses, so the app can share
  with a member its capped directory doesn't list by id rather than as an
  outside address. Uncached. Not with ``ids`` (400).

Every value is checked before MindsHub is asked, by the rule MindsHub applies,
because MindsHub refuses a whole lookup for one value it doesn't take:

- an id that isn't a UUID answers 400 ``{"code": "invalid_id", "detail",
  "ids": [...]}`` naming every such id; an id is asked in its canonical
  lowercase form;
- an address MindsHub wouldn't take
  (cowork.services.mindshub_people.is_email_address) answers 400
  ``{"code": "invalid_email", "detail", "emails": [...]}`` naming every such
  address;
- more than MAX_IDS_PER_REQUEST ids or MAX_EMAILS_PER_REQUEST addresses
  answers 400 (cowork.services.sharing_contract).

Outside MindsHub mode there is no organization to read, so the answer is
empty and reachable. MindsHub that couldn't be asked is ``reachable: false``;
a 4xx from MindsHub, such as its 403 for a caller who is not a current member
of the active org, is MindsHubRefused (cowork.services.mindshub_client),
never ``reachable: false``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import caller_bearer
from cowork.schemas.mindshub_people import MindsHubPeopleView, MindsHubPersonRow
from cowork.services.mindshub_client import canonical_uuid
from cowork.services.mindshub_people import (
    OrgMember,
    is_email_address,
    list_org_directory,
    org_members_by_emails,
    org_members_by_ids,
)
from cowork.services.sharing_contract import (
    MAX_EMAILS_PER_REQUEST,
    MAX_EMAILS_QUERY_CHARS,
    MAX_IDS_PER_REQUEST,
    MAX_IDS_QUERY_CHARS,
)

# Gated as the teams router is: MindsHub answers per caller, and refuses an
# absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

INVALID_ID = "invalid_id"
INVALID_EMAIL = "invalid_email"


def _listed(*, raw: str) -> list[str]:
    return [part for part in (p.strip() for p in raw.split(",")) if part]


def _refused(*, code: str, detail: str, field: str, values: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"code": code, "detail": detail, field: values},
    )


def _too_many(*, most: int, what: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ask about at most {most} {what} at once."
    )


def _person_rows(members: Iterable[OrgMember]) -> list[MindsHubPersonRow]:
    """MindsHub's member rows as this route answers them."""
    return [MindsHubPersonRow.model_validate(member, from_attributes=True) for member in members]


def _found(*, members: list[OrgMember] | None) -> MindsHubPeopleView:
    return MindsHubPeopleView(reachable=members is not None, people=_person_rows(members or ()))


@router.get("/", response_model=MindsHubPeopleView)
async def get_mindshub_people(
    request: Request,
    scope: ScopeDep,
    # Comma-separated user ids.
    ids: Annotated[str, Query(max_length=MAX_IDS_QUERY_CHARS)] = "",
    # Comma-separated addresses.
    emails: Annotated[str, Query(max_length=MAX_EMAILS_QUERY_CHARS)] = "",
) -> MindsHubPeopleView | JSONResponse:
    listed_ids, listed_emails = _listed(raw=ids), _listed(raw=emails)
    if listed_ids and listed_emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Send ids or emails, not both."
        )
    if scope is None or not scope.org_mode:
        return MindsHubPeopleView(reachable=True, truncated=False)
    if listed_ids:
        canonical = {value: canonical_uuid(value) for value in listed_ids}
        malformed = [value for value, parsed in canonical.items() if parsed is None]
        if malformed:
            return _refused(
                code=INVALID_ID, detail="Each id must be a UUID.", field="ids", values=malformed
            )
        wanted_ids = list(dict.fromkeys(parsed for parsed in canonical.values() if parsed))
        if len(wanted_ids) > MAX_IDS_PER_REQUEST:
            raise _too_many(most=MAX_IDS_PER_REQUEST, what="ids")
        return _found(
            members=await org_members_by_ids(bearer_token=caller_bearer(request), ids=wanted_ids)
        )
    if listed_emails:
        wanted_emails = list(dict.fromkeys(email.lower() for email in listed_emails))
        malformed = [email for email in wanted_emails if not is_email_address(email)]
        if malformed:
            return _refused(
                code=INVALID_EMAIL,
                detail="Each entry must be an email address.",
                field="emails",
                values=malformed,
            )
        if len(wanted_emails) > MAX_EMAILS_PER_REQUEST:
            raise _too_many(most=MAX_EMAILS_PER_REQUEST, what="email addresses")
        return _found(
            members=await org_members_by_emails(
                bearer_token=caller_bearer(request), emails=wanted_emails
            )
        )
    directory = await list_org_directory(bearer_token=caller_bearer(request))
    return MindsHubPeopleView(
        reachable=directory.reachable,
        people=_person_rows(directory.people),
        truncated=directory.truncated,
    )
