"""People in the caller's MindsHub organization, read as them (ENG-3787).

Two reads of the auth service's ``GET /v1/me/org-members/``, both with the
caller's own bearer against the operator's host (same rules as
``hub_workspaces``):

- no params: the whole active-org directory (the auth service caps it at 500
  and says ``truncated``), the caller excluded. What the share dialog lists and
  filters locally. Never cached here: the app is its only reader and keeps
  its own copy, so a second one here would only make the app's copy older.
- ``?ids=a,b``: which of those Keycloak user ids are members of the active
  org, with their names and addresses. What sharing a skill with people checks
  a newly added id against, and what the app names stored ids with
  (``GET /hub/people/?ids=``). Never cached: a refusal has to reflect who is
  in the org now.

Every failure is "unreachable", never an exception, including the auth
service's 403 for a caller who is not a current member of the active org. The directory renders
nobody on it; the id check turns it into a 503 at the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from cowork.services.hub_workspaces import get_auth_json

logger = logging.getLogger(__name__)

ORG_MEMBERS_PATH = "/me/org-members/"

# The auth service's bound on one ``ids`` lookup; a longer batch is chunked
# rather than sent to be refused.
MAX_IDS_PER_LOOKUP = 50


class OrgMember(BaseModel):
    """One member, as the auth service puts it on the wire."""

    id: str
    name: str = ""
    email: str = ""


class OrgDirectory(BaseModel):
    """The whole active-org directory. ``reachable`` false means the auth
    service could not be asked; ``truncated`` is the auth service's: it held
    more members than it returns."""

    people: list[OrgMember] = Field(default_factory=list)
    reachable: bool = False
    truncated: bool = False


def _rows(payload: Any) -> list[OrgMember] | None:
    """``{"results": [...]}`` or a bare list; None when it is neither. A row
    with no id is dropped."""
    if payload is None:
        return None
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    members: list[OrgMember] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            members.append(OrgMember.model_validate({**row, "id": str(row["id"])}))
        except Exception as exc:
            logger.debug("dropping an unreadable org member row: %s", exc)
    return members


async def org_members_by_ids(
    *, bearer_token: str, ids: Iterable[str]
) -> list[OrgMember] | None:
    """The members of the caller's active org among ``ids`` (Keycloak user
    ids, compared lowercased), or None when the auth service couldn't be
    asked. One call per MAX_IDS_PER_LOOKUP ids; any chunk failing makes the
    whole answer None. An id it doesn't return is not a member."""
    wanted = list(dict.fromkeys(i.strip().lower() for i in ids if i and i.strip()))
    if not wanted:
        return []
    if not bearer_token:
        return None
    found: dict[str, OrgMember] = {}
    for start in range(0, len(wanted), MAX_IDS_PER_LOOKUP):
        chunk = wanted[start : start + MAX_IDS_PER_LOOKUP]
        path = f"{ORG_MEMBERS_PATH}?{urlencode({'ids': ','.join(chunk)})}"
        members = _rows(await get_auth_json(path, bearer_token))
        if members is None:
            return None
        for member in members:
            key = member.id.strip().lower()
            if key in wanted:
                found.setdefault(key, member)
    return [found[i] for i in wanted if i in found]


async def list_org_directory(*, bearer_token: str) -> OrgDirectory:
    """Everyone in the caller's active org but the caller (the auth service's
    parameterless read), read now."""
    if not bearer_token:
        return OrgDirectory()
    payload = await get_auth_json(ORG_MEMBERS_PATH, bearer_token)
    members = _rows(payload)
    if members is None:
        return OrgDirectory()
    return OrgDirectory(
        people=members,
        reachable=True,
        truncated=isinstance(payload, dict) and payload.get("truncated") is True,
    )
