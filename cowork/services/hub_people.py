"""People in the caller's MindsHub organization, read as them (ENG-3787).

Two reads of auth's ``GET /v1/me/org-members/``, both with the caller's own
bearer against the operator's host (same rules as ``hub_workspaces``):

- no params: the whole active-org directory (auth caps it at 500 and says
  ``truncated``), the caller excluded. What the share dialog lists and
  filters locally. Cached per caller for a few minutes.
- ``?ids=a,b``: which of those Keycloak user ids are members of the active
  org, with their names and addresses. What sharing a skill with people checks
  a newly added id against, and what the app names stored ids with
  (``GET /hub/people/?ids=``). Never cached: a refusal has to reflect who is
  in the org now.

Every failure is "unreachable", never an exception, including auth's 403 for a
caller who is not a current member of the active org. The directory renders
nobody on it; the id check turns it into a 503 at the caller.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from cowork.services.hub_workspaces import cache_key, get_auth_json, sweep_cache

logger = logging.getLogger(__name__)

ORG_MEMBERS_PATH = "/me/org-members/"

# Auth's bound on one ``ids`` lookup; a longer batch is chunked rather than
# sent to be refused.
MAX_IDS_PER_LOOKUP = 50

# The directory changes rarely, so it is kept a few minutes; a failure is
# retried soon.
_DIRECTORY_TTL_OK = 300.0
_DIRECTORY_TTL_FAIL = 15.0
_DIRECTORY_MAX_TTL_S = max(_DIRECTORY_TTL_OK, _DIRECTORY_TTL_FAIL)


class OrgMember(BaseModel):
    """One member, as auth puts it on the wire."""

    id: str
    name: str = ""
    email: str = ""


class OrgDirectory(BaseModel):
    """The whole active-org directory. ``reachable`` false means auth could not
    be asked; ``truncated`` is auth's: it held more members than it returns."""

    people: list[OrgMember] = Field(default_factory=list)
    reachable: bool = False
    truncated: bool = False


_directory_cache: dict[tuple[str, str, str, str], tuple[float, OrgDirectory]] = {}


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
    ids, compared lowercased), or None when auth couldn't be asked. One call
    per MAX_IDS_PER_LOOKUP ids; any chunk failing makes the whole answer
    None. An id auth doesn't return is not a member."""
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


async def list_org_directory(*, bearer_token: str, org_id: str, user_id: str) -> OrgDirectory:
    """Everyone in the caller's active org but the caller (auth's
    parameterless read), cached per caller."""
    if not bearer_token:
        return OrgDirectory()
    key = cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    entry = _directory_cache.get(key)
    if entry is not None:
        stamped, value = entry
        ttl = _DIRECTORY_TTL_OK if value.reachable else _DIRECTORY_TTL_FAIL
        if (time.monotonic() - stamped) < ttl:
            return value
    payload = await get_auth_json(ORG_MEMBERS_PATH, bearer_token)
    members = _rows(payload)
    result = (
        OrgDirectory()
        if members is None
        else OrgDirectory(
            people=members,
            reachable=True,
            truncated=isinstance(payload, dict) and payload.get("truncated") is True,
        )
    )
    sweep_cache(_directory_cache, max_ttl_s=_DIRECTORY_MAX_TTL_S)
    _directory_cache[key] = (time.monotonic(), result)
    return result


def reset_caches_for_tests() -> None:
    _directory_cache.clear()
