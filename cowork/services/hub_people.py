"""People in the caller's MindsHub organization, read as them (ENG-3787).

Two reads of auth's ``GET /v1/me/org-members/``, both with the caller's own
bearer against the operator's host (same rules as ``hub_workspaces``):

- ``?q=<text>&limit=<n>``: the skill sharing picker's people search. Members
  of the caller's active org, the caller excluded. Cached briefly per caller
  per query, so typing back over a prefix doesn't ask again.
- ``?emails=a@x,b@y``: which of those addresses are members of the active
  org. What sharing a skill with people checks a newly added address against.
  Never cached: a refusal has to reflect who is in the org now.

Every failure is "unreachable", never an exception. The search renders
nothing on it; the email check turns it into a 503 at the caller.
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

# Auth's own bounds: a query shorter than this is refused there, so it is
# answered empty here without asking; a limit and a batch past these are
# clamped and chunked rather than sent to be refused.
MIN_QUERY_CHARS = 2
MAX_LIMIT = 20
DEFAULT_LIMIT = 10
MAX_EMAILS_PER_LOOKUP = 50

# Short on purpose: the search is keystroke-driven, and a member who just
# joined should show up within a breath.
_TTL_OK = 30.0
_TTL_FAIL = 10.0
_MAX_TTL_S = max(_TTL_OK, _TTL_FAIL)


class OrgMember(BaseModel):
    """One member, as auth puts it on the wire."""

    id: str
    name: str = ""
    email: str = ""


class OrgMemberSearch(BaseModel):
    """A people search. ``reachable`` false means auth could not be asked."""

    people: list[OrgMember] = Field(default_factory=list)
    reachable: bool = False


_search_cache: dict[
    tuple[tuple[str, str, str, str], str, int], tuple[float, OrgMemberSearch]
] = {}


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


def _clamp_limit(limit: int | None) -> int:
    if not limit or limit < 1:
        return DEFAULT_LIMIT
    return min(limit, MAX_LIMIT)


async def search_org_members(
    *, bearer_token: str, org_id: str, user_id: str, q: str, limit: int | None = None
) -> OrgMemberSearch:
    """Members of the caller's active org matching ``q``. Under
    MIN_QUERY_CHARS answers empty and reachable without asking auth."""
    query = (q or "").strip()
    if len(query) < MIN_QUERY_CHARS:
        return OrgMemberSearch(reachable=True)
    if not bearer_token:
        return OrgMemberSearch()
    size = _clamp_limit(limit)
    key = (
        cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token),
        query.lower(),
        size,
    )
    entry = _search_cache.get(key)
    if entry is not None:
        stamped, value = entry
        if (time.monotonic() - stamped) < (_TTL_OK if value.reachable else _TTL_FAIL):
            return value
    path = f"{ORG_MEMBERS_PATH}?{urlencode({'q': query, 'limit': size})}"
    members = _rows(await get_auth_json(path, bearer_token))
    result = (
        OrgMemberSearch()
        if members is None
        else OrgMemberSearch(people=members, reachable=True)
    )
    sweep_cache(_search_cache, max_ttl_s=_MAX_TTL_S)
    _search_cache[key] = (time.monotonic(), result)
    return result


async def org_members_among(
    *, bearer_token: str, emails: Iterable[str]
) -> frozenset[str] | None:
    """Which of ``emails`` (lowercase) belong to the caller's active org, or
    None when auth couldn't be asked. One call per MAX_EMAILS_PER_LOOKUP
    addresses; any chunk failing makes the whole answer None."""
    wanted = list(dict.fromkeys(e.strip().lower() for e in emails if e and e.strip()))
    if not wanted:
        return frozenset()
    if not bearer_token:
        return None
    found: set[str] = set()
    for start in range(0, len(wanted), MAX_EMAILS_PER_LOOKUP):
        chunk = wanted[start : start + MAX_EMAILS_PER_LOOKUP]
        path = f"{ORG_MEMBERS_PATH}?{urlencode({'emails': ','.join(chunk)})}"
        members = _rows(await get_auth_json(path, bearer_token))
        if members is None:
            return None
        found.update(m.email.strip().lower() for m in members if m.email)
    return frozenset(found)


def reset_caches_for_tests() -> None:
    _search_cache.clear()
