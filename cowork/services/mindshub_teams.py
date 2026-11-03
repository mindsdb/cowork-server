"""The caller's MindsHub teams and skill-sharing permissions, read as them.

Two reads of MindsHub for sharing in MindsHub mode: ``GET /v1/me/teams/``
for the teams the caller belongs to, and ``GET /v1/entitlements/me/`` for
``permissions[]``, where it reports ``skills.share_org``,
``skills.write_org`` and ``artifacts.share_org``. A
third, ``GET /v1/organizations/current/teams/``, lists every team of the
caller's organization, uncached, for the check an artifact save makes before
it publishes when an admin adds a team they aren't in (org_team_ids). An
artifact save that adds a team reads the caller's teams past the cache too
(read_mindshub_teams_now), so a team deleted within the cache's TTL is
refused before anything is published.

Same rules as every MindsHub read (cowork.services.mindshub_client): the
bearer is the caller's own, the host is the operator's, and the cache is
keyed per credential per caller. How MindsHub answered decides what a read
returns, never an exception, and callers fail closed on what isn't known:

- MindsHub couldn't be asked (no answer, a 5xx, a body that isn't the list):
  ``reachable`` false. An unreachable team listing grants no team skill, and
  unreachable permissions grant no capability; a change that rests on them is
  a 503.
- MindsHub refused (a 4xx other than 429): it answered that the caller has
  none, so the read is ``reachable`` with no teams or no permissions.
- MindsHub throttled the caller (a 429): the last answer it gave for this
  caller is served again, for at most _MAX_THROTTLED_AGE_S after MindsHub gave
  it, and the throttle is logged. With no such answer the read is
  ``reachable`` false, as for an outage. The read an artifact save makes
  past the cache (read_mindshub_teams_now) logs the throttle too, and raises
  MindsHubRefused (429) rather than serve an earlier answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, Field

from cowork.services.mindshub_client import (
    ENTITLEMENTS_PATH,
    MINDSHUB_TTL_OK_S,
    CallerCacheKey,
    CallerTtlCache,
    MindsHubAnswer,
    MindsHubRefused,
    MindsHubReply,
    canonical_uuid,
    get_mindshub_reply,
)
from cowork.services.sharing_contract import ORG_TEAMS_PATH, TEAMS_PATH

logger = logging.getLogger(__name__)

# How long after MindsHub last answered its answer is served while MindsHub
# throttles the caller. Bounded, so a caller who keeps MindsHub's throttle
# full can't keep a team or permission MindsHub has since taken away: that
# lasts at most this long, against MINDSHUB_TTL_OK_S when MindsHub answers.
_MAX_THROTTLED_AGE_S = 2 * MINDSHUB_TTL_OK_S


class MindsHubTeam(BaseModel):
    """One team, as MindsHub puts it on the wire. ``id`` is the team's UUID
    in its canonical lowercase form."""

    id: str
    display_name: str = ""
    is_default: bool = False


class MindsHubTeamListing(BaseModel):
    """The caller's teams. ``reachable`` false means MindsHub could not be
    asked, which is not the same as belonging to no team."""

    teams: list[MindsHubTeam] = Field(default_factory=list)
    reachable: bool = False

    @property
    def team_ids(self) -> frozenset[str]:
        return frozenset(team.id for team in self.teams)

    @property
    def default_team_id(self) -> str:
        """The org's default team, whose members are everyone in the organization:
        sharing with it is sharing with everyone. Empty when MindsHub named
        none."""
        return next((team.id for team in self.teams if team.is_default), "")


class MindsHubPermissions(BaseModel):
    """The caller's resolved permissions in the active organization."""

    permissions: frozenset[str] = frozenset()
    reachable: bool = False


class TeamsAndPermissions(NamedTuple):
    """The caller's teams and permissions (fetch_teams_and_permissions)."""

    teams: MindsHubTeamListing
    permissions: MindsHubPermissions


# Each kept as long as a throttled read may serve it.
_teams_cache: CallerTtlCache[MindsHubTeamListing] = CallerTtlCache(keep_s=_MAX_THROTTLED_AGE_S)
_permissions_cache: CallerTtlCache[MindsHubPermissions] = CallerTtlCache(keep_s=_MAX_THROTTLED_AGE_S)

_Read = TypeVar("_Read", MindsHubTeamListing, MindsHubPermissions)


def _parse_teams(payload: Any) -> MindsHubTeamListing:
    """``{"results": [...]}`` or a bare list; a row with no id, or one whose
    id isn't a UUID, is dropped."""
    if payload is None:
        return MindsHubTeamListing()
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return MindsHubTeamListing()
    teams: list[MindsHubTeam] = []
    for row in rows:
        team_id = canonical_uuid(row.get("id")) if isinstance(row, dict) and row.get("id") else None
        if team_id is None:
            continue
        try:
            teams.append(MindsHubTeam.model_validate({**row, "id": team_id}))
        except Exception as exc:
            logger.debug("dropping an unreadable team row: %s", exc)
    return MindsHubTeamListing(teams=teams, reachable=True)


def _parse_permissions(payload: Any) -> MindsHubPermissions:
    if not isinstance(payload, dict) or not isinstance(payload.get("permissions"), list):
        return MindsHubPermissions()
    return MindsHubPermissions(
        permissions=frozenset(str(p) for p in payload["permissions"] if isinstance(p, str)),
        reachable=True,
    )


async def _cached_read(
    cache: CallerTtlCache[_Read],
    *,
    path: str,
    bearer_token: str,
    org_id: str,
    user_id: str,
    parse: Callable[[Any], _Read],
    unknown: Callable[[], _Read],
    none_granted: Callable[[], _Read],
) -> _Read:
    """``path`` read for this caller through ``cache``, by how MindsHub
    answered (the module docstring). ``parse`` reads an answer; ``unknown``
    is what couldn't be asked; ``none_granted`` is MindsHub's refusal, kept
    only the short TTL, since a credential that expired a moment ago is
    refused too."""
    key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    fresh = cache.fresh(key)
    if fresh is not None:
        return fresh
    reply = await get_mindshub_reply(path, bearer_token)
    if reply.answer is not MindsHubAnswer.THROTTLED:
        return _cache_answer(cache, key, reply, parse=parse, unknown=unknown, none_granted=none_granted)
    last_known = cache.answered_within(key, max_age_s=_MAX_THROTTLED_AGE_S)
    _log_throttled(
        path,
        outcome="served_last_known" if last_known is not None else "unknown",
        user_id=user_id,
        org_id=org_id,
    )
    if last_known is not None:
        cache.serve_again(key)
        return last_known
    value = unknown()
    cache.put(key, value)
    return value


def _log_throttled(path: str, *, outcome: str, user_id: str, org_id: str) -> None:
    """The one log line for a read of ``path`` MindsHub throttled, with what
    the read did about it (``outcome``)."""
    logger.warning(
        "MindsHub throttled a read made for the caller",
        extra={
            "event": "mindshub_throttled",
            "route": path,
            "outcome": outcome,
            "user_id": user_id or None,
            "organization_id": org_id or None,
        },
    )


def _cache_answer(
    cache: CallerTtlCache[_Read],
    key: CallerCacheKey,
    reply: MindsHubReply,
    *,
    parse: Callable[[Any], _Read],
    unknown: Callable[[], _Read],
    none_granted: Callable[[], _Read],
) -> _Read:
    """What ``reply``, any answer but a throttle, says for the caller,
    cached under ``key``: a refusal for the short TTL only."""
    if reply.answer is MindsHubAnswer.REFUSED:
        value, ttl_s = none_granted(), cache.ttl_fail_s
    elif reply.answer is MindsHubAnswer.OK:
        value, ttl_s = parse(reply.payload), None
    else:
        value, ttl_s = unknown(), None
    cache.put(key, value, ttl_s=ttl_s)
    return value


async def fetch_mindshub_teams(*, bearer_token: str, org_id: str, user_id: str) -> MindsHubTeamListing:
    """The teams this caller belongs to in their active organization."""
    if not bearer_token:
        return MindsHubTeamListing()
    return await _cached_read(
        _teams_cache, path=TEAMS_PATH, bearer_token=bearer_token, org_id=org_id, user_id=user_id,
        parse=_parse_teams, unknown=MindsHubTeamListing,
        none_granted=lambda: MindsHubTeamListing(reachable=True),
    )


async def read_mindshub_teams_now(*, bearer_token: str, org_id: str, user_id: str) -> MindsHubTeamListing:
    """The teams this caller belongs to, read from MindsHub now rather than
    from the cache, for a check that must see a team deleted a moment ago:
    an artifact save that adds a team. MindsHub's answer reads, and is
    cached, as fetch_mindshub_teams reads and caches it. A throttle is
    MindsHubRefused (429): the answer from before it is the one this read
    must not use."""
    if not bearer_token:
        return MindsHubTeamListing()
    reply = await get_mindshub_reply(TEAMS_PATH, bearer_token)
    if reply.answer is MindsHubAnswer.THROTTLED:
        _log_throttled(TEAMS_PATH, outcome="refused", user_id=user_id, org_id=org_id)
        raise MindsHubRefused(mindshub_status=reply.status_code)
    key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    return _cache_answer(
        _teams_cache, key, reply, parse=_parse_teams, unknown=MindsHubTeamListing,
        none_granted=lambda: MindsHubTeamListing(reachable=True),
    )


async def fetch_mindshub_permissions(*, bearer_token: str, org_id: str, user_id: str) -> MindsHubPermissions:
    """The caller's ``permissions[]`` from their entitlements."""
    if not bearer_token:
        return MindsHubPermissions()
    return await _cached_read(
        _permissions_cache, path=ENTITLEMENTS_PATH, bearer_token=bearer_token, org_id=org_id,
        user_id=user_id, parse=_parse_permissions, unknown=MindsHubPermissions,
        none_granted=lambda: MindsHubPermissions(reachable=True),
    )


async def fetch_teams_and_permissions(
    *, bearer_token: str, org_id: str, user_id: str
) -> TeamsAndPermissions:
    """Both reads at once; each fails closed on its own."""
    teams, permissions = await asyncio.gather(
        fetch_mindshub_teams(bearer_token=bearer_token, org_id=org_id, user_id=user_id),
        fetch_mindshub_permissions(bearer_token=bearer_token, org_id=org_id, user_id=user_id),
    )
    return TeamsAndPermissions(teams=teams, permissions=permissions)


async def org_team_ids(*, bearer_token: str) -> frozenset[str] | None:
    """Every team of the caller's active organization, read now, or None when
    MindsHub couldn't be asked; MindsHubRefused when it refused. Uncached:
    the check it serves must see a team deleted a moment ago."""
    if not bearer_token:
        return None
    reply = await get_mindshub_reply(ORG_TEAMS_PATH, bearer_token)
    if reply.answer in (MindsHubAnswer.REFUSED, MindsHubAnswer.THROTTLED):
        raise MindsHubRefused(mindshub_status=reply.status_code)
    if reply.answer is not MindsHubAnswer.OK:
        return None
    listing = _parse_teams(reply.payload)
    return listing.team_ids if listing.reachable else None


def reset_caches_for_tests() -> None:
    _teams_cache.clear()
    _permissions_cache.clear()
