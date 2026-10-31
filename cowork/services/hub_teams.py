"""The caller's MindsHub teams and skill-sharing permissions, read as them.

Two auth reads for skill sharing in hosted org mode (ENG-3787):
``GET /v1/me/teams/`` for the teams the caller belongs to, and
``GET /v1/entitlements/me/`` for ``permissions[]``, where auth reports
``skills.share_org`` and ``skills.write_org``.

Same rules as ``hub_workspaces``: the bearer is the caller's own, the host is
the operator's, and the cache is keyed per credential per caller. Every failure
is "unreachable", never an exception, and callers fail closed on it: an
unreachable team listing grants no team skill, and unreachable permissions
grant neither capability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from cowork.services.hub_workspaces import cache_key, get_auth_json, sweep_cache

logger = logging.getLogger(__name__)

TEAMS_PATH = "/me/teams/"
ENTITLEMENTS_PATH = "/entitlements/me/"

# Capabilities auth reports in ``permissions[]``.
SKILLS_SHARE_ORG = "skills.share_org"
SKILLS_WRITE_ORG = "skills.write_org"

# The hub workspace listing's split TTL: one round trip per minute per caller
# when auth answers, and a quick retry when it did not.
_TTL_OK = 60.0
_TTL_FAIL = 15.0
_MAX_TTL_S = max(_TTL_OK, _TTL_FAIL)


class HubTeam(BaseModel):
    """One team, as auth puts it on the wire."""

    id: str
    display_name: str = ""
    is_default: bool = False


class HubTeamListing(BaseModel):
    """The caller's teams. ``reachable`` false means auth could not be asked,
    which is not the same as belonging to no team."""

    teams: list[HubTeam] = Field(default_factory=list)
    reachable: bool = False

    @property
    def team_ids(self) -> frozenset[str]:
        return frozenset(team.id for team in self.teams)

    @property
    def default_team_id(self) -> str:
        """The org's default team, whose members are the whole organization:
        sharing with it is sharing org-wide. Empty when auth named none."""
        return next((team.id for team in self.teams if team.is_default), "")


class HubPermissions(BaseModel):
    """The caller's resolved permissions in the active organization."""

    permissions: frozenset[str] = frozenset()
    reachable: bool = False


_teams_cache: dict[tuple[str, str, str, str], tuple[float, HubTeamListing]] = {}
_permissions_cache: dict[tuple[str, str, str, str], tuple[float, HubPermissions]] = {}


def _fresh(entry: tuple[float, Any] | None) -> Any | None:
    if not entry:
        return None
    stamped, value = entry
    return value if (time.monotonic() - stamped) < (_TTL_OK if value.reachable else _TTL_FAIL) else None


def _parse_teams(payload: Any) -> HubTeamListing:
    """``{"results": [...]}`` or a bare list; a row with no id is dropped."""
    if payload is None:
        return HubTeamListing()
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return HubTeamListing()
    teams: list[HubTeam] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            teams.append(HubTeam.model_validate({**row, "id": str(row["id"])}))
        except Exception as exc:
            logger.debug("dropping an unreadable team row: %s", exc)
    return HubTeamListing(teams=teams, reachable=True)


def _parse_permissions(payload: Any) -> HubPermissions:
    if not isinstance(payload, dict) or not isinstance(payload.get("permissions"), list):
        return HubPermissions()
    return HubPermissions(
        permissions=frozenset(str(p) for p in payload["permissions"] if isinstance(p, str)),
        reachable=True,
    )


async def fetch_hub_teams(*, bearer_token: str, org_id: str, user_id: str) -> HubTeamListing:
    """The teams this caller belongs to in their active organization."""
    if not bearer_token:
        return HubTeamListing()
    key = cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _fresh(_teams_cache.get(key))
    if cached is not None:
        return cached
    listing = _parse_teams(await get_auth_json(TEAMS_PATH, bearer_token))
    sweep_cache(_teams_cache, max_ttl_s=_MAX_TTL_S)
    _teams_cache[key] = (time.monotonic(), listing)
    return listing


async def fetch_hub_permissions(*, bearer_token: str, org_id: str, user_id: str) -> HubPermissions:
    """The caller's ``permissions[]`` from their entitlements."""
    if not bearer_token:
        return HubPermissions()
    key = cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _fresh(_permissions_cache.get(key))
    if cached is not None:
        return cached
    permissions = _parse_permissions(await get_auth_json(ENTITLEMENTS_PATH, bearer_token))
    sweep_cache(_permissions_cache, max_ttl_s=_MAX_TTL_S)
    _permissions_cache[key] = (time.monotonic(), permissions)
    return permissions


async def fetch_teams_and_permissions(
    *, bearer_token: str, org_id: str, user_id: str
) -> tuple[HubTeamListing, HubPermissions]:
    """Both reads at once; each fails closed on its own."""
    teams, permissions = await asyncio.gather(
        fetch_hub_teams(bearer_token=bearer_token, org_id=org_id, user_id=user_id),
        fetch_hub_permissions(bearer_token=bearer_token, org_id=org_id, user_id=user_id),
    )
    return teams, permissions


def reset_caches_for_tests() -> None:
    _teams_cache.clear()
    _permissions_cache.clear()
