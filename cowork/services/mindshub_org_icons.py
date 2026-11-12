"""The icons of the organizations the caller belongs to.

Read from auth's ``GET /v1/organizations/available/`` with the caller's own
credential, cached per caller for 60 s (CallerTtlCache). auth answers each
org by its Keycloak ``id`` with ``brand_icon_url``, in ``results`` and
``blocked_organizations`` alike; both are listed, since the org switcher
shows both. Each URL is made browser-loadable as the branding route's are
(cowork.services.org_branding.absolute_auth_url): a root-relative one is
joined to the configured public auth origin, and any other scheme dropped.

The route takes only a session: auth refuses an API key there, so a
credential that isn't a session JWT is no organizations, never a request.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from cowork.principal import is_session_jwt
from cowork.services.mindshub_client import (
    MINDSHUB_TTL_OK_S,
    CallerCacheKey,
    CallerTtlCache,
    MindsHubAnswer,
    canonical_uuid,
    get_mindshub_reply,
)
from cowork.services.org_branding import absolute_auth_url

logger = logging.getLogger(__name__)

AVAILABLE_ORGS_PATH = "/organizations/available/"


class OrgIcon(BaseModel):
    id: str
    icon_url: str | None = None


class OrgIcons(BaseModel):
    """``reachable`` false: auth couldn't be asked."""

    reachable: bool = False
    orgs: list[OrgIcon] = Field(default_factory=list)


_cache: CallerTtlCache[OrgIcons] = CallerTtlCache(ttl_ok_s=MINDSHUB_TTL_OK_S)


def _parse(payload: dict[str, Any]) -> OrgIcons:
    orgs: list[OrgIcon] = []
    seen: set[str] = set()
    for key in ("results", "blocked_organizations"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            org_id = canonical_uuid(row.get("id"))
            if org_id is None or org_id in seen:
                continue
            seen.add(org_id)
            orgs.append(OrgIcon(id=org_id, icon_url=absolute_auth_url(row.get("brand_icon_url"))))
    return OrgIcons(reachable=True, orgs=orgs)


async def fetch_org_icons(*, bearer_token: str, org_id: str, user_id: str) -> OrgIcons:
    """The caller's organizations' icons. No session credential is no
    organizations, reachable, and so is auth's 403. Any other refusal (a 401
    for an expired session, a 404) says nothing about the caller's orgs, so
    it is ``reachable`` false, as an outage is, and not cached: the switcher
    keeps the icons it has. Any answer from auth with nothing usable logs
    ``org_icons_read_unknown`` with auth's status, at WARNING so every
    deployment prints it, as the branding read does. Never raises."""
    if not is_session_jwt(bearer_token):
        return OrgIcons(reachable=True)
    key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _cache.fresh(key)
    if cached is not None:
        return cached
    reply = await get_mindshub_reply(AVAILABLE_ORGS_PATH, bearer_token)
    if reply.answer is MindsHubAnswer.OK and isinstance(reply.payload, dict):
        value = _parse(reply.payload)
    elif reply.answer is MindsHubAnswer.REFUSED and reply.status_code == 403:
        value = OrgIcons(reachable=True)
    else:
        if reply.status_code is not None:
            logger.warning(
                "auth's organizations read answered nothing usable",
                extra={"event": "org_icons_read_unknown", "status_code": reply.status_code},
            )
        return OrgIcons()
    _cache.put(key, value)
    return value
