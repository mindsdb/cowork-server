"""Whether the caller's organization has Zero Data Retention (ZDR) on.

Read from ``zdr_mode`` on the caller's entitlements, through the same cached
read as their permissions (cowork.services.mindshub_teams).
"""

from __future__ import annotations

import logging

from cowork.services.mindshub_teams import ZdrState, fetch_mindshub_permissions, recent_org_zdr_state

logger = logging.getLogger(__name__)

# How long an org's last ZDR answer here stands in for a push that read none of its own.
PUSH_ZDR_ANSWER_MAX_AGE_S = 30 * 60


async def zdr_state(*, bearer_token: str, org_id: str, user_id: str) -> ZdrState:
    """ZDR for the caller's organization; UNKNOWN without a credential or a readable answer."""
    if not bearer_token:
        return ZdrState.UNKNOWN
    permissions = await fetch_mindshub_permissions(bearer_token=bearer_token, org_id=org_id, user_id=user_id)
    return permissions.zdr_state


async def zdr_on(*, bearer_token: str, org_id: str, user_id: str) -> bool:
    """True when ZDR is on for the caller's organization, or can't be read:
    under ZDR MindsHub refuses non-ZDR models, so unknown counts as on."""
    return await zdr_state(bearer_token=bearer_token, org_id=org_id, user_id=user_id) is not ZdrState.OFF


async def zdr_state_for(*, scope, settings, org_bearer: str = "", turn_key: str = "") -> ZdrState:
    """``zdr_state`` asked as the caller: their bearer on the web, else the turn's key or the stored MindsHub key."""
    from cowork.common.settings.user_settings import Provider, provider_api_key_str

    if scope.org_mode and org_bearer:
        bearer = org_bearer
    else:
        bearer = turn_key or provider_api_key_str(settings, Provider.MINDS_CLOUD)
    return await zdr_state(bearer_token=bearer or "", org_id=scope.org_id or "", user_id=scope.user_id or "")


async def zdr_on_for(*, scope, settings, org_bearer: str = "", turn_key: str = "") -> bool:
    """``zdr_state_for`` as a yes/no; unknown counts as on."""
    state = await zdr_state_for(scope=scope, settings=settings, org_bearer=org_bearer, turn_key=turn_key)
    return state is not ZdrState.OFF


async def browser_zdr_state(*, scope, settings, org_bearer: str = "", turn_key: str = "") -> ZdrState:
    """``browser_zdr_on`` that tells a failed or unreadable answer (UNKNOWN) from ZDR being on."""
    from cowork.handlers.jev_route import browser_needs_zdr

    if not browser_needs_zdr(settings):
        return ZdrState.OFF
    try:
        return await zdr_state_for(scope=scope, settings=settings, org_bearer=org_bearer, turn_key=turn_key)
    except Exception:
        logger.warning("Could not read ZDR for the browser; keeping it off", exc_info=True)
        return ZdrState.UNKNOWN


async def browser_zdr_on(*, scope, settings, org_bearer: str = "", turn_key: str = "") -> bool:
    """``zdr_on_for`` for the browser, asked only when its steps would go to Jev on MindsHub.

    False without a read otherwise; an unreadable answer counts as on, which only keeps the browser off."""
    from cowork.handlers.jev_route import browser_needs_zdr

    if not browser_needs_zdr(settings):
        return False
    try:
        return await zdr_on_for(scope=scope, settings=settings, org_bearer=org_bearer, turn_key=turn_key)
    except Exception:
        logger.warning("Could not read ZDR for the browser; keeping it off", exc_info=True)
        return True


def push_zdr_on(org_id: str | None, zdr: ZdrState | None = None) -> bool:
    """Whether a push for *org_id* must leave user content out: *zdr* when the push read it,
    else the org's last answer read here (fallback); unknown or none counts as on."""
    if zdr is not None:
        return zdr is not ZdrState.OFF
    try:
        return recent_org_zdr_state(org_id or "", max_age_s=PUSH_ZDR_ANSWER_MAX_AGE_S) is not ZdrState.OFF
    except Exception:
        logger.warning("Could not read ZDR for a push; leaving its content out", exc_info=True)
        return True


async def zdr_state_quietly(*, bearer_token: str, org_id: str, user_id: str) -> ZdrState:
    """``zdr_state`` that never raises: a failed read is UNKNOWN."""
    try:
        return await zdr_state(bearer_token=bearer_token, org_id=org_id, user_id=user_id)
    except Exception:
        logger.warning("Could not read ZDR", exc_info=True)
        return ZdrState.UNKNOWN
