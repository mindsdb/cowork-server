"""Whether the caller's organization has Zero Data Retention (ZDR) on.

Read from ``zdr_mode`` on the caller's entitlements, through the same cached
read as their permissions (cowork.services.mindshub_teams).
"""

from __future__ import annotations

from cowork.services.mindshub_teams import fetch_mindshub_permissions


async def zdr_on(*, bearer_token: str, org_id: str, user_id: str) -> bool:
    """True when ZDR is on for the caller's organization, or can't be read:
    under ZDR MindsHub refuses non-ZDR models, so unknown counts as on."""
    if not bearer_token:
        return True
    permissions = await fetch_mindshub_permissions(bearer_token=bearer_token, org_id=org_id, user_id=user_id)
    return permissions.zdr_on


async def zdr_on_for(*, scope, settings, org_bearer: str = "", turn_key: str = "") -> bool:
    """``zdr_on`` asked as the caller: their bearer on the web, else the turn's key or the stored MindsHub key."""
    from cowork.common.settings.user_settings import Provider, provider_api_key_str

    if scope.org_mode and org_bearer:
        bearer = org_bearer
    else:
        bearer = turn_key or provider_api_key_str(settings, Provider.MINDS_CLOUD)
    return await zdr_on(bearer_token=bearer or "", org_id=scope.org_id or "", user_id=scope.user_id or "")
