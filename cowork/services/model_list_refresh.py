"""The saved copy of the MindsHub model list, and refreshing it before unattended turns.

Model resolution reads the org's saved list (``minds_model_enabled``,
``minds_model_zdr_blocked``, ...) without a network call. Only the client's
``recommended-models`` load refreshes it, so a channel message or a scheduled
run, which nobody watches, refreshes it here first when it is older than
``SAVED_MODEL_LIST_MAX_AGE_S``: otherwise a change such as an admin turning
zero data retention on leaves those turns picking a model MindsHub refuses.

Refreshed as the turn's identity: on the web with a short-lived minted key
(the org keeps no stored one), on the desktop with the stored key. A failed
refresh is logged and the turn goes on with the saved copy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from uuid import uuid4

from cowork.db.scoped import ScopedSession, TenantScope, unsafe_unscoped_session
from cowork.services.providers import (
    MindsModelListing,
    fetch_minds_models,
    persist_enabled_model_map,
    persist_image_model_map,
    persist_org_model_availability,
    persist_role_defaults_map,
    persist_zdr_blocked,
)

logger = logging.getLogger(__name__)

# How old the saved list may be before an unattended turn refreshes it.
SAVED_MODEL_LIST_MAX_AGE_S = 5 * 60
# The most a refresh may hold up the turn: the key mint, the listing and the writes.
_REFRESH_TIMEOUT_S = 15.0
# Long enough for the one listing call; revoked right after anyway.
_LISTING_KEY_TTL_S = 60

# (org_id, user_id) -> when this process last refreshed (or tried to refresh) that saved list.
_refreshed_at: dict[tuple[str | None, str | None], float] = {}


def _key(scope: TenantScope) -> tuple[str | None, str | None]:
    return (scope.org_id, scope.user_id)


def note_saved_model_list_refreshed(scope: TenantScope) -> None:
    """Record that the saved list for *scope* was just written from a live listing."""
    _refreshed_at[_key(scope)] = time.monotonic()


def persist_catalog_maps(
    session: ScopedSession, *, prior, listing: MindsModelListing, write_availability: bool
) -> None:
    """The catalog's writes, as one unit: the availability maps when the listing
    named any model, then the role defaults.

    Org mode splits the availability write. The listing is the caller's own
    (auth applies the model rules for their org, workspace and team), and the
    map is an org row every member's resolution reads, so the caller's
    restricted models are stored for the caller alone. A desktop install has
    one member and keeps them in the map.
    """
    raw = unsafe_unscoped_session(session)
    scope = session.scope
    if write_availability:
        if scope.org_mode:
            persist_org_model_availability(raw, scope, prior=prior, listing=listing)
        else:
            persist_enabled_model_map(
                raw, scope, prior.minds_model_enabled, listing.enabled, listing.ids
            )
            persist_image_model_map(raw, scope, prior.minds_image_model_enabled, listing)
        persist_zdr_blocked(raw, scope, prior.minds_model_zdr_blocked, listing)
    persist_role_defaults_map(raw, scope, prior.minds_role_defaults, listing.role_defaults)


async def refresh_saved_model_list_if_stale(scope: TenantScope) -> None:
    """Refresh *scope*'s saved model list when older than SAVED_MODEL_LIST_MAX_AGE_S. Never raises.

    Stamped before the refresh, so a busy channel's next messages, and a
    MindsHub that keeps failing, wait out the window rather than ask again."""
    key = _key(scope)
    now = time.monotonic()
    last = _refreshed_at.get(key)
    if last is not None and now - last < SAVED_MODEL_LIST_MAX_AGE_S:
        return
    _refreshed_at[key] = now
    try:
        async with asyncio.timeout(_REFRESH_TIMEOUT_S):
            await _refresh(scope)
    except Exception as exc:
        # Message only: the frames hold a MindsHub key.
        logger.warning("Could not refresh the saved model list; using the saved copy: %s", type(exc).__name__)


async def _refresh(scope: TenantScope) -> None:
    from cowork.common.settings.user_settings import Provider
    from cowork.db.units import run_db
    from cowork.services.settings import unit_setting_service

    if scope.org_mode and not scope.org_id:
        return
    prior = await run_db(lambda session: unit_setting_service(session).load(), scope=scope)
    if scope.org_mode:
        roles = (prior.resolved_planning_provider, prior.resolved_coding_provider, prior.resolved_router_provider)
        if Provider.MINDS_CLOUD not in roles:
            return
        listing = await _org_listing(scope, workspace_id=prior.hub_workspace_id or None)
    elif prior.minds_api_key is not None and prior.minds_url:
        listing = await fetch_minds_models(
            prior.minds_url, prior.minds_api_key.get_secret_value(), force_refresh=True, mindshub=True,
        )
    else:
        return
    if listing.ids is None and not listing.enabled:
        # Nothing came back: keep the saved copy.
        return
    await run_db(
        partial(persist_catalog_maps, prior=prior, listing=listing, write_availability=bool(listing.ids)),
        scope=scope,
    )


async def _org_listing(scope: TenantScope, *, workspace_id: str | None) -> MindsModelListing:
    """The org catalog as the turn's identity, on a key minted for this one call and revoked after."""
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.services.providers import _org_catalog_url
    from cowork.turnqueue.producer import mint_llm_block, revoke_turn_key_quietly

    correlation_id = f"model-list-{uuid4()}"
    try:
        block = await mint_llm_block(
            org_id=scope.org_id, user_id=scope.user_id, correlation_id=correlation_id,
            settings=TurnQueueSettings(), workspace_id=workspace_id, ttl_seconds=_LISTING_KEY_TTL_S,
        )
        return await fetch_minds_models(
            _org_catalog_url(), block["api_key"], force_refresh=True, mindshub=True,
            tenant_key=scope.org_id, user_id=scope.user_id,
        )
    finally:
        try:
            async with asyncio.timeout(2):
                await revoke_turn_key_quietly(correlation_id)
        except Exception:
            pass


def reset_for_tests() -> None:
    _refreshed_at.clear()
