"""MindsHub workspaces, read on behalf of the caller.

A MindsHub Workspace is an org-internal container that owns MindsHub resources
(API keys, artifacts, model entitlements). It lives in MindsHub and has
nothing to do with the filesystem directories this repo calls workspaces.

**Why the sidecar makes this call instead of the renderer.** MindsHub's ingress
allows three console origins per environment and no Cowork host
(``nginx.ingress.kubernetes.io/cors-allow-origin`` in MindsHub's ``values-<env>.yaml``),
and a per-PR Cowork host cannot be added to a static list because its name
carries the PR number. The packaged Electron app would get away with it (it runs
with ``webSecurity: false``), the web SPA would not. Reading here works for both
shells, and the active-workspace pick has to be stored here anyway.

**The bearer is the caller's own, and the host is the operator's.** Same rule
``fetch_org_model_catalog`` follows: the stored provider key and ``minds_url``
are both tenant-settable, so using either would let an org admin point a
member's credential at a host they chose. ``default_minds_auth_host`` derives
from ENV, which the desktop propagates when it spawns this process.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from cowork.services.mindshub_client import CallerCacheKey, CallerTtlCache, get_mindshub_json

logger = logging.getLogger(__name__)


class MindsHubWorkspace(BaseModel):
    """One workspace, exactly as MindsHub's ``WorkspaceSerializer`` puts it on the wire.

    Snake_case here because this mirrors MindsHub's payload; the response model in
    ``cowork.schemas.mindshub_workspaces`` is what re-cases it for the renderer.

    ``role`` is the caller's own strongest standing on this workspace, which MindsHub
    derives per row. Unvalidated against a fixed set on purpose: MindsHub owns the
    vocabulary (``manager`` / ``member`` today), and a value this repo does not
    recognize should render as itself rather than fail the whole listing.
    """

    id: str
    slug: str = ""
    display_name: str = ""
    is_default: bool = False
    archived_at: Optional[str] = None
    role: str = ""


class MindsHubWorkspaceListing(BaseModel):
    """The result of one listing read.

    ``reachable`` is the difference between "this organization has one workspace"
    and "we could not ask". An empty list with ``reachable=False`` must not be
    rendered as a single-workspace organization, because that is how a caller
    with three workspaces silently loses two of them during a MindsHub outage.
    """

    workspaces: list[MindsHubWorkspace] = Field(default_factory=list)
    reachable: bool = False


# Per caller (CallerCacheKey): MindsHub answers this listing per caller, and
# the grant check on `PUT /active` reads the same entry, so an entry shared
# across callers would let a member store a workspace they hold no grant on.
_listing_cache: CallerTtlCache[MindsHubWorkspaceListing] = CallerTtlCache()


async def fetch_mindshub_workspaces(*, bearer_token: str, org_id: str, user_id: str) -> MindsHubWorkspaceListing:
    """The workspaces this caller may use in their active organization.

    Reads ``GET /v1/organizations/current/workspaces/``, which MindsHub gates on its
    role catalog rather than the relationship engine, so it answers in an
    environment with no OpenFGA configured. Every route under
    ``/v1/workspaces/<id>/`` does not, and would answer 503 there.

    **The answer is per caller, which is why the cache is too.** MindsHub's
    ``WorkspaceService.visible_workspaces`` returns every workspace in the
    organization to an owner or admin and only granted ones to a member, and
    derives ``role`` for that caller. Two people in one organization get
    different listings, so ``user_id`` is required rather than defaulted: a
    caller that forgot to pass it would share an entry with everyone else in
    their org, and this listing is what the switch checks a grant against.

    Unreachable returns ``reachable=False`` with an empty list rather than
    raising: the caller renders nothing and keeps whatever it already had.
    """
    if not bearer_token:
        return MindsHubWorkspaceListing()

    cache_key = CallerCacheKey.for_caller(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _listing_cache.fresh(cache_key)
    if cached is not None:
        return cached

    payload = await get_mindshub_json("/organizations/current/workspaces/", bearer_token)
    listing = _parse_listing(payload)
    _listing_cache.put(cache_key, listing)
    return listing


def _parse_listing(payload: Any) -> MindsHubWorkspaceListing:
    """Turn MindsHub's body into a listing, dropping rows this repo cannot read.

    MindsHub answers ``{"results": [...]}``; a bare list is accepted too so a
    pagination change upstream degrades to a full list rather than an empty menu.
    A row with no ``id`` is dropped because it cannot be picked, and one row
    failing validation drops that row rather than the listing: losing one
    workspace from the menu beats losing all of them.
    """
    if payload is None:
        return MindsHubWorkspaceListing()
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return MindsHubWorkspaceListing()
    workspaces: list[MindsHubWorkspace] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            workspaces.append(MindsHubWorkspace.model_validate(row))
        except Exception as exc:
            logger.debug("dropping an unreadable workspace row: %s", exc)
    return MindsHubWorkspaceListing(workspaces=workspaces, reachable=True)


def resolve_active(
    workspaces: list[MindsHubWorkspace], stored_id: str
) -> Optional[MindsHubWorkspace]:
    """Which workspace is active: the stored pick, else the default, else the first.

    The same three-step fallback the console applies, so the two clients agree on
    which row carries the check. A stored id can easily name nothing live: the
    workspace was archived away, the grant was revoked, or the person moved to
    another organization. Falling through beats leaving no row checked at all.
    """
    if stored_id:
        for workspace in workspaces:
            if workspace.id == stored_id:
                return workspace
    for workspace in workspaces:
        if workspace.is_default:
            return workspace
    return workspaces[0] if workspaces else None


def forget_stale_mindshub_workspace(*, org_id: str, user_id: str, workspace_id: str) -> None:
    """Clear a stored pick MindsHub just refused to mint against.

    The menu already stops showing it (``resolve_active`` falls through to the
    default), but nothing rewrites the stored value, so every later turn would
    send it again and be refused again. Cleared only while it still names the
    refused workspace: the person may have picked a live one since this turn
    read it, and that newer choice must survive. Best effort: the turn has
    already recovered on the default, and the next refusal tries this again.
    """
    from cowork.db.scoped import scope_for_identity
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    # scope_for_identity picks the org, personal or local scope from the
    # tenancy settings.
    scope = scope_for_identity(org_id=org_id, user_id=user_id)
    session = None
    try:
        session = get_open_session()
        settings = SettingService(session, scope)
        if settings.load().hub_workspace_id == workspace_id:
            settings.upsert_setting("hub_workspace_id", "")
    except Exception:
        logger.warning(
            "could not clear stale hub workspace pick for org %s user %s",
            org_id, user_id, exc_info=True,
        )
    finally:
        if session is not None:
            session.close()


def selectable(
    workspaces: list[MindsHubWorkspace], active_id: Optional[str]
) -> list[MindsHubWorkspace]:
    """The rows worth offering as places to work.

    Archiving stamps a date and changes nothing else, so MindsHub keeps returning an
    archived workspace in every listing and the filtering is the client's to do.
    The active one stays on the list either way, so the row under the check can
    never vanish from beneath it. Matches the console's rule.
    """
    return [
        workspace
        for workspace in workspaces
        if not workspace.archived_at or workspace.id == active_id
    ]


def reset_caches_for_tests() -> None:
    """Drop the cache so each test starts from a known state."""
    _listing_cache.clear()
