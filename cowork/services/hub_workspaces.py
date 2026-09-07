"""MindsHub workspaces, read on behalf of the caller.

A MindsHub Workspace is an org-internal container that owns hub resources (API
keys, artifacts, model entitlements). It lives in the auth service and has
nothing to do with the filesystem directories this repo calls workspaces.

**Why the sidecar makes this call instead of the renderer.** Auth's ingress
allows three console origins per environment and no Cowork host
(``nginx.ingress.kubernetes.io/cors-allow-origin`` in auth's ``values-<env>.yaml``),
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

import asyncio
import hashlib
import logging
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The gate name auth declares in `configs/statsig_gates.json` and evaluates with
# its server SDK. A CONTRACT with auth: a typo here reads exactly like a gate
# that is switched off, and nothing catches it, so it is written once.
GATE_AUTHORIZATION_UI = "authorization_ui"

# Where the evaluated gate sits in the entitlements payload. Absent means off,
# which is also what a version of auth that predates the field reports, so this
# ships dark and lights up when auth's half lands.
GATES_FIELD = "feature_gates"

# Same split-TTL shape as the model catalog: an ANSWER is cached long enough that
# opening the menu repeatedly costs one round trip, and a non-answer is cached
# briefly so a route that is not deployed yet does not add a round trip to every
# open. Answer, not verdict: a gate auth evaluated as off is a real answer and
# gets the long TTL. Both are short because this is a kill switch: a gate flipped
# off has to reach a running app quickly, and auth caches the entitlements read
# behind its own TTL as well, so these two add up.
_TTL_OK = 60.0
_TTL_FAIL = 15.0

# Hard TOTAL budget for one call. httpx.Timeout is per-operation, so it alone
# does not bound a redirect chain or a trickled body; the outer wait_for does.
# This sits on a menu-open path, so a degraded auth service must not hang the UI.
_TIMEOUT_S = 5.0


class HubWorkspace(BaseModel):
    """One workspace, exactly as auth's ``WorkspaceSerializer`` puts it on the wire.

    Snake_case here because this mirrors auth's payload; the response model in
    ``cowork.schemas.hub_workspaces`` is what re-cases it for the renderer.

    ``role`` is the caller's own strongest standing on this workspace, which auth
    derives per row. Unvalidated against a fixed set on purpose: auth owns the
    vocabulary (``manager`` / ``member`` today), and a value this repo does not
    recognise should render as itself rather than fail the whole listing.
    """

    id: str
    slug: str = ""
    display_name: str = ""
    is_default: bool = False
    archived_at: Optional[str] = None
    role: str = ""


class HubWorkspaceListing(BaseModel):
    """The result of one listing read.

    ``reachable`` is the difference between "this organization has one workspace"
    and "we could not ask". An empty list with ``reachable=False`` must not be
    rendered as a single-workspace organization, because that is how a caller
    with three workspaces silently loses two of them during an auth outage.
    """

    workspaces: list[HubWorkspace] = Field(default_factory=list)
    reachable: bool = False


# Keyed by (auth host, organization, user, CREDENTIAL DIGEST). Auth answers this
# listing per caller, not per organization: an owner or admin gets every
# workspace in the org and a member only the ones they hold a grant on, and
# `role` is that caller's own standing. Keyed on the organization alone, one
# admin's menu is served to every member of their org for the whole TTL, and the
# grant check on `PUT /active` reads the same entry, so a member can store a
# workspace they hold no grant on. This process serves every user of every
# organization at once (`COWORK_TENANCY_MODE=org`, two replicas), so the caller
# has to be in the key.
#
# **The bearer is in the key as well as the identity, and not for tidiness.**
# `user_id` comes from the gateway-set principal, and the VALUE is whatever the
# bearer in `X-MindsHub-Authorization` answered, so on their own the two can name
# different people. They always do on a desktop install: `scope_from_principal`
# returns `LOCAL_SCOPE` outside org mode, so `user_id` is None for every request
# and this key would collapse to one shared entry. Sign out, sign in as another
# MindsHub account, and the previous account's workspaces are served for the rest
# of the TTL, with `PUT /active` authorizing against them. Keying on what the
# answer was actually fetched with fixes org mode and the desktop shell in one
# rule, because a new session means a new token means a new entry.
#
# A digest rather than the token: these keys are held for the process lifetime
# and turn up in a repr or a heap dump, and nothing here needs to reverse it.
_CacheKey = tuple[str, str, str, str]

# (stamped, verdict, whether auth answered at all)
_gate_cache: dict[_CacheKey, tuple[float, bool, bool]] = {}
_listing_cache: dict[_CacheKey, tuple[float, HubWorkspaceListing]] = {}

# Entries are never read again once the caller goes away, and nothing overwrites
# them, so without a sweep both dicts grow for as long as the process lives. Per
# caller keying made that per person rather than per organization. Swept on write
# rather than on a timer: the write is already the only place that touches the
# dict, and an expired entry costs nothing until then. Derived from the TTLs
# rather than restated, so raising one cannot start evicting live entries.
_MAX_TTL_S = max(_TTL_OK, _TTL_FAIL)


def _cache_key(*, org_id: str, user_id: str, bearer_token: str) -> _CacheKey:
    """One entry per credential per caller per organization per deployment."""
    digest = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()[:32]
    return (_auth_v1(), org_id or "", user_id or "", digest)


def _sweep(cache: dict[_CacheKey, tuple[Any, ...]]) -> None:
    """Drop entries no TTL could still consider fresh.

    Indexes the timestamp rather than unpacking, because the two caches carry
    different tuple widths and this has to serve both.
    """
    cutoff = time.monotonic() - _MAX_TTL_S
    for key in [k for k, entry in cache.items() if entry[0] < cutoff]:
        del cache[key]


def _auth_v1() -> str:
    """The auth service's public ``/v1`` base for this deployment."""
    from cowork.common.settings.app_settings import default_minds_auth_host

    return f"{default_minds_auth_host().rstrip('/')}/v1"


async def _get_json(path: str, bearer_token: str) -> Optional[Any]:
    """GET one auth path with the caller's bearer. None on any failure.

    Never raises and never runs longer than ``_TIMEOUT_S``. Every failure mode
    collapses to None: a refusal, an outage, a route that does not exist yet, a
    body that is not JSON. Callers turn None into the fail-closed answer, so a
    surface this gates stays dark rather than half-rendered.
    """

    async def _fetch() -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_TIMEOUT_S),
            follow_redirects=True,
            max_redirects=2,
        ) as client:
            return await client.get(
                f"{_auth_v1()}{path}",
                headers={"Authorization": f"Bearer {bearer_token}"},
            )

    try:
        response = await asyncio.wait_for(_fetch(), _TIMEOUT_S)
    except Exception as exc:
        logger.debug("auth GET %s failed: %s", path, exc)
        return None
    if response.status_code >= 400:
        logger.debug("auth GET %s returned HTTP %s", path, response.status_code)
        return None
    try:
        return response.json()
    except ValueError:
        logger.debug("auth GET %s returned a non-JSON body", path)
        return None


async def authorization_ui_enabled(*, bearer_token: str, org_id: str, user_id: str) -> bool:
    """Whether the authorization surfaces are switched on for this caller.

    Auth evaluates the Statsig gate with its server SDK and reports the verdict
    in the entitlements payload, so there is one gate governing the console and
    Cowork rather than two that can disagree. Cowork holds no Statsig client and
    no SDK key of its own.

    False whenever the answer is not a definite yes: no bearer, auth unreachable,
    a version of auth with no gates field, or the gate off. Those are all states
    where nobody knows whether the surface is safe to show, and a dark feature
    that stays dark is the cheap outcome.

    Cached per caller rather than per organization, because auth evaluates the
    gate for whoever presents the bearer: `authorization_ui` declares
    ``idType: userID``, so a rule below 100% or a per-user override answers
    differently for two people in one organization.
    """
    from cowork.common.settings.app_settings import get_app_settings

    # An ON-only development override, the same shape as the console's
    # staff-limited session override: it can turn the surface on where no gate
    # rule targets you, and it can never turn one off, so it cannot be used to
    # escape the kill switch. Not the flag; the gate is the flag.
    if get_app_settings().hub_workspaces_force_on:
        return True
    if not bearer_token:
        return False

    cache_key = _cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _gate_cache.get(cache_key)
    if cached:
        stamped, value, answered = cached
        if (time.monotonic() - stamped) < (_TTL_OK if answered else _TTL_FAIL):
            return value

    payload = await _get_json("/entitlements/me/", bearer_token)
    gates = payload.get(GATES_FIELD) if isinstance(payload, dict) else None
    enabled = bool(gates.get(GATE_AUTHORIZATION_UI)) if isinstance(gates, dict) else False
    # The TTL follows whether AUTH ANSWERED, not what it said. "Auth evaluated
    # the gate and it is off" is exactly as authoritative as "it is on", and
    # picking the TTL from the verdict gave the negative one the 15s failure
    # budget. That is the state this feature ships in, so with per-caller keys it
    # meant re-asking auth four times more often than needed, per person rather
    # than per organization.
    answered = payload is not None
    _sweep(_gate_cache)
    _gate_cache[cache_key] = (time.monotonic(), enabled, answered)
    return enabled


async def fetch_hub_workspaces(*, bearer_token: str, org_id: str, user_id: str) -> HubWorkspaceListing:
    """The workspaces this caller may use in their active organization.

    Reads ``GET /v1/organizations/current/workspaces/``, which auth gates on the
    role catalog rather than the relationship engine, so it answers in an
    environment with no OpenFGA configured. Every route under
    ``/v1/workspaces/<id>/`` does not, and would answer 503 there.

    **The answer is per caller, which is why the cache is too.** auth's
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
        return HubWorkspaceListing()

    cache_key = _cache_key(org_id=org_id, user_id=user_id, bearer_token=bearer_token)
    cached = _listing_cache.get(cache_key)
    if cached:
        stamped, value = cached
        if (time.monotonic() - stamped) < (_TTL_OK if value.reachable else _TTL_FAIL):
            return value

    payload = await _get_json("/organizations/current/workspaces/", bearer_token)
    listing = _parse_listing(payload)
    _sweep(_listing_cache)
    _listing_cache[cache_key] = (time.monotonic(), listing)
    return listing


def _parse_listing(payload: Any) -> HubWorkspaceListing:
    """Turn auth's body into a listing, dropping rows this repo cannot read.

    Auth answers ``{"results": [...]}``; a bare list is accepted too so a
    pagination change upstream degrades to a full list rather than an empty menu.
    A row with no ``id`` is dropped because it cannot be picked, and one row
    failing validation drops that row rather than the listing: losing one
    workspace from the menu beats losing all of them.
    """
    if payload is None:
        return HubWorkspaceListing()
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return HubWorkspaceListing()
    workspaces: list[HubWorkspace] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            workspaces.append(HubWorkspace.model_validate(row))
        except Exception as exc:
            logger.debug("dropping an unreadable workspace row: %s", exc)
    return HubWorkspaceListing(workspaces=workspaces, reachable=True)


def resolve_active(
    workspaces: list[HubWorkspace], stored_id: str
) -> Optional[HubWorkspace]:
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


def selectable(
    workspaces: list[HubWorkspace], active_id: Optional[str]
) -> list[HubWorkspace]:
    """The rows worth offering as places to work.

    Archiving stamps a date and changes nothing else, so auth keeps returning an
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
    """Drop both caches so each test starts from a known state."""
    _gate_cache.clear()
    _listing_cache.clear()


# Public names for the auth transport, shared with ``hub_usage``. Same
# functions; the underscore versions stay for this module's own callers.
auth_v1 = _auth_v1
get_auth_json = _get_json
