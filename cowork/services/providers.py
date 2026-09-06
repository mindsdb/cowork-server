"""Provider service — testing, validation, and config-readiness checks."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, NamedTuple, Optional
from urllib.parse import urlparse

import httpx

from cowork.common.settings import runtime_credential
from cowork.common.settings.app_settings import AGENT_ROLE_NAMES, default_minds_api_host

if TYPE_CHECKING:
    from cowork.common.settings.user_settings import UserSettings

logger = logging.getLogger(__name__)

# Model used to probe MindsHub connectivity/auth (health test + onboarding
# validation). MUST be universally callable: MindsHub bills per model (a model
# the org's wallet can't pay for is denied), so probing a paid model would fail
# for an out-of-credits account even though the key is valid. mindshub_air is
# the free included model (drawn from the monthly allowance), so it resolves
# without depending on wallet balance — the probe then reflects reachability +
# key validity only, not billing availability. Probing a paid model here caused
# ENG-576 ("MindsHub failed its last test" / "Invalid API key" false-negatives).
MINDS_PROBE_MODEL = "mindshub_air"


def minds_chat_base_url(minds_url: str) -> str:
    """Derive the OpenAI-compatible chat base URL from a raw minds_url.

    mdb.ai needs /api/v1, api.mindshub.ai needs /v1.  If the URL
    already ends with /v1, return it as-is.
    """
    base = minds_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/api/v1" if "mdb.ai" in base else f"{base}/v1"


def is_minds_host(url: str | None) -> bool:
    """True when `url` points at a MindsHub inference host.

    Used to decide whether a probe that omitted its model should fall back to
    MINDS_PROBE_MODEL rather than to a generic OpenAI default that MindsHub does
    not serve.

    Compares the parsed hostname, never a substring of the whole URL: a
    substring test matches `mindshub.ai.example.test` and a `?next=` parameter
    carrying our host, and the base URL here is partly user-supplied (the
    openai-compatible card's own field). Covers the prod host, the per-env
    `api.<env>.mindshub.ai` / `api-<ns>.dev.mindshub.ai` hosts, and the legacy
    `mdb.ai` host.

    A self-hosted gateway on some other hostname does not match, the same
    limitation `build_llm_client` documents where it states the MindsHub flavor
    outright instead of deriving it from the URL. The consequence here is only
    that such a caller keeps today's generic default.

    Matching the `mdb.ai` host chooses the model, not the path. The
    openai-compatible probe appends `/v1` generically and never applies
    minds_chat_base_url's `mdb.ai` -> `/api/v1` rule, so a bare `mdb.ai` base
    with no version segment is still probed at a path that host does not serve.
    """
    raw = (url or "").strip()
    if not raw:
        return False
    # A bare `api.mindshub.ai/v1` has no scheme, so urlparse would read the whole
    # thing as a path and find no hostname. Prefixing `//` makes it a netloc.
    #
    # The try is load-bearing, not defensive: `.hostname` raises ValueError on an
    # unbalanced `[` or `]`, this runs in validate_provider OUTSIDE
    # validate_openai_compatible's except, and base_url is free text off the
    # provider card. Without it, `baseUrl: "["` leaves the service layer and the
    # endpoint answers 500 where it used to answer ok:false. The TypeScript copy
    # guards the same case with try/catch around `new URL`.
    try:
        parsed = urlparse(raw if "//" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    return host in ("mindshub.ai", "mdb.ai") or host.endswith((".mindshub.ai", ".mdb.ai"))


# Working prod publish host. Prod's api host (api.mindshub.ai) does NOT serve the
# publish API — it lives on the legacy 4nton.ai host — so prod, plus anything we
# can't map to a non-prod MindsHub env, falls back here.
PUBLISH_FAILSAFE_URL = "https://4nton.ai"


def publish_url_for_endpoint(endpoint_url: str | None) -> str:
    """Publish base URL for the MindsHub env the given endpoint points at.

    The publish API (root-mounted ``/upload``, ``/list``, ``/delete/{id}``) is
    served on the *non-prod* MindsHub api hosts, so a provider pointed at
    ``api.<env>.mindshub.ai`` (dev/staging) publishes to that same host. Prod
    (``api.mindshub.ai``) has no publish routes, and anything unrecognised
    (mdb.ai, a custom endpoint, empty) falls back to the legacy ``4nton.ai``
    host. anton appends the route path; an explicit `publish_url` /
    `ANTON_PUBLISH_URL` overrides this.
    """
    host = (urlparse(endpoint_url or "").hostname or "").lower()
    if host.startswith("api.") and host.endswith(".mindshub.ai") and host != "api.mindshub.ai":
        return f"https://{host}"
    return PUBLISH_FAILSAFE_URL
# Gemini speaks OpenAI-compatible at Google's endpoint — NOT api.openai.com.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


async def _current_runtime_minds_credential() -> str:
    """Return the desktop's current MindsHub credential for one LLM request.

    Providers built from a handed-over runtime credential must not fall back to
    the seed captured when the chat session was created. That seed can be an
    expired access token after the desktop refreshes or signs out.
    """
    credential = runtime_credential.get_minds_credential()
    if credential is None:
        raise _provider_auth_error(
            "The MindsHub session credential is no longer available."
        )
    return credential


def _provider_auth_error(message: str) -> ConnectionError:
    """Anton's typed 401, or the bare ConnectionError an older anton maps to.

    Imported lazily so a version-skewed anton loses the typed discriminator
    rather than failing this module's import and taking the whole server's boot
    with it. `handlers/turn_errors.is_auth_error` reads both shapes.
    """
    try:
        from anton.core.llm.provider import ProviderAuthError

        return ProviderAuthError(message)
    except Exception:
        return ConnectionError(f"Invalid API key — {message}")


def provider_base_url(
    provider: str, *, openai_base_url: str = "", minds_url: str = ""
) -> str | None:
    """Single source of truth for a provider's inference base URL.

    Returns the base URL the OpenAIProvider should use, or ``None`` to let the
    SDK use its built-in default (direct Anthropic/OpenAI hosts).

    The crux: ``openai_base_url`` is a *shared* DB slot reused by the openai,
    gemini, and openai-compatible cards. Only **openai-compatible** legitimately
    needs a user-supplied base. If openai or gemini were allowed to read that
    shared slot, a value left behind by a prior provider setup (e.g. MindsHub
    writing ``https://api.mindshub.ai/v1``) would silently misroute the next
    provider's request — and its API key — to the wrong vendor. So each
    provider's base is derived deterministically here and never inherited:

      - anthropic / openai → ``None`` (SDK default host; never the shared slot)
      - gemini             → Google's OpenAI-compatible endpoint
      - minds-cloud        → derived from the dedicated ``minds_url`` slot
      - openai-compatible  → the shared slot (this is the one that owns it)

    openai-compatible with an *empty* base returns None, NOT api.openai.com:
    forcing a BYO endpoint's key onto OpenAI's host would leak that key to the
    wrong vendor. An empty openai-compatible base is a misconfiguration that
    config_status surfaces ("Set a base URL") rather than silently routing.
    """
    p = (provider or "").replace("_", "-")
    if p == "minds-cloud":
        return minds_chat_base_url(minds_url)
    if p == "gemini":
        return GEMINI_BASE_URL
    if p == "openai-compatible":
        return openai_base_url or None
    # anthropic, openai → SDK default; never inherit the shared openai_base_url.
    return None


# ── Live model listing ───────────────────────────────────────────────

# MindsHub exposes an OpenAI-compatible `/v1/models` route. We surface
# that list in the Settings model picker so cowork tracks whatever the
# router currently supports instead of a hand-maintained constant —
# app_settings.RECOMMENDED_MODELS["minds-cloud"] is intentionally empty,
# so this live list is the only source of minds-cloud model names. The
# deprecated MindsHub sentinel aliases are hidden from this route by design.
#
# Cached so a rapid sequence of Settings opens doesn't re-hit the
# network. Failures are cached too — with a shorter TTL — so a route
# that isn't deployed yet doesn't add a round-trip to every load.
_MINDS_MODELS_TTL = 300.0       # successful fetch
_MINDS_MODELS_FAIL_TTL = 30.0   # negative result (down / not deployed)
# Hard TOTAL budget for one /v1/models fetch. httpx.Timeout is per-operation, so
# `follow_redirects` chains and trickled responses (each chunk under the per-op
# read timeout) can otherwise run far past it — minutes, unbounded. This fetch
# sits on the desktop boot path before the socket binds, so it must have a real
# ceiling; the outer asyncio.wait_for below enforces it.
_MINDS_MODELS_TIMEOUT_S = 6.0


class MindsModelListing(NamedTuple):
    """What one `/v1/models` fetch yields, keyed by model id where it's a map.

    A named tuple rather than a plain tuple because the picker needs six pieces
    of this, and positional unpacking of five same-typed dicts is a silent
    field-order bug waiting to happen in a stubbed test.

    ``ids`` is None on any failure, which is how the caller knows to keep the
    list it already has instead of emptying the picker. The maps are keyed by
    model id and are empty (never None) whenever the gateway doesn't publish that
    field — a plain OpenAI-compatible endpoint publishes none of
    ``labels``/``providers``/``families``.
    """

    ids: Optional[list[str]]
    efforts: dict[str, dict]
    enabled: dict[str, bool]
    # MindsHub's human-readable `label` ("Claude Sonnet 5"). Display-only.
    labels: dict[str, str]
    # The provider serving the model ("anthropic", "fireworks", …), so the picker
    # can group the catalog into sections. Who serves it, not who trained it: one
    # serving provider fronts several vendors' models.
    providers: dict[str, str]
    # The moving alias each model belongs to, defaulted to the id itself.
    # ``families[id] == id`` means the version behind this alias moves, which is
    # what the picker tags "latest"; anything else names the moving alias this
    # row is a frozen version of.
    families: dict[str, str]
    # Agent role -> the model id the catalog declares as that role's default,
    # inverted from the per-row ``default_for`` list. Keyed by role rather than by
    # model id, unlike every other map here, because that is the question asked of
    # it: resolution wants "what starts the planning role", not "which roles does
    # this model lead". Empty whenever the gateway publishes no defaults, which
    # includes every gateway that predates the field and every plain
    # OpenAI-compatible endpoint.
    #
    # Required rather than defaulted, like every other field here. A NamedTuple
    # default is one object shared by every instance that omits it, which is the
    # hazard ``_empty_listing`` is a factory to avoid; a default would reintroduce
    # it for the sake of not touching three test fakes.
    role_defaults: dict[str, str]


def _empty_listing() -> MindsModelListing:
    """The "we got nothing" listing: ``ids`` None, every map empty.

    One place to build it, so the failure paths don't each repeat the literals
    that have to agree. A factory rather than a shared constant because a
    failure is also cached, per base URL: handing every entry the same dict
    objects means one in-place write downstream would corrupt the cached failure
    of every gateway at once. Nothing mutates them today.
    """
    return MindsModelListing(None, {}, {}, {}, {}, {}, {})


# Keyed by (base_url, tenant): tenant is the org id for an org-scoped catalog
# (the `enabled` map is wallet-specific) and None for a single-user/BYOK fetch.
_minds_models_cache: dict[tuple[str, str | None], tuple[float, MindsModelListing]] = {}


# Substrings that identify an embeddings model by id, used only for endpoints
# that don't flag one. OpenAI's /v1/models has no field for it, so a BYO
# openai-compatible endpoint (or OpenAI itself) lists `text-embedding-3-small`
# indistinguishably from a chat model. Kept to the "embedding"/"embed" word
# rather than family names (bge, gte, e5) so a chat model is never dropped by a
# coincidence; a family-named embeddings model just stays in the list, which is
# the pre-existing behavior.
_EMBEDDING_ID_HINTS = ("text-embedding", "embedding-", "-embedding", "-embed-", "embed-")


def _is_embedding_row(row: dict, model_id: str) -> bool:
    """Whether a /v1/models row is an embeddings model rather than a chat one.

    MindsHub marks these explicitly with `"embedding": true`, and that flag is
    authoritative in both directions — an endpoint that says false is believed.
    Only when the field is absent entirely does the id get inspected.
    """
    if "embedding" in row:
        return bool(row.get("embedding"))
    lowered = model_id.lower()
    return any(hint in lowered for hint in _EMBEDDING_ID_HINTS)


def cached_minds_models(minds_url: str, tenant_key: str | None = None) -> MindsModelListing | None:
    """The last successful ``/v1/models`` listing for this gateway, however old.

    A synchronous read for callers that cannot await the fetch (the coding
    endpoints run in a threadpool) and need only what the gateway advertises per
    model, which changes on the gateway's release cadence rather than the cache
    TTL's. None until something has fetched the listing since the process
    started, and None for a negatively cached failure.
    """
    if not minds_url:
        return None
    cached = _minds_models_cache.get((minds_chat_base_url(minds_url), tenant_key))
    if not cached or not cached[1].ids:
        return None
    return cached[1]


async def fetch_minds_models(
    minds_url: str, api_key: str, *, force_refresh: bool = False, tenant_key: str | None = None
) -> MindsModelListing:
    """Fetch supported models from MindsHub's OpenAI-compatible `/v1/models`.

    Returns a :class:`MindsModelListing`. ``ids`` is the model-id list (or None
    on any failure so the caller falls back to the static list), ``efforts`` maps
    a model id to ``{"efforts": [...], "default": "..."}`` for every model that
    advertises ``reasoning_efforts``, ``enabled`` maps a model id to its
    ``enabled`` flag, ``labels`` maps a model id to MindsHub's human-readable
    ``label`` string, and ``providers``/``families`` carry the picker's grouping
    metadata: the provider serving the model, and which moving alias it belongs
    to. MindsHub marks a model the org's wallet can't currently pay for / whose
    free allowance is spent as ``"enabled": false`` so the picker can show it as
    locked with an "add credits" affordance; a model missing from ``enabled`` is
    treated as available. ``labels`` is purely a display aid for the picker — the
    model id remains the value used everywhere else (selection, storage,
    resolution); a model missing from ``labels`` falls back to the client's
    id-derived label. Embeddings models are dropped entirely — they share this
    listing but aren't chat/completion models (see ``_is_embedding_row``).

    ``force_refresh`` skips the cache *read* for a cached success (a fresh
    result is still cached for subsequent calls) — used when the caller knows
    the cached answer may be stale, e.g. the Settings picker re-checking on
    dropdown open after the user tops up credits in an external tab.

    A cached *failure* is still honored under ``force_refresh``: the negative
    TTL exists so an unreachable MindsHub isn't re-probed on every call, and
    the picker opens on demand, so bypassing it would make every open pay the
    fetch budget for as long as the outage lasts.

    Never raises and never runs longer than ``_MINDS_MODELS_TIMEOUT_S`` — a
    slow/degraded gateway (hang, redirect loop, trickled body) returns an empty
    listing that is negatively cached, so callers on the boot path can't hang.
    """
    if not minds_url or not api_key:
        return _empty_listing()
    base = minds_chat_base_url(minds_url)
    cache_key = (base, tenant_key)

    now = time.monotonic()
    cached = _minds_models_cache.get(cache_key)
    if cached:
        ts, val = cached
        ttl = _MINDS_MODELS_TTL if val.ids else _MINDS_MODELS_FAIL_TTL
        # force_refresh only overrides the success TTL — see the negative-cache
        # note above.
        if (now - ts) < ttl and not (force_refresh and val.ids):
            return val

    def _remember(val: MindsModelListing) -> MindsModelListing:
        _minds_models_cache[cache_key] = (time.monotonic(), val)
        return val

    async def _fetch() -> httpx.Response:
        # `max_redirects` low (not the httpx default of 20) so a same-origin
        # redirect loop can't multiply the per-op timeout into a long stall;
        # the endpoint is hit directly at `/models/` so no real redirect is
        # expected. The outer wait_for is the true ceiling.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_MINDS_MODELS_TIMEOUT_S),
            follow_redirects=True,
            max_redirects=2,
        ) as client:
            # Trailing slash is required: the MindsHub router serves the
            # listing at `/models/` and a recent minds-inference release
            # stopped cleanly redirecting the slashless `/models`, which
            # left this fetch empty and emptied the model picker. Hitting
            # `/models/` directly is what the other frameworks' shared
            # model-catalog helper already does.
            return await client.get(
                f"{base}/models/",
                headers={"Authorization": f"Bearer {api_key}"},
            )

    try:
        # Hard TOTAL budget: httpx.Timeout is per-operation, so it alone does
        # not bound a redirect chain or a trickled response. asyncio.wait_for
        # cancels the whole fetch at the ceiling and the TimeoutError falls
        # through to the negative-cache path below.
        r = await asyncio.wait_for(_fetch(), _MINDS_MODELS_TIMEOUT_S)
        if r.status_code >= 400:
            logger.debug("minds /models fetch returned HTTP %s", r.status_code)
            return _remember(_empty_listing())
        data = r.json()
    except Exception as exc:
        logger.debug("minds /models fetch failed: %s", exc)
        return _remember(_empty_listing())

    # OpenAI shape: {"object": "list", "data": [{"id": "...", ...}]}.
    # Accept a bare list too, defensively. Each row may carry the non-standard
    # extension fields `reasoning_efforts` (list), `default_reasoning_effort`
    # (str), `enabled` (bool), and `label` / `provider` / `family` (str) —
    # OpenAI clients ignore unknown keys; we surface them for the picker.
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return _remember(_empty_listing())
    ids: list[str] = []
    efforts: dict[str, dict] = {}
    enabled: dict[str, bool] = {}
    labels: dict[str, str] = {}
    providers: dict[str, str] = {}
    families: dict[str, str] = {}
    role_defaults: dict[str, str] = {}

    def _text(row: dict, key: str) -> Optional[str]:
        """A non-empty string field, or None. Anything else is treated as absent.

        The picker degrades cleanly on a missing field (no grouping) but not on a
        present-but-junk one, so a null / number / blank from an unexpected
        gateway is read as "didn't publish it" rather than becoming a section
        heading called "42".
        """
        value = row.get(key)
        if not isinstance(value, str):
            return None
        return value.strip() or None

    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        model_id = str(row.get("id")).strip()
        if not model_id:
            continue
        # Embedding models share this listing but aren't chat/completion
        # models — chosen for planning/coding roles they'd error every turn.
        # Filtered here, the single place every row is parsed, so neither the
        # picker nor default-resolution ever sees them.
        if _is_embedding_row(row, model_id):
            continue
        ids.append(model_id)
        # A model the org's wallet can't currently pay for (or whose free
        # allowance is spent) is listed with enabled=false so the picker can
        # show it as locked with an "add credits" affordance. Missing → available.
        if "enabled" in row:
            enabled[model_id] = bool(row.get("enabled"))
        levels = row.get("reasoning_efforts")
        if isinstance(levels, list) and levels:
            entry: dict = {"efforts": [str(x) for x in levels]}
            default = row.get("default_reasoning_effort")
            if default:
                entry["default"] = str(default)
            efforts[model_id] = entry
        # Display-only — the id (model_id) stays the value used for
        # selection/storage/resolution everywhere else.
        if label := _text(row, "label"):
            labels[model_id] = label
        family = _text(row, "family")
        if provider := _text(row, "provider"):
            providers[model_id] = provider
            # Defaulted to the id so the map is dense for every model this gateway
            # describes: "this alias moves" then reads as families[id] == id, with
            # no missing-key branch at the render site.
            families[model_id] = family or model_id
        elif family:
            # A family with no provider: unplaceable in a section, but it must
            # still be recorded. Leaving it out makes it ABSENT from the map, and
            # a consumer reading absent as "is its own head" would tag a frozen
            # version "latest" — the one claim it must never make. Recorded, the
            # app sees a pin whose head it cannot find and lists it plainly.
            families[model_id] = family
        # Which agent roles this model is the catalog's default for. Inverted
        # here, at the single place every row is parsed, so resolution can ask
        # "what starts the planning role" without walking the catalog.
        #
        # A role we do not serve is dropped rather than recorded. The catalog is
        # editable in a console, so an unknown role is a typo far more often than
        # a role a newer client understands, and carrying it would mean a stored
        # map whose keys nothing reads. The gateway-side gate rejects that typo
        # before it ships; this is what keeps a live one out of the settings row.
        # First declaration wins, which cannot arise against a validated catalog
        # and keeps the parse total when it does.
        for role in row.get("default_for") or ():
            if isinstance(role, str) and role in AGENT_ROLE_NAMES:
                role_defaults.setdefault(role, model_id)
    return _remember(
        MindsModelListing(
            (ids or None), efforts, enabled, labels, providers, families, role_defaults
        )
    )


async def fetch_org_model_catalog(
    *, org_id: str, bearer_token: str, refresh: bool = False
) -> MindsModelListing:
    """MindsHub catalog for an org, cached per org.

    Uses the OPERATOR endpoint (env/namespace-derived), never a tenant-settable
    URL, and the caller's own bearer — so a member's JWT can't be forwarded to an
    admin-chosen host. `org_id` is required; the `enabled` map is wallet-specific.
    """
    if not org_id:
        raise ValueError("org catalog requires an organization id")
    from cowork.common.settings.app_settings import (
        TurnQueueSettings,
        default_turn_minds_api_host,
    )
    url = TurnQueueSettings().minds_base_url or f"{default_turn_minds_api_host()}/v1"
    return await fetch_minds_models(url, bearer_token, force_refresh=refresh, tenant_key=org_id)


# ── Model-value validation on write (ENG-1358) ───────────────────────
#
# WRITER INVENTORY — every path that can put a model id into the settings DB.
# Recorded here because the ENG-597 lesson is that a check added to one writer is
# defeated by the writer nobody enumerated; if you add a writer, it belongs on
# this list and behind this gate.
#
#   GATED (both call `_reject_unservable_models` in api/v1/endpoints/settings.py):
#     1. PUT /api/v1/settings/{key}   → SettingService.upsert_setting
#     2. PUT /api/v1/settings/        → SettingService.save_all
#   Every product surface that writes a model does so OVER HTTP, so it travels
#   one of those two (see the scope limit on _reject_unservable_models — the
#   in-process service methods are not gated):
#     - the Settings model picker (cowork SettingsView.jsx) — values already come
#       from the live catalog, so it was never the leak. Note it saves provider +
#       credential + model in ONE bulk PUT, which is why the gate must resolve
#       against pending state;
#     - desktop onboarding's `syncModelsToDb` (cowork syncSettings.ts), which
#       replays `ANTON_*_MODEL` lines from .env VERBATIM — unvalidated until now,
#       and the most plausible origin of the ENG-1358 report;
#     - mindsdb/cowork-enterprise `scripts/docker-entrypoint.py`
#       (`sync_settings_when_healthy`) — the container's own Python writer, NOT
#       the Electron `syncSettings.ts`, which ships unused there. It PUTs both
#       models from `ANTON_*_MODEL` on first boot and swallows failures;
#     - cowork_evals `src/eval_service/provisioning/cowork_admin.py`
#       (`apply_runtime_settings`),
#       which raises on any >=400 — every shipped run spec pins a `latest:` alias,
#       hence the legacy-prefix allowance in model_value_rejection;
#     - cowork-kinaxis-preview's divergent inline `syncOnboardingModels`;
#     - anything hand-rolled (curl, scripts, a console surface).
#
#   GATED FOR SHAPE ONLY: `minds_role_defaults`, the cached role -> model id map
#     this endpoint's own writer fills from the catalog. `PUT /settings/{key}`
#     accepts it like any declared field, and its values become the model an
#     unset role starts on, so `_reject_malformed_role_defaults` 400s a body that
#     is not role -> non-empty id. Catalog membership is NOT checked: the value
#     comes from the catalog to begin with, and `_enabled_aware_default` discards
#     a model the availability map does not affirm, so an id that no longer
#     resolves costs a role its declared default rather than a turn.
#
#   NOT GATED, because they cannot carry a model key at all — the ENG-739
#   carve-out keeps planning/coding/router models out of `SETTING_ENV_ALIASES`,
#   so a bulk .env sync can never re-pin a picker choice:
#     3. POST /api/v1/settings/raw    → SettingService.bulk_upsert
#     4. cowork/migrations.py (first-boot .env→DB seed)
#     5. cowork/main/minds-auth.ts (login / token refresh) — excludes models
#   Two more write settings but only their own fixed key: channels.py
#   (`channels_harness`) and the `minds_model_enabled` refresh in settings.py.
#
#   OUT OF REACH of this gate: the standalone `anton` CLI reads models from
#   ~/.anton/.env directly and never touches this DB (ENG-1140 covers its
#   equivalent bug).

#: The settings whose VALUE names a model id. `UserSettings` types these as bare
#: `str | None`, so the settings API validates the key and the type and nothing
#: else — an id no provider can serve saves cleanly and only fails much later, at
#: turn time. Router model is included: same field shape, same failure.
MODEL_VALUE_SETTINGS = frozenset({"planning_model", "coding_model", "router_model"})


#: Deprecated-but-live alias namespace. MindsHub aliases are bare (``sonnet``),
#: but the gateway still RESOLVES a ``latest:`` prefix (app_settings.py:23), and
#: minds-auth.ts deliberately preserves such a pin as a user choice. `/v1/models`
#: is a listing, not the servable set, so a strict membership test would 400 an
#: id prod serves today — including every shipped cowork_evals run spec.
#: Stripping the prefix and re-testing keeps `latest:nonsense` rejected, which a
#: bare "starts with latest:" allowance would not.
_LEGACY_MODEL_PREFIX = "latest:"


async def model_value_rejection(
    settings: UserSettings,
    key: str,
    value: str,
    *,
    org_id: str | None = None,
    bearer_token: str = "",
) -> str | None:
    """Why ``value`` is not a servable model for ``key``, or None to allow it.

    ENG-1358: a model id that MindsHub cannot serve could be written into
    `planning_model` / `coding_model` and nothing caught it — not on write, not
    by the (deliberately model-blind) connection test, and at turn time only as
    a 404. This is the write-time check.

    ``settings`` must be the state the write PRODUCES (``load_pending``), not the
    stored one — see that method.

    In org (hosted) tenancy no MindsHub key is stored at all (a per-turn key is
    minted, ``user_settings._has_key``), so the catalog comes from the operator
    endpoint keyed by ``org_id`` + the caller's own bearer, exactly as
    ``recommended_models`` does. Without those the org path has no evidence and
    allows the write.

    **Soft-fail is the whole contract.** This returns None — allow the write —
    for every case except "we hold a real catalog and this id is definitively
    not in it":

    - not a model-valued setting, or an empty value (clearing the pin);
    - the target provider isn't MindsHub (a BYOK/custom endpoint has its own
      catalog, and Anthropic publishes no `/v1/models` at all);
    - no credentials to fetch a catalog with — onboarding writes a model before
      they exist, and blocking that would deadlock a fresh install;
    - the catalog fetch failed, timed out, or came back empty.

    An offline or degraded MindsHub must never stop someone changing settings,
    so every failure mode above resolves to "allow". The fetch is the cached one
    the picker already uses (TTL + negative TTL), so the common case costs no
    network call, and a MindsHub outage costs one timeout, once.

    Never put the API key — or the raw catalog — in the returned string: it is
    surfaced to the client as a 400 body.
    """
    if key not in MODEL_VALUE_SETTINGS or not value or not value.strip():
        return None

    # Which provider will actually serve this model. Only MindsHub has a catalog
    # we can check; anything else is the user's own endpoint.
    provider_attr = {
        "planning_model": "resolved_planning_provider",
        "coding_model": "resolved_coding_provider",
        "router_model": "resolved_router_provider",
    }[key]
    try:
        from cowork.common.settings.user_settings import Provider

        if getattr(settings, provider_attr, None) != Provider.MINDS_CLOUD:
            return None
        stored_key = settings.minds_api_key
        api_key = stored_key.get_secret_value() if stored_key is not None else ""
    except Exception:
        # Settings shapes vary across versions/scopes; a resolution failure is
        # not grounds to block a write.
        logger.debug("model validation: could not resolve provider for %s", key, exc_info=True)
        return None

    try:
        if org_id and bearer_token:
            listing = await fetch_org_model_catalog(
                org_id=org_id, bearer_token=bearer_token
            )
        elif api_key and settings.minds_url:
            listing = await fetch_minds_models(settings.minds_url, api_key)
        else:
            return None
    except Exception:
        # The fetchers swallow their own errors, but never let an unexpected one
        # turn into a failed settings save.
        logger.warning("model validation: catalog fetch failed for %s — allowing write", key)
        return None

    # ids is None on any failure and the list is empty for a gateway that serves
    # no chat models; in both cases we hold no evidence, so we allow.
    if not listing.ids:
        return None
    if value in listing.ids:
        return None
    if value.startswith(_LEGACY_MODEL_PREFIX) and (
        value[len(_LEGACY_MODEL_PREFIX):] in listing.ids
    ):
        return None

    # Definitive: a real catalog that does not contain this id.
    known = ", ".join(sorted(listing.ids)[:5])
    return (
        f"'{value}' is not a model this provider offers. "
        f"Pick one from the model list in Settings (for example: {known})."
    )


def persist_enabled_model_map(
    session, scope, prior_json: str | None, live_enabled: dict, live_ids: list[str] | None = None
) -> bool:
    """Guarded write of the `minds_model_enabled` availability map.

    Shared by every writer (the recommended-models endpoint and the startup /
    credential-sync warm below) so the invariants can't drift between them:

    - Only ever write with real evidence. A fetch failure yields neither a
      catalogue nor flags (``live_ids`` None and ``live_enabled`` ``{}``); with
      nothing to go on we hold the known-good map rather than clobber it with an
      empty one — silently re-locking the canonical default is the ENG-597 bug.
      But a real catalogue is evidence on its own: a gateway that returns ids
      WITHOUT ``enabled`` flags (version skew / a plain OpenAI-compatible
      endpoint) still tells us which ids are served, so we prune the ids it
      dropped — preserving the flags already stored for the survivors, since we
      can't re-derive which paid aliases are locked from a flag-less response.
      Otherwise a retired id (a ``mindshub_air`` MindsHub stopped serving)
      lingers as "still served" and resolution keeps selecting a model that
      404s.
    - Persist the FULL served catalogue, not just the flagged rows.
      ``fetch_minds_models`` only records rows that publish the optional
      ``enabled`` field, so ``live_enabled`` alone is sparse: a served model
      that omits the flag (``missing = available``) is absent from it. The
      resolution logic (``_enabled_aware_default`` / ``_resolved_model``) reads
      key ABSENCE from a non-empty stored map as "retired from the catalogue"
      and steers off it — so a sparse map would misread a working free model as
      retired and resolve a ``mindshub_air`` default/pin to a locked paid model.
      Once we actually have availability metadata (``live_enabled`` non-empty),
      fold every served id in with the flag it published, defaulting the
      unflagged ones to ``True``, so key absence means genuinely-not-served.
    - Persist ORDER-PRESERVING JSON — never ``sort_keys``. The first-enabled
      fallback (``_enabled_aware_default``) iterates the map in insertion order
      and returns the first *enabled* model, which must stay the gateway's own
      /v1/models ranking (a remote order we don't control or pin, and which
      changes per deployment — a paid alias often ranks ahead of the free
      ``mindshub_air``, which is then reached only because those aliases are
      marked disabled). Densifying over
      ``live_ids`` (already in that ranking) preserves it; alphabetizing would
      substitute our ordering for the gateway's and could promote a different
      enabled model.
    - Write only on a real change (the compare is order-sensitive too, so a
      gateway re-ranking also refreshes): ``upsert_setting`` commits a row and
      invalidates the settings cache, so an unconditional write churns every
      ``UserSettings`` reader.

    Returns True iff the stored map was updated.
    """
    from cowork.services.settings import SettingService

    try:
        prior = json.loads(prior_json or "{}")
        if not isinstance(prior, dict):
            prior = {}
    except (ValueError, TypeError):
        prior = {}

    if live_enabled:
        # Gateway published availability: densify over the served catalogue so
        # key ABSENCE means genuinely not-served — every served id folded in
        # with the flag it published, unflagged rows defaulting to available
        # (missing = available), in the gateway's own /v1/models order.
        live_enabled = {mid: live_enabled.get(mid, True) for mid in (live_ids or live_enabled)}
    elif live_ids:
        # A real catalogue that published NO enabled flags (gateway version
        # skew / a plain OpenAI-compatible endpoint). We can't re-derive which
        # paid aliases are locked, so PRESERVE the flags already stored for the
        # ids that survive — but still PRUNE the ids the catalogue dropped.
        # Otherwise a retired id (a ``mindshub_air`` MindsHub stopped serving)
        # lingers as "still served" and resolution keeps selecting a model that
        # 404s. A never-before-seen served id defaults to available.
        live_enabled = {mid: prior.get(mid, True) for mid in live_ids}
    else:
        # No flags AND no catalogue — no evidence at all (a fetch failure), so
        # hold the known-good map rather than clobber it with an empty one.
        return False

    desired = json.dumps(live_enabled)
    if desired == json.dumps(prior):
        return False
    SettingService(session, scope).upsert_setting("minds_model_enabled", desired)
    return True
def persist_role_defaults_map(session, scope, prior_json: str | None, live_role_defaults: dict) -> bool:
    """Guarded write of the `minds_role_defaults` map the catalogue declares.

    The sibling of ``persist_enabled_model_map``, and deliberately NOT the same
    function. That one carries rules this map has no use for: it densifies over
    the served catalogue and prunes retired ids, because resolution reads key
    ABSENCE from the availability map as "not served". This map is keyed by agent
    role, every key is a role we serve, and absence just means the catalogue
    declared no default for that role.

    Two rules it does share, for its own reasons:

    - Never clobber a known-good map with an empty one. A gateway that predates
      ``default_for`` publishes nothing, and writing ``{}`` would drop every role
      back to a constant only a client release can change. So an empty
      declaration leaves the stored map alone, and resolution keeps using it.
    - Write only on a real change: ``upsert_setting`` commits a row and
      invalidates the settings cache, so an unconditional write churns every
      ``UserSettings`` reader, and this endpoint is hit on every boot and every
      settings open.

    Stored SORTED, unlike the availability map. Nothing reads this map's order,
    so sorting means a gateway that re-ranks the same declarations does not count
    as a change and does not trigger a write.

    Returns True iff the stored map was updated.
    """
    from cowork.services.settings import SettingService

    if not live_role_defaults:
        return False
    desired = json.dumps(live_role_defaults, sort_keys=True)
    try:
        prior = json.loads(prior_json or "{}")
        if not isinstance(prior, dict):
            prior = {}
    except (ValueError, TypeError):
        prior = {}
    if desired == json.dumps(prior, sort_keys=True):
        return False
    SettingService(session, scope).upsert_setting("minds_role_defaults", desired)
    return True


async def warm_enabled_model_map(session, scope=None) -> bool:
    """Desktop: populate `minds_model_enabled` from /v1/models so the FIRST turn
    resolves an affordable model for a free-tier user with a stored paid pin
    (ENG-748).

    Since ENG-1652 the minds-cloud role defaults are the free model in both
    modes, so the UNSET default is already affordable (floored by
    ``role_defaults``). A stored PAID pin is the case left: it is steered off an
    unaffordable model only by the availability map, via the wallet-aware
    fallback in ``_resolved_model``. That map is refreshed lazily on GET
    /recommended-models, so a brand-new sign-in that sends before the picker ever
    loads resolves the pin against an EMPTY map — an empty map is no evidence, so
    the pin is kept and MindsHub denies the empty free-tier wallet
    (``wallet_empty`` 402) on message one. Warming the map at the two
    guaranteed-pre-first-turn seams (server startup with a stored key, and
    immediately after a credential sync) closes that race without touching the
    turn path.

    Fail-open: ``fetch_minds_models`` never raises and returns an empty listing
    on any error, bounded by a hard total budget (``_MINDS_MODELS_TIMEOUT_S``)
    so a degraded gateway can't stall the caller, and ``persist_enabled_model_map``
    never writes an empty map — so an unreachable MindsHub leaves the stored map
    untouched and is never worse than today. Returns True iff the map was
    updated.
    """
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services.settings import SettingService

    scope = scope if scope is not None else LOCAL_SCOPE
    s = SettingService(session, scope).load()
    if s.minds_api_key is None or not s.minds_url:
        return False
    listing = await fetch_minds_models(s.minds_url, s.minds_api_key.get_secret_value())
    return persist_enabled_model_map(
        session, scope, s.minds_model_enabled, listing.enabled, listing.ids
    )


# ── Config readiness ─────────────────────────────────────────────────


def check_config_status(settings: UserSettings) -> dict[str, Any]:
    """Derive configReady / configError from the loaded settings."""
    status = settings.config_status
    return {
        "configReady": status["config_ready"],
        "configError": status["config_error"] or "",
        "providerLabel": status["provider_label"],
    }


# ── Provider pinging ─────────────────────────────────────────────────


def _provider_error_message(resp: httpx.Response) -> Optional[str]:
    """Best-effort human message from a provider error response, or None.

    Handles OpenAI's object body (``{"error": {"message": ...}}``) AND Google's
    Gemini OpenAI-compat *chat* errors, which arrive as a single-element ARRAY
    (``[{"error": {"message": ...}}]``) — the shape that made ENG-1145 surface as
    a contentless "HTTP 404" everywhere the message was read as ``.error.message``
    on the top-level object. Also tolerates a bare ``{"message": ...}``."""
    try:
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        return msg.strip() if isinstance(msg, str) and msg.strip() else None
    if isinstance(err, str) and err.strip():
        return err.strip()
    msg = data.get("message")
    return msg.strip() if isinstance(msg, str) and msg.strip() else None


# Auth-SHAPED provider messages: only the genuinely "your key is bad" phrasings.
# A bare "api key" substring also appears in permission ("The API key does not
# have permission to use this model") and quota ("Quota exceeded for this API
# key") errors — neither of which a new key fixes — so matching that substring
# sends a user with a perfectly good key off to regenerate it. Must stay in step
# with cowork's src/main/provider-error.ts AUTH_SHAPED (ENG-1145 review).
_AUTH_SHAPED_RE = re.compile(
    r"api[_ ]?key (is )?(not valid|invalid)|invalid api[_ ]?key|pass a valid api[_ ]?key",
    re.IGNORECASE,
)


def _is_auth_error(status_code: int, msg: Optional[str]) -> bool:
    """Whether a provider error is a bad-key failure (fix the key), vs something
    the message should be surfaced for verbatim. 401/403 are auth by status;
    Gemini answers a bad key with 400, so fall back to an auth-shaped message —
    but only the tight forms above, letting permission/quota messages pass
    through (ENG-1145 review)."""
    if status_code in (401, 403):
        return True
    return bool(msg and _AUTH_SHAPED_RE.search(msg))


async def ping_provider(p: dict[str, Any]) -> tuple[str, str]:
    """Ping a single provider and return (status, detail)."""
    ptype = p.get("type")
    key = (p.get("apiKey") or "").strip()
    timeout = httpx.Timeout(12.0)

    async def _check(url: str, headers: dict[str, str]) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        if r.status_code < 400:
            return ("ok", f"HTTP {r.status_code}")
        # Include the provider's own message in the fail detail so the Settings
        # dot tooltip distinguishes a bad key from a real outage — Gemini returns
        # 400 "Please pass a valid API key", not 401 (ENG-1145).
        msg = _provider_error_message(r)
        return ("fail", f"HTTP {r.status_code}: {msg}" if msg else f"HTTP {r.status_code}")

    async def _chat_probe(url: str, headers: dict[str, str], model: str) -> tuple[str, str]:
        """Exercise the actual inference path with a tiny completion.

        This is the only route guaranteed to behave the same as a real
        task: `/models` and other listing endpoints are not deployed on
        every MindsHub host (they 404/401 even for valid keys), which
        produced false negatives even though chat completions worked.
        A 401/403 still means a rejected key; any other non-2xx is a
        genuine failure surfaced with its HTTP code.

        `max_tokens` is kept at a small-but-safe 20 rather than 1 — some
        models reject a 1-token budget (or can't emit even a stop token),
        which would fail the probe for a perfectly valid key."""
        payload = {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": "ping"}]}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.post(url, headers=headers, json=payload)
        if r.status_code < 400:
            return ("ok", f"HTTP {r.status_code}")
        # Same as _check: surface the gateway's own message so minds-cloud's
        # Settings dot carries the actionable reason (wallet/allowance/model)
        # instead of a bare "HTTP 502"/"HTTP 429". _chat_probe exercises real
        # chat completions, so its failures carry exactly that (ENG-1145 review,
        # ENG-576). Body read after close is safe — non-streaming POST.
        msg = _provider_error_message(r)
        return ("fail", f"HTTP {r.status_code}: {msg}" if msg else f"HTTP {r.status_code}")

    try:
        if ptype == "anthropic":
            if not key:
                return "fail", "missing API key"
            return await _check("https://api.anthropic.com/v1/models",
                                {"x-api-key": key, "anthropic-version": "2023-06-01"})
        if ptype == "openai":
            if not key:
                return "fail", "missing API key"
            return await _check("https://api.openai.com/v1/models",
                                {"Authorization": f"Bearer {key}"})
        if ptype == "gemini":
            if not key:
                return "fail", "missing API key"
            return await _check("https://generativelanguage.googleapis.com/v1beta/openai/models",
                                {"Authorization": f"Bearer {key}"})
        if ptype == "openai-compatible":
            base = (p.get("baseUrl") or "").rstrip("/")
            if not base:
                return "fail", "missing base URL"
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            return await _check(f"{base}/models", headers)
        if ptype == "minds-cloud":
            if not key:
                return "fail", "missing API key"
            base = (p.get("mindsUrl") or default_minds_api_host()).rstrip("/")
            chat_url = minds_chat_base_url(base)
            # Probe with a UNIVERSALLY-CALLABLE model, never the configured/
            # default one. This is a connectivity + auth check for the provider,
            # not a billing-availability check — and MindsHub bills per model (a
            # model the wallet can't pay for is denied). The old default was
            # CODING_MODEL_DEFAULTS["minds_cloud"] = "haiku" (paid), so every
            # out-of-credits account saw "MindsHub failed its last test" even
            # though chat worked on mindshub_air (ENG-576). mindshub_air is the
            # free included model (drawn from the monthly allowance), so it
            # resolves without depending on wallet balance — the dot then
            # reflects reachability/key validity only. (Testing the user's role
            # model was also rejected in the ENG-577 review for adding
            # false-negatives + token cost.)
            return await _chat_probe(
                f"{chat_url}/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                MINDS_PROBE_MODEL,
            )
    except httpx.HTTPError as e:
        return "fail", f"{type(e).__name__}: {e}"
    except Exception as e:
        logger.warning("Provider %s ping error: %s", ptype, e)
        return "fail", f"{type(e).__name__}: {e}"
    return "fail", "unknown provider type"


async def ping_providers(providers: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Ping multiple providers in parallel. Returns (statuses, details) dicts keyed by type."""
    results = await asyncio.gather(*[ping_provider(p) for p in providers], return_exceptions=True)
    statuses: dict[str, str] = {}
    details: dict[str, str] = {}
    for p, r in zip(providers, results):
        if isinstance(r, Exception):
            statuses[p["type"]] = "fail"
            details[p["type"]] = f"{type(r).__name__}: {r}"
        else:
            statuses[p["type"]], details[p["type"]] = r
    return statuses, details


# ── Provider credential validation ───────────────────────────────────


async def validate_anthropic(api_key: str, model: str = "claude-sonnet-4-6") -> dict[str, Any]:
    try:
        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": "ping"}]},
            )
            if r.status_code in (200, 201):
                return {"ok": True}
            return {"ok": False, "error": _provider_error_message(r) or f"HTTP {r.status_code}"}
    except Exception:
        return {"ok": False, "error": "Cannot connect"}


async def validate_minds(api_key: str, base_url: str = "") -> dict[str, Any]:
    base_url = base_url or default_minds_api_host()
    # Probe the real inference path rather than `/models`: listing routes
    # are not deployed on every MindsHub host and 404/401 even for valid
    # keys, which blocked onboarding with a working key. A 1-token chat
    # completion is the same surface a real task exercises.
    try:
        chat_base = minds_chat_base_url(base_url.rstrip("/"))
        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.post(
                f"{chat_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": MINDS_PROBE_MODEL, "max_tokens": 20,
                      "messages": [{"role": "user", "content": "ping"}]},
            )
        if r.status_code in (401, 403):
            return {"ok": False, "error": "Invalid API key"}
        if 200 <= r.status_code < 300:
            return {"ok": True}
        return {"ok": False, "error": _provider_error_message(r) or f"HTTP {r.status_code}"}
    except Exception:
        return {"ok": False, "error": "Cannot connect"}


async def validate_openai_compatible(api_key: str, base_url: str = "https://api.openai.com/v1",
                                     model: str | None = None,
                                     max_tokens: int | None = None) -> dict[str, Any]:
    """Probe an openai-compatible chat endpoint.

    `max_tokens` is opt-in, and deliberately not sent by default. It belongs on a
    probe we know the shape of: MindsHub bills per model, so an uncapped probe
    draws a full-length completion from the included allowance every time
    onboarding runs. It does not belong on an arbitrary endpoint, because
    `max_tokens` is not universal — OpenAI's reasoning models reject it and want
    `max_completion_tokens`, and `o3`/`o4-mini` are both in RECOMMENDED_MODELS, so
    sending it unconditionally would report a working key as invalid for exactly
    the models this ticket is about. `validate_provider` passes it on the MindsHub
    fallback only.

    `follow_redirects` matches validate_minds and _chat_probe. Safe on a
    user-supplied base URL because httpx strips `Authorization` on a redirect that
    leaves the origin (`_redirect_headers`), so the key cannot follow a bounce to
    another host.
    """
    try:
        normalized = base_url.rstrip("/")
        chat_url = f"{normalized}/chat/completions" if re.search(r"/v\d", normalized) else f"{normalized}/v1/chat/completions"
        timeout = httpx.Timeout(15.0)
        payload: dict[str, Any] = {"model": model or "gpt-5.5",
                                   "messages": [{"role": "user", "content": "ping"}]}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.post(
                chat_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            if r.status_code in (200, 201):
                return {"ok": True}
            msg = _provider_error_message(r)
            # Gemini (and some others) return 400 — not 401/403 — for a bad key,
            # so key on an auth-shaped message too, or a bad key reads as an
            # opaque "HTTP 400" the user can't act on. Tight match: a permission
            # or quota message also contains "api key" but is not a bad key
            # (ENG-1145 review).
            if _is_auth_error(r.status_code, msg):
                return {"ok": False, "error": "Invalid API key"}
            return {"ok": False, "error": msg or f"HTTP {r.status_code}"}
    except Exception:
        return {"ok": False, "error": "Cannot connect"}


async def validate_provider(provider: str, api_key: str,
                            base_url: str | None = None,
                            model: str | None = None) -> dict[str, Any]:
    """Validate credentials for a given provider type."""
    if provider == "anthropic":
        return await validate_anthropic(api_key, model or "claude-sonnet-4-6")
    if provider == "minds":
        return await validate_minds(api_key, base_url or default_minds_api_host())
    if provider == "openai-compatible":
        resolved_base = base_url or "https://api.openai.com/v1"
        # A caller that omits the model against a MindsHub host gets the free
        # probe model. Neither alternative works there: validate_openai_compatible's
        # generic default is not a MindsHub alias so it 404s, and the recommended
        # MindsHub model is paid so a wallet with no balance 402s. Both read back
        # as an invalid key and send a new user to bring-your-own-key with a
        # working key already saved. The connectivity probe above was fixed for
        # this same reason; the validation path never got the same treatment.
        #
        # Only when the model is omitted. An explicit model is always validated
        # as asked, or picking one model in the provider card would silently
        # validate a different one and report a pass the user cannot trust.
        if not model and is_minds_host(resolved_base):
            model = MINDS_PROBE_MODEL
            # Cap it the way validate_minds and _chat_probe do: this probe is
            # MindsHub-bound and otherwise draws a full-length completion from the
            # included allowance on every onboarding attempt. 20 rather than 1 for
            # the reason _chat_probe documents. Only here, because max_tokens is
            # not safe to send to an arbitrary endpoint (see
            # validate_openai_compatible).
            return await validate_openai_compatible(api_key, resolved_base, model, max_tokens=20)
        return await validate_openai_compatible(api_key, resolved_base, model)
    return {"ok": False, "error": "Unknown provider"}


def build_llm_client(effort_override: str | None = None):
    """Build an Anton LLMClient from the current user settings.

    Shared by the main responses handler and the credential probe handler
    so provider construction logic stays in one place.

    Reasoning effort is a persisted per-role setting
    (``planning_reasoning_effort`` / ``coding_reasoning_effort``) — chosen in the
    Settings UI beside each model dropdown, just like the model itself. Each
    level is forwarded in the provider's native shape (Anthropic
    ``output_config``, OpenAI ``reasoning`` / ``reasoning_effort``); None leaves
    the model's own default.

    `effort_override`, when set, is the composer's per-task Effort pick — it
    takes precedence over the persisted per-role setting for BOTH planning and
    coding roles for this call, bypassing the stored-vs-resolved staleness
    guard below (an explicit per-task pick is inherently valid for the model
    actually in use this turn, unlike a stale persisted choice that may have
    been made for a different model).
    """
    from anton.core.llm.anthropic import AnthropicProvider
    from anton.core.llm.client import LLMClient
    from anton.core.llm.openai import OpenAIProvider

    from cowork.common.settings.user_settings import (
        Provider,
        get_user_settings,
        provider_api_key,
    )

    settings = get_user_settings()

    # The published package still permits Anton releases from before ENG-2116.
    # Inspect the constructor once per client build so those versions keep
    # working without pretending they can refresh a credential per request.
    # A signature we cannot inspect is treated as unsupported: omitting a new
    # kwarg is safer than raising TypeError on every MindsHub turn.
    try:
        openai_provider_params = inspect.signature(OpenAIProvider).parameters
    except (TypeError, ValueError):
        openai_provider_params = {}
    supports_api_key_provider = "api_key_provider" in openai_provider_params
    warned_about_static_runtime_credential = False

    def _make_provider(role: Provider, effort: str | None = None):
        nonlocal warned_about_static_runtime_credential
        # Only pass reasoning_effort when it's actually set. This keeps
        # build_llm_client compatible with anton builds whose provider __init__
        # predates the kwarg (passing reasoning_effort=None unconditionally would
        # TypeError on every call, taking the whole agent down — not just effort
        # users) and avoids handing an unset effort to a provider that can't take it.
        effort_kw = {"reasoning_effort": effort} if effort else {}
        # Base URL is derived per-provider via provider_base_url() — never by
        # blindly reading the shared openai_base_url slot — so one provider's
        # stale base can't misroute another provider's key (see that helper).
        base = provider_base_url(
            role.value,
            openai_base_url=settings.openai_base_url or "",
            minds_url=settings.minds_url,
        )
        # Key is resolved per-provider via provider_api_key() — each provider
        # reads its own slot (gemini/openai-compatible fall back to the shared
        # openai slot when unset), so configuring one provider can't overwrite
        # or misroute another's key.
        key = provider_api_key(settings, role)
        if role == Provider.MINDS_CLOUD:
            if key is None:
                raise ValueError(f"{role.label} API key is not configured")
            # The runtime credential is local-only by contract. A static
            # settings/env key and every org-mode per-turn credential leave
            # this callback unset and keep their existing lifetime.
            #
            # The live supplier is available only when Anton advertises the
            # constructor kwarg. Older versions allowed by the package metadata
            # keep the construction-time credential and log that they cannot
            # adopt desktop refreshes; passing the unsupported kwarg would
            # TypeError on every signed-in MindsHub turn.
            #
            # This refreshes the MAIN-PROCESS provider only. anton snapshots
            # export_connection_info().api_key once per ChatSession and hands
            # that string to the scratchpad subprocess, which has no supplier —
            # so a pad-side LLM call still runs on the construction-time token.
            # ENG-2116 scopes that out; it needs a pad IPC contract.
            credential_kw: dict = {}
            if runtime_credential.get_minds_credential() is not None:
                if supports_api_key_provider:
                    credential_kw["api_key_provider"] = (
                        _current_runtime_minds_credential
                    )
                elif not warned_about_static_runtime_credential:
                    logger.warning(
                        "Installed anton does not support a per-request API-key "
                        "supplier; active MindsHub providers will keep their "
                        "construction-time credential until anton is upgraded"
                    )
                    warned_about_static_runtime_credential = True
            # The MindsHub gateway executes web_search / web_fetch server-side
            # over its chat.completions passthrough:
            # - the flavor must be set, or OpenAIProvider defaults to generic,
            #   reports no native web tools, and anton falls back to a
            #   web_search that needs an Exa/Brave key Cowork never asks for;
            # - state it outright rather than deriving it from the base URL —
            #   this branch already knows the endpoint is MindsHub, and a
            #   self-hosted gateway not spelling "mindshub.ai" would otherwise
            #   fall through to generic and lose search entirely.
            return OpenAIProvider(
                api_key=key.get_secret_value(),
                base_url=base,
                flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
                **credential_kw,
                **effort_kw,
            )
        if role in (Provider.OPENAI_COMPATIBLE, Provider.GEMINI):
            # A local endpoint authenticates by being reachable, so an
            # openai-compatible provider with a base URL and no key is a valid
            # config, not a broken one.
            #
            # Passing a non-empty string matters: anton drops a falsy api_key,
            # and the SDK then falls back to an ambient OPENAI_API_KEY — which
            # must never be sent to whatever machine the user pointed us at.
            # Derived from the role rather than written as a literal, so it is
            # what it looks like — a marker for an endpoint that authenticates
            # nobody — rather than something a reader or a scanner has to take
            # on trust as "not really a credential".
            if key is None and role == Provider.OPENAI_COMPATIBLE and base:
                return OpenAIProvider(
                    api_key=f"{role.value}-no-auth", base_url=base, **effort_kw
                )
            if key is None:
                raise ValueError(f"{role.label} API key is not configured")
            # No base for openai-compatible → OpenAIProvider would silently
            # default to api.openai.com and leak the BYO key to OpenAI. Fail
            # loudly instead (config_status surfaces this as "Set a base URL",
            # but callers don't all gate on config_ready, so enforce it here at
            # the build site too). gemini always has a base (Google), so this
            # only guards openai-compatible.
            if role == Provider.OPENAI_COMPATIBLE and not base:
                raise ValueError("OpenAI-compatible base URL is not configured")
            return OpenAIProvider(
                api_key=key.get_secret_value(), base_url=base, **effort_kw
            )
        provider_map = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}
        cls = provider_map.get(role.value)
        if cls is None:
            raise ValueError(f"Unknown provider: {role.value}")
        if key is None:
            raise ValueError(f"{role.label} API key is not configured")
        # base is None for anthropic/openai → SDK default host (OpenAIProvider
        # accepts base_url=None; AnthropicProvider takes no base_url kwarg).
        #
        # Direct OpenAI deliberately keeps the default (generic) flavor. The
        # flavor that would enable OpenAI's native web tools, FLAVOR_OPENAI,
        # also switches the whole transport from chat.completions to the
        # Responses API, whose path in anton does not yet:
        # - report truncation (no `response.incomplete` handler, so
        #   stop_reason and token usage stay unset and truncation recovery
        #   never fires),
        # - forward images returned inside a tool_result,
        # - attach Langfuse trace headers.
        # Native web search here waits on those gaps being closed in anton.
        if cls is OpenAIProvider:
            return cls(api_key=key.get_secret_value(), base_url=base, **effort_kw)
        return cls(api_key=key.get_secret_value(), **effort_kw)

    # Routing & summarization role: the cheap front-model that runs history
    # summarization (and later gates turns). Only pass it when the installed
    # anton's LLMClient accepts the kwargs — older builds predate ENG-648 and
    # would TypeError, taking the whole agent down. When absent, anton falls
    # back to the coding role internally, so behavior is preserved.
    router_kw: dict = {}
    try:
        _params = inspect.signature(LLMClient.__init__).parameters
        if "router_provider" in _params:
            router_kw = {
                "router_provider": _make_provider(settings.resolved_router_provider, None),
                "router_model": settings.resolved_router_model,
            }
    except (ValueError, TypeError):
        router_kw = {}

    # A reasoning-effort level is chosen in the Settings UI for a specific
    # model. When resolution swaps the model out from under the stored choice
    # (provider switch, or a wallet-locked aux model falling back to an
    # affordable one — ENG-1632), the stored effort must not travel with it:
    # the substitute may not advertise that level and the gateway 400s the
    # call. Same-model resolution keeps the effort.
    def _effort_for(stored: str | None, resolved: str | None, effort: str | None):
        return effort if effort and stored == resolved else None

    planning_effort = effort_override or _effort_for(
        settings.planning_model,
        settings.resolved_planning_model,
        settings.planning_reasoning_effort,
    )
    coding_effort = effort_override or _effort_for(
        settings.coding_model,
        settings.resolved_coding_model,
        settings.coding_reasoning_effort,
    )

    # Use the *resolved* provider/model (not the raw stored fields) so a
    # configured key takes effect even when planning_provider still points at
    # a keyless provider — the same resolution config_status reports, so the
    # readiness gate never claims "ready" for a client that would then throw.
    return LLMClient(
        planning_provider=_make_provider(
            settings.resolved_planning_provider,
            planning_effort,
        ),
        planning_model=settings.resolved_planning_model,
        coding_provider=_make_provider(
            settings.resolved_coding_provider,
            coding_effort,
        ),
        coding_model=settings.resolved_coding_model,
        **router_kw,
    )


def resolve_stored_key(settings: UserSettings, ptype: str) -> str:
    """Get the stored (unmasked) API key for a UI provider type."""
    from cowork.common.settings.user_settings import (
        UI_TYPE_TO_PROVIDER,
        provider_api_key_str,
    )
    provider = UI_TYPE_TO_PROVIDER.get(ptype)
    if provider is None:
        return ""
    # provider_api_key_str applies the gemini/openai-compatible → openai fallback,
    # so existing single-key configs still resolve here (Test button, key reveal).
    return provider_api_key_str(settings, provider)
