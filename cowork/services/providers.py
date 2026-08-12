"""Provider service — testing, validation, and config-readiness checks."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, NamedTuple, Optional
from urllib.parse import urlparse

import httpx

from cowork.common.settings.app_settings import default_minds_api_host

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


def _empty_listing() -> MindsModelListing:
    """The "we got nothing" listing: ``ids`` None, every map empty.

    One place to build it, so the failure paths don't each repeat six literals
    that have to agree. A factory rather than a shared constant because a
    failure is also cached, per base URL: handing every entry the same five dict
    objects means one in-place write downstream would corrupt the cached failure
    of every gateway at once. Nothing mutates them today.
    """
    return MindsModelListing(None, {}, {}, {}, {}, {})


# Cache value: (timestamp, listing). listing.ids is None on failure.
_minds_models_cache: dict[str, tuple[float, MindsModelListing]] = {}


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


async def fetch_minds_models(
    minds_url: str, api_key: str, *, force_refresh: bool = False
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
    full HTTP timeout for as long as the outage lasts.
    """
    if not minds_url or not api_key:
        return _empty_listing()
    base = minds_chat_base_url(minds_url)

    now = time.monotonic()
    cached = _minds_models_cache.get(base)
    if cached:
        ts, val = cached
        ttl = _MINDS_MODELS_TTL if val.ids else _MINDS_MODELS_FAIL_TTL
        # force_refresh only overrides the success TTL — see the negative-cache
        # note above.
        if (now - ts) < ttl and not (force_refresh and val.ids):
            return val

    def _remember(val: MindsModelListing) -> MindsModelListing:
        _minds_models_cache[base] = (time.monotonic(), val)
        return val

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(6.0), follow_redirects=True
        ) as client:
            # Trailing slash is required: the MindsHub router serves the
            # listing at `/models/` and a recent minds-inference release
            # stopped cleanly redirecting the slashless `/models`, which
            # left this fetch empty and emptied the model picker. Hitting
            # `/models/` directly is what the other frameworks' shared
            # model-catalog helper already does.
            r = await client.get(
                f"{base}/models/",
                headers={"Authorization": f"Bearer {api_key}"},
            )
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
    return _remember(
        MindsModelListing((ids or None), efforts, enabled, labels, providers, families)
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
                                     model: str | None = None) -> dict[str, Any]:
    try:
        normalized = base_url.rstrip("/")
        chat_url = f"{normalized}/chat/completions" if re.search(r"/v\d", normalized) else f"{normalized}/v1/chat/completions"
        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                chat_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model or "gpt-5.5", "messages": [{"role": "user", "content": "ping"}]},
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
        return await validate_openai_compatible(api_key, base_url or "https://api.openai.com/v1", model)
    return {"ok": False, "error": "Unknown provider"}


def build_llm_client():
    """Build an Anton LLMClient from the current user settings.

    Shared by the main responses handler and the credential probe handler
    so provider construction logic stays in one place.

    Reasoning effort is a persisted per-role setting
    (``planning_reasoning_effort`` / ``coding_reasoning_effort``) — chosen in the
    Settings UI beside each model dropdown, just like the model itself. Each
    level is forwarded in the provider's native shape (Anthropic
    ``output_config``, OpenAI ``reasoning`` / ``reasoning_effort``); None leaves
    the model's own default.
    """
    from anton.core.llm.client import LLMClient
    from anton.core.llm.anthropic import AnthropicProvider
    from anton.core.llm.openai import OpenAIProvider

    from cowork.common.settings.user_settings import (
        get_user_settings,
        provider_api_key,
        Provider,
    )

    settings = get_user_settings()

    def _make_provider(role: Provider, effort: str | None = None):
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
                **effort_kw,
            )
        if role in (Provider.OPENAI_COMPATIBLE, Provider.GEMINI):
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
    import inspect as _inspect

    router_kw: dict = {}
    try:
        _params = _inspect.signature(LLMClient.__init__).parameters
        if "router_provider" in _params:
            router_kw = {
                "router_provider": _make_provider(settings.resolved_router_provider, None),
                "router_model": settings.resolved_router_model,
            }
    except (ValueError, TypeError):
        router_kw = {}

    # Use the *resolved* provider/model (not the raw stored fields) so a
    # configured key takes effect even when planning_provider still points at
    # a keyless provider — the same resolution config_status reports, so the
    # readiness gate never claims "ready" for a client that would then throw.
    return LLMClient(
        planning_provider=_make_provider(
            settings.resolved_planning_provider, settings.planning_reasoning_effort
        ),
        planning_model=settings.resolved_planning_model,
        coding_provider=_make_provider(
            settings.resolved_coding_provider, settings.coding_reasoning_effort
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
