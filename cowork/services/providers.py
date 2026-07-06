"""Provider service — testing, validation, and config-readiness checks."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import httpx

from cowork.common.settings.app_settings import CODING_MODEL_DEFAULTS, default_minds_api_host

if TYPE_CHECKING:
    from cowork.common.settings.user_settings import UserSettings

logger = logging.getLogger(__name__)


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
# Cache value: (timestamp, (ids, efforts_map)). ids is None on failure.
_minds_models_cache: dict[str, tuple[float, tuple[Optional[list[str]], dict[str, dict]]]] = {}


async def fetch_minds_models(
    minds_url: str, api_key: str
) -> tuple[Optional[list[str]], dict[str, dict]]:
    """Fetch supported models from MindsHub's OpenAI-compatible `/v1/models`.

    Returns ``(ids, efforts)`` where ``ids`` is the model-id list (or None on
    any failure so the caller falls back to the static list) and ``efforts``
    maps a model id to ``{"efforts": [...], "default": "..."}`` for every model
    that advertises ``reasoning_efforts`` — the source of truth for which models
    accept an effort level and at which levels.
    """
    if not minds_url or not api_key:
        return None, {}
    base = minds_chat_base_url(minds_url)

    now = time.monotonic()
    cached = _minds_models_cache.get(base)
    if cached:
        ts, val = cached
        ttl = _MINDS_MODELS_TTL if val[0] else _MINDS_MODELS_FAIL_TTL
        if (now - ts) < ttl:
            return val

    def _remember(
        val: tuple[Optional[list[str]], dict[str, dict]],
    ) -> tuple[Optional[list[str]], dict[str, dict]]:
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
            return _remember((None, {}))
        data = r.json()
    except Exception as exc:
        logger.debug("minds /models fetch failed: %s", exc)
        return _remember((None, {}))

    # OpenAI shape: {"object": "list", "data": [{"id": "...", ...}]}.
    # Accept a bare list too, defensively. Each row may carry the non-standard
    # extension fields `reasoning_efforts` (list) and `default_reasoning_effort`
    # (str) — OpenAI clients ignore unknown keys; we surface them for the picker.
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return _remember((None, {}))
    ids: list[str] = []
    efforts: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        model_id = str(row.get("id")).strip()
        if not model_id:
            continue
        ids.append(model_id)
        levels = row.get("reasoning_efforts")
        if isinstance(levels, list) and levels:
            entry: dict = {"efforts": [str(x) for x in levels]}
            default = row.get("default_reasoning_effort")
            if default:
                entry["default"] = str(default)
            efforts[model_id] = entry
    return _remember(((ids or None), efforts))


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


async def ping_provider(p: dict[str, Any]) -> tuple[str, str]:
    """Ping a single provider and return (status, detail)."""
    ptype = p.get("type")
    key = (p.get("apiKey") or "").strip()
    timeout = httpx.Timeout(12.0)

    async def _check(url: str, headers: dict[str, str]) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            return ("ok", f"HTTP {r.status_code}") if r.status_code < 400 else ("fail", f"HTTP {r.status_code}")

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
        return ("ok", f"HTTP {r.status_code}") if r.status_code < 400 else ("fail", f"HTTP {r.status_code}")

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
            model = (p.get("model") or "").strip() or CODING_MODEL_DEFAULTS["minds_cloud"]
            return await _chat_probe(
                f"{chat_url}/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                model,
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
            msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}") if r.content else f"HTTP {r.status_code}"
            return {"ok": False, "error": msg}
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
                json={"model": CODING_MODEL_DEFAULTS["minds_cloud"], "max_tokens": 20,
                      "messages": [{"role": "user", "content": "ping"}]},
            )
        if r.status_code in (401, 403):
            return {"ok": False, "error": "Invalid API key"}
        if 200 <= r.status_code < 300:
            return {"ok": True}
        msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}") if r.content else f"HTTP {r.status_code}"
        return {"ok": False, "error": msg}
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
            if r.status_code in (401, 403):
                return {"ok": False, "error": "Invalid API key"}
            msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}") if r.content else f"HTTP {r.status_code}"
            return {"ok": False, "error": msg}
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
            return OpenAIProvider(
                api_key=key.get_secret_value(), base_url=base, **effort_kw
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
        if cls is OpenAIProvider:
            return cls(api_key=key.get_secret_value(), base_url=base, **effort_kw)
        return cls(api_key=key.get_secret_value(), **effort_kw)

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
