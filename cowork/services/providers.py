"""Provider service — testing, validation, and config-readiness checks."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional

import httpx
from pydantic import SecretStr

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
_minds_models_cache: dict[str, tuple[float, Optional[list[str]]]] = {}


async def fetch_minds_models(minds_url: str, api_key: str) -> Optional[list[str]]:
    """Fetch supported model ids from MindsHub's OpenAI-compatible
    `/v1/models` endpoint. Returns the model-id list, or None on any
    failure so the caller falls back to the static list."""
    if not minds_url or not api_key:
        return None
    base = minds_chat_base_url(minds_url)

    now = time.monotonic()
    cached = _minds_models_cache.get(base)
    if cached:
        ts, val = cached
        ttl = _MINDS_MODELS_TTL if val else _MINDS_MODELS_FAIL_TTL
        if (now - ts) < ttl:
            return val

    def _remember(val: Optional[list[str]]) -> Optional[list[str]]:
        _minds_models_cache[base] = (time.monotonic(), val)
        return val

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(6.0), follow_redirects=True
        ) as client:
            r = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if r.status_code >= 400:
            logger.debug("minds /models fetch returned HTTP %s", r.status_code)
            return _remember(None)
        data = r.json()
    except Exception as exc:
        logger.debug("minds /models fetch failed: %s", exc)
        return _remember(None)

    # OpenAI shape: {"object": "list", "data": [{"id": "...", ...}]}.
    # Accept a bare list too, defensively.
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return _remember(None)
    ids = [
        str(row.get("id")).strip()
        for row in rows
        if isinstance(row, dict) and row.get("id")
    ]
    ids = [i for i in ids if i]
    return _remember(ids or None)


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
            base = (p.get("mindsUrl") or "https://api.mindshub.ai").rstrip("/")
            chat_url = minds_chat_base_url(base)
            return await _check(f"{chat_url}/models", {"Authorization": f"Bearer {key}"})
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
                json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]},
            )
            if r.status_code in (200, 201):
                return {"ok": True}
            msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}") if r.content else f"HTTP {r.status_code}"
            return {"ok": False, "error": msg}
    except Exception:
        return {"ok": False, "error": "Cannot connect"}


async def validate_minds(api_key: str, base_url: str = "https://mdb.ai") -> dict[str, Any]:
    try:
        chat_base = minds_chat_base_url(base_url.rstrip("/"))
        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(f"{chat_base}/models", headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code in (401, 403):
            return {"ok": False, "error": "Invalid API key"}
        if 200 <= r.status_code < 300:
            return {"ok": True}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
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
        return await validate_minds(api_key, base_url or "https://mdb.ai")
    if provider == "openai-compatible":
        return await validate_openai_compatible(api_key, base_url or "https://api.openai.com/v1", model)
    return {"ok": False, "error": "Unknown provider"}


def build_llm_client():
    """Build an Anton LLMClient from the current user settings.

    Shared by the main responses handler and the credential probe handler
    so provider construction logic stays in one place.
    """
    from anton.core.llm.client import LLMClient
    from anton.core.llm.anthropic import AnthropicProvider
    from anton.core.llm.openai import OpenAIProvider

    from cowork.common.settings.user_settings import get_user_settings, Provider

    settings = get_user_settings()

    def _make_provider(role: Provider):
        if role == Provider.MINDS_CLOUD:
            key = settings.minds_api_key
            if key is None:
                raise ValueError("MindsHub API key is not configured")
            return OpenAIProvider(
                api_key=key.get_secret_value(),
                base_url=minds_chat_base_url(settings.minds_url),
            )
        if role in (Provider.OPENAI_COMPATIBLE, Provider.GEMINI):
            key = settings.openai_api_key
            if key is None:
                raise ValueError("OpenAI API key is not configured")
            return OpenAIProvider(
                api_key=key.get_secret_value(),
                base_url=settings.openai_base_url or "https://api.openai.com/v1",
            )
        provider_map = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}
        cls = provider_map.get(role.value)
        if cls is None:
            raise ValueError(f"Unknown provider: {role.value}")
        key = getattr(settings, f"{role.value}_api_key")
        if key is None:
            raise ValueError(f"{role.value} API key is not configured")
        return cls(api_key=key.get_secret_value())

    return LLMClient(
        planning_provider=_make_provider(settings.planning_provider),
        planning_model=settings.planning_model,
        coding_provider=_make_provider(settings.coding_provider),
        coding_model=settings.coding_model,
    )


def resolve_stored_key(settings: UserSettings, ptype: str) -> str:
    """Get the stored (unmasked) API key for a UI provider type."""
    from cowork.common.settings.user_settings import UI_TYPE_TO_PROVIDER
    provider = UI_TYPE_TO_PROVIDER.get(ptype)
    if provider is None:
        return ""
    field = provider.api_key_field
    val = getattr(settings, field, None)
    return val.get_secret_value() if isinstance(val, SecretStr) else ""
