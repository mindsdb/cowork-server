"""Provider service — testing, validation, and config-readiness checks."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import httpx

from cowork.common.settings.app_settings import CODING_MODEL_DEFAULTS

if TYPE_CHECKING:
    from cowork.common.settings.user_settings import UserSettings

logger = logging.getLogger(__name__)


async def fetch_openai_compatible_models(base_url: str, api_key: str) -> Optional[list[str]]:
    """Fetch model ids from any OpenAI-compatible `/models` endpoint (NVIDIA
    NIM, Gemini's AI-Studio endpoint, OpenAI itself, generic openai-compatible).
    Returns None on any failure so callers fall back to a manually-typed list."""
    if not base_url or not api_key:
        return None
    base = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0), follow_redirects=True) as client:
            r = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code >= 400:
            return None
        data = r.json()
    except Exception as exc:
        logger.debug("openai-compatible /models fetch failed for %s: %s", base, exc)
        return None
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    ids = [str(row.get("id")).strip() for row in rows if isinstance(row, dict) and row.get("id")]
    ids = [i for i in ids if i]
    return ids or None


async def fetch_anthropic_models(api_key: str) -> Optional[list[str]]:
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            r = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
        if r.status_code >= 400:
            return None
        data = r.json()
    except Exception as exc:
        logger.debug("anthropic /models fetch failed: %s", exc)
        return None
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    ids = [str(row.get("id")).strip() for row in rows if isinstance(row, dict) and row.get("id")]
    return ids or None


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
    if provider == "openai-compatible":
        return await validate_openai_compatible(api_key, base_url or "https://api.openai.com/v1", model)
    return {"ok": False, "error": "Unknown provider"}


# ── Provider registry — multi-source failover ───────────────────────
#
# Registry entries (cowork.models.provider_config.ProviderConfig) are
# each one API key + base URL + model list, keyed by a stable slug.
# `build_llm_client(model="{slug}/{model_id}")` resolves that pick as
# the primary candidate and appends every other enabled registry entry
# as a failover fallback (see FailoverLLMProvider) — a free-tier 429 on
# the chosen model degrades to the next free source instead of failing
# the turn. When the registry is empty, behavior is byte-for-byte the
# legacy single-slot path (`_build_legacy_llm_client`), so a fresh
# install with only the old Settings fields configured keeps working
# unchanged.

_REGISTRY_DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
}


def _provider_for_row(row):
    """Build an LLMProvider for one registry row, or None if it can't be used
    (missing key, or a base_url-requiring type with no base_url set)."""
    from anton.core.llm.anthropic import AnthropicProvider
    from anton.core.llm.openai import OpenAIProvider
    from cowork.services.provider_registry import ProviderRegistryService

    key = ProviderRegistryService.decrypt_key(row)
    if not key:
        return None
    if row.type == "anthropic":
        return AnthropicProvider(api_key=key)

    base_url = row.base_url or _REGISTRY_DEFAULT_BASE_URLS.get(row.type)
    if not base_url:
        logger.warning("Provider '%s' (type=%s) has no base_url configured — skipping", row.slug, row.type)
        return None
    return OpenAIProvider(api_key=key, base_url=base_url)


def _candidates_for_rows(rows, *, pick_last_model: bool):
    from cowork.services.failover_provider import Candidate

    candidates = []
    for row in rows:
        if not row.enabled or not row.models:
            continue
        provider = _provider_for_row(row)
        if provider is None:
            continue
        model_id = row.models[-1] if pick_last_model and len(row.models) > 1 else row.models[0]
        candidates.append(Candidate(provider=provider, model=model_id, label=f"{row.slug}/{model_id}"))
    return candidates


def _resolve_model_override(rows, model: str):
    """Resolve a '{slug}/{model_id}' request string to (Candidate, slug), or None."""
    from cowork.services.failover_provider import Candidate

    slug, _, model_id = model.partition("/")
    if not model_id:
        return None
    row = next((r for r in rows if r.slug == slug and r.enabled), None)
    if row is None:
        return None
    provider = _provider_for_row(row)
    if provider is None:
        return None
    return Candidate(provider=provider, model=model_id, label=f"{row.slug}/{model_id}"), slug


def build_llm_client(model: str | None = None):
    """Build an Anton LLMClient, optionally pinned to a specific registered
    model for this turn, with automatic failover across the rest of the
    enabled provider registry.

    `model` is a "{slug}/{model_id}" string as offered by the composer's
    model picker (see ProviderRegistryService). When it doesn't resolve
    (unknown slug, disabled provider, or None), falls back to priority-
    ordered default routing across the registry, and if the registry has
    no entries at all, to the legacy single-slot Settings fields —
    unchanged behavior for installs that haven't touched the new registry.

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
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.db.session import get_open_session
    from cowork.services.failover_provider import FailoverLLMProvider
    from cowork.services.provider_registry import ProviderRegistryService

    settings = get_user_settings()
    session = get_open_session()
    try:
        rows = ProviderRegistryService(session).list(include_disabled=False)
    finally:
        session.close()

    if not rows:
        return _build_legacy_llm_client(settings)

    rows_sorted = sorted(rows, key=lambda r: (r.priority, r.slug))

    if model:
        resolved = _resolve_model_override(rows_sorted, model)
        if resolved is not None:
            primary, primary_slug = resolved
            fallback = _candidates_for_rows(
                [r for r in rows_sorted if r.slug != primary_slug], pick_last_model=False
            )
            provider = FailoverLLMProvider([primary, *fallback])
            return LLMClient(
                planning_provider=provider,
                planning_model=primary.model,
                coding_provider=provider,
                coding_model=primary.model,
            )
        logger.warning("Requested model '%s' did not resolve to a registered provider; using default routing", model)

    planning_candidates = _candidates_for_rows(rows_sorted, pick_last_model=False)
    if not planning_candidates:
        return _build_legacy_llm_client(settings)
    coding_candidates = _candidates_for_rows(rows_sorted, pick_last_model=True) or planning_candidates

    planning_provider = FailoverLLMProvider(planning_candidates)
    coding_provider = FailoverLLMProvider(coding_candidates)
    return LLMClient(
        planning_provider=planning_provider,
        planning_model=planning_candidates[0].model,
        coding_provider=coding_provider,
        coding_model=coding_candidates[0].model,
    )


def _build_legacy_llm_client(settings):
    """The original single-slot provider resolution, kept verbatim as the
    fallback path for installs with an empty provider registry."""
    from anton.core.llm.client import LLMClient
    from anton.core.llm.anthropic import AnthropicProvider
    from anton.core.llm.openai import OpenAIProvider

    from cowork.common.settings.user_settings import Provider

    def _make_provider(role: Provider):
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
