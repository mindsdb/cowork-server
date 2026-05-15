"""Cowork-owned inference profile resolution."""

from __future__ import annotations

from typing import Any

from .schemas import ResolvedInferenceProfile


def _provider_capabilities(provider_type: str) -> dict[str, bool]:
    base = {
        "streaming": True,
        "tool_calling": True,
        "json_schema": False,
        "vision": False,
        "long_context": False,
    }
    if provider_type in {"minds-cloud", "openai", "openai-compatible", "gemini"}:
        base.update({"json_schema": True, "vision": provider_type in {"openai", "gemini"}})
    if provider_type == "anthropic":
        base.update({"vision": True, "long_context": True})
    return base


def _api_key_ref_for(provider_type: str) -> str:
    if provider_type == "anthropic":
        return "ANTON_ANTHROPIC_API_KEY"
    if provider_type == "minds-cloud":
        return "ANTON_MINDS_API_KEY"
    if provider_type in {"openai", "gemini", "openai-compatible"}:
        return "ANTON_OPENAI_API_KEY"
    return ""


def _provider_label(settings_route: Any, provider_type: str) -> str:
    return settings_route.PROVIDER_TYPE_LABELS.get(provider_type, provider_type or "Unknown")


def resolve_inference_profile() -> ResolvedInferenceProfile:
    """Resolve the active Cowork inference profile.

    The extracted server package owns the profile schema and validation
    primitives. Concrete settings storage is supplied by the embedding app or
    deployment, so this placeholder intentionally fails until wired there.
    """
    raise NotImplementedError("Inference profile resolution must be provided by the embedding Cowork application.")


def build_inference_profile(
    *,
    provider_type: str,
    provider_label: str | None = None,
    base_url: str = "",
    planning_model: str,
    coding_model: str = "",
    coding_provider_type: str | None = None,
    coding_provider_label: str | None = None,
    coding_base_url: str | None = None,
) -> ResolvedInferenceProfile:
    coding_provider_type = coding_provider_type or provider_type
    provider_label = provider_label or provider_type or "Unknown"
    coding_provider_label = coding_provider_label or coding_provider_type or provider_label
    coding_base_url = coding_base_url if coding_base_url is not None else base_url
    return ResolvedInferenceProfile(
        id=f"{provider_type}:{planning_model}:{coding_model}",
        provider_type=provider_type,
        provider_label=provider_label,
        base_url=base_url,
        api_key_ref=_api_key_ref_for(provider_type),
        planning_provider_type=provider_type,
        planning_provider_label=provider_label,
        planning_base_url=base_url,
        planning_api_key_ref=_api_key_ref_for(provider_type),
        coding_provider_type=coding_provider_type,
        coding_provider_label=coding_provider_label,
        coding_base_url=coding_base_url,
        coding_api_key_ref=_api_key_ref_for(coding_provider_type),
        planning_model=planning_model,
        coding_model=coding_model,
        capabilities={
            **_provider_capabilities(provider_type),
            **{
                f"coding_{key}": value
                for key, value in _provider_capabilities(coding_provider_type).items()
            },
        },
    )


def profile_for_storage(profile: ResolvedInferenceProfile) -> dict[str, Any]:
    return profile.safe_dump()


def validate_inference_profile(profile: ResolvedInferenceProfile) -> tuple[bool, str]:
    if profile.provider_type in {"", "unknown"}:
        return False, "No inference provider is configured."
    if not profile.planning_model:
        return False, "No planning model is configured."
    if profile.provider_type == "openai-compatible" and not profile.base_url:
        return False, "OpenAI-compatible inference requires a base URL."
    if profile.coding_model and profile.coding_provider_type == "openai-compatible" and not profile.coding_base_url:
        return False, "OpenAI-compatible coding inference requires a base URL."
    return True, ""
