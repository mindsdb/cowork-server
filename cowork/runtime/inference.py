"""Cowork-owned inference resolver primitives."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cowork.runtime.schemas import ResolvedInferenceProfile


class InferenceProviderConfig(BaseModel):
    type: str
    label: str = ""
    base_url: str = ""
    api_key_ref: str = ""
    is_default: bool = False
    models: dict[str, str] = Field(default_factory=dict)


class InferenceSettings(BaseModel):
    providers: list[InferenceProviderConfig] = Field(default_factory=list)
    model_mode: str = "default"
    model_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    default_models: dict[str, tuple[str, str]] = Field(default_factory=dict)


def provider_capabilities(provider_type: str) -> dict[str, bool]:
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


def resolve_inference_profile(settings: InferenceSettings) -> ResolvedInferenceProfile:
    provider = next((p for p in settings.providers if p.is_default), None)
    if provider is None and settings.providers:
        provider = settings.providers[0]
    if provider is None:
        return ResolvedInferenceProfile()

    planning = provider.models.get("planning", "")
    coding = provider.models.get("coding", "")
    defaults = settings.default_models.get(provider.type)
    if settings.model_mode != "custom" and defaults:
        planning, coding = defaults

    planning_override = settings.model_overrides.get("planning")
    coding_override = settings.model_overrides.get("coding")
    if planning_override and planning_override.get("providerType") == provider.type:
        planning = planning_override.get("model") or planning
    if coding_override and coding_override.get("providerType") == provider.type:
        coding = coding_override.get("model") or coding

    label = provider.label or provider.type
    return ResolvedInferenceProfile(
        id=f"{provider.type}:{planning}:{coding}",
        provider_type=provider.type,
        provider_label=label,
        base_url=provider.base_url,
        api_key_ref=provider.api_key_ref,
        planning_model=planning,
        coding_model=coding,
        capabilities=provider_capabilities(provider.type),
    )


def validate_inference_profile(profile: ResolvedInferenceProfile) -> tuple[bool, str]:
    if profile.provider_type in {"", "unknown"}:
        return False, "No inference provider is configured."
    if not profile.planning_model:
        return False, "No planning model is configured."
    if profile.provider_type == "openai-compatible" and not profile.base_url:
        return False, "OpenAI-compatible inference requires a base URL."
    return True, ""


def profile_for_storage(profile: ResolvedInferenceProfile) -> dict[str, Any]:
    return profile.safe_dump()

