import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, get_args

from pydantic import Field, SecretStr, field_validator, model_validator

from cowork.common.settings.app_settings import (
    CODING_MODEL_DEFAULTS,
    PLANNING_MODEL_DEFAULTS,
    Settings,
    get_app_settings,
)


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"
    @property
    def label(self) -> str:
        return PROVIDER_LABELS[self]

    @property
    def ui_value(self) -> str:
        """Provider id as exposed to the UI / anton (dashes, not underscores).

        Single source for the ``value.replace("_", "-")`` normalization that was
        otherwise reinvented at every provider boundary (provider_base_url,
        _resolve_coding, the hermes harness, the AntonSettings bridge). A future
        provider name can't normalize correctly in one place and wrong in
        another — the latter silently routes to AnthropicProvider."""
        return self.value.replace("_", "-")


PROVIDER_LABELS: dict["Provider", str] = {
    Provider.ANTHROPIC: "Anthropic",
    Provider.OPENAI: "OpenAI",
    Provider.GEMINI: "Gemini",
    Provider.OPENAI_COMPATIBLE: "OpenAI-compatible",
}

_PROVIDER_KEY_FIELDS: dict["Provider", str] = {
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.OPENAI: "openai_api_key",
    Provider.GEMINI: "openai_api_key",
    Provider.OPENAI_COMPATIBLE: "openai_api_key",
}

# gemini and openai-compatible historically shared the single openai_api_key
# slot (alongside openai), which meant configuring one could overwrite/misroute
# another provider's key. They now have dedicated slots, but we fall back to the
# shared openai_api_key when a dedicated slot is empty so existing single-key
# configs keep working with no migration; isolation kicks in once a distinct key
# is set in the new field.
_SHARED_KEY_FALLBACK_FIELDS = frozenset({"gemini_api_key", "openai_compatible_api_key"})


def provider_api_key(settings: "UserSettings", provider: "Provider"):
    """Resolve a provider's API key from its dedicated slot.

    Returns the SecretStr in the provider's own key field; for gemini /
    openai-compatible, falls back to the shared ``openai_api_key`` when their
    dedicated slot is unset (backward compatibility — see above). Returns None
    when nothing is configured.
    """
    field = _PROVIDER_KEY_FIELDS[provider]
    val = getattr(settings, field, None)
    if val is None and field in _SHARED_KEY_FALLBACK_FIELDS:
        return settings.openai_api_key
    return val


def provider_api_key_str(settings: "UserSettings", provider: "Provider") -> str:
    """``provider_api_key`` as a plain unmasked string ('' when unset).

    Most call sites (key reveal, the Test button, hermes env sync, the OC model
    overlay) just need the raw value to hand to a client. Folding the
    ``SecretStr → str`` unwrap into one helper removes the per-site inline
    imports and the subtly different empty-handling variants that had drifted
    across reveal_key / resolve_stored_key / recommended_models / hermes."""
    val = provider_api_key(settings, provider)
    return val.get_secret_value() if isinstance(val, SecretStr) else ""


def _resolved_model(
    resolved_provider: "Provider",
    preferred_provider: "Provider",
    user_model: str | None,
    defaults: dict[str, str],
) -> str | None:
    """Resolve a role's model given the readiness resolver's provider switch.

    The single load-bearing rule, shared by resolved_planning_model and
    resolved_coding_model so it can't drift between the two:

      - provider NOT switched → keep the user's chosen model.
      - provider switched → use the resolved provider's canonical default.
        NEVER fall back to the original provider's model — that would hand e.g.
        a Claude id to an openai-compatible / MindsHub endpoint (misrouting).
      - resolved provider has no canonical default (openai-compatible) → None,
        so config_status's model gate reports "select a model" rather than
        silently running a wrong model.
    """
    if resolved_provider == preferred_provider:
        return user_model
    return defaults.get(resolved_provider.value)

# Provider types as exposed to the UI (uses dashes, not underscores)
UI_PROVIDER_TYPES = ("anthropic", "openai", "gemini", "openai-compatible")

UI_PROVIDER_TYPE_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "openai-compatible": "OpenAI-compatible",
}

UI_TYPE_TO_PROVIDER: dict[str, "Provider"] = {
    "anthropic": Provider.ANTHROPIC,
    "openai": Provider.OPENAI,
    "gemini": Provider.GEMINI,
    "openai-compatible": Provider.OPENAI_COMPATIBLE,
}


class _DynamicOptions:
    """Annotated metadata marker for fields whose valid options are resolved lazily from a callable."""

    def __init__(self, fn: Callable[[], list[str]]) -> None:
        self._fn = fn

    def get(self) -> list[str]:
        return self._fn()


_BUILD_STAMP_PATH = Path.home() / ".cowork" / "server-build-stamp.json"


def _read_build_stamp() -> dict | None:
    try:
        if _BUILD_STAMP_PATH.is_file():
            return json.loads(_BUILD_STAMP_PATH.read_text())
    except Exception:
        pass
    return None


_config_status_cache: dict | None = None
_config_status_cache_at: float = 0.0
_CONFIG_STATUS_CACHE_TTL: float = 60.0


def _harness_options() -> list[str]:
    from cowork.harnesses.base import _registry
    return list(_registry.keys())


class UserSettings(Settings):
    # The recommended-model catalog and per-provider model defaults are
    # global, application-level config and live in app_settings
    # (RECOMMENDED_MODELS / RECOMMENDED_PAIR / *_MODEL_DEFAULTS).

    # ── Provider / model settings ──

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        title="Anthropic API Key",
        description="API key for Anthropic Claude models. Required if not using OpenAI.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        title="OpenAI API Key",
        description="API key for OpenAI models. Required if not using Anthropic.",
    )
    google_oauth_client_id: str | None = Field(
        default=None,
        title="Google OAuth Client ID",
        description="Enables one-click 'Sign in with Google' for Gmail, Google Ads, GA4, Drive, and Calendar connectors, so every user of this install can connect those apps without pasting their own credentials. From Google Cloud Console -> APIs & Services -> Credentials (Desktop app OAuth client).",
    )
    google_oauth_client_secret: SecretStr | None = Field(
        default=None,
        title="Google OAuth Client Secret",
        description="Paired with the Client ID above, from the same Google Cloud Console credential.",
    )
    planning_provider: Provider = Field(
        default=Provider.ANTHROPIC,
        title="Planning Provider",
        description="The provider to use for the reasoning/planning model.",
    )
    planning_model: str | None = Field(
        default=None,
        title="Planning Model",
        description="The reasoning model used for planning. Defaults to the recommended model for the selected provider.",
    )
    coding_provider: Provider = Field(
        default=Provider.ANTHROPIC,
        title="Coding Provider",
        description="The provider to use for the coding model.",
    )
    coding_model: str | None = Field(
        default=None,
        title="Coding Model",
        description="The coding model. Defaults to the recommended model for the selected provider.",
    )
    planning_reasoning_effort: str | None = Field(
        default=None,
        title="Planning Reasoning Effort",
        description=(
            "Opaque reasoning-effort level for the planning model (e.g. 'low' | "
            "'medium' | 'high'). None uses the model's default. Only meaningful for "
            "models that advertise effort levels."
        ),
    )
    coding_reasoning_effort: str | None = Field(
        default=None,
        title="Coding Reasoning Effort",
        description=(
            "Opaque reasoning-effort level for the coding model. None uses the "
            "model's default. Only meaningful for models that advertise effort levels."
        ),
    )
    harness: Annotated[str, _DynamicOptions(_harness_options)] = Field(
        default="anton",
        title="Harness",
        description="The AI harness used to generate responses.",
    )
    channels_harness: Annotated[str, _DynamicOptions(_harness_options)] = Field(
        default_factory=lambda: (get_app_settings().channels_harness or "anton"),
        title="Channel Agent",
        description="The AI harness that serves messaging-channel conversations.",
    )

    # ── UI preferences ──

    greeting: str = Field(
        default="Let's knock something off your list",
        title="Greeting",
        description="The greeting message shown on the home screen.",
    )
    tone: str = Field(
        default="balanced",
        title="Tone",
        description="The conversational tone for responses.",
    )
    auto_pin: bool = Field(
        default=True,
        title="Auto Pin",
        description="Automatically pin important items.",
    )
    show_dots: bool = Field(
        default=True,
        title="Show Dots",
        description="Show dot grid background.",
    )
    show_counters: bool = Field(
        default=True,
        title="Show Counters",
        description="Show counters in the UI.",
    )
    accent_variant: str = Field(
        default="aqua",
        title="Accent Variant",
        description="UI accent color variant.",
    )
    memory_enabled: bool = Field(
        default=True,
        title="Memory Enabled",
        description="Enable conversation memory.",
    )
    memory_mode: str = Field(
        default="autopilot",
        title="Memory Mode",
        description="How memory is managed (autopilot or manual).",
    )
    episodic_memory: bool = Field(
        default=True,
        title="Episodic Memory",
        description="Enable episodic memory for conversations.",
    )
    proactive_dashboards: bool = Field(
        default=False,
        title="Proactive Dashboards",
        description="Enable proactive dashboard suggestions.",
    )
    act_first: bool = Field(
        default=True,
        title="Act first, ask later",
        description=(
            "Act on reasonable defaults and state assumptions inline instead of "
            "stopping to ask. Turn off for a more cautious, ask-first agent."
        ),
    )
    ui_update_mode: str = Field(
        default="manual",
        title="UI Update Mode",
        description="How UI updates are applied (manual or auto).",
    )
    publish_url: str = Field(
        default="",
        title="Publish URL",
        description="Base URL for publishing artifacts. When empty, derived from the MindsHub endpoint (api[.env].mindshub.ai → view[.env].mindshub.ai, else prod); set explicitly to override.",
    )
    openai_base_url: str = Field(
        default="",
        title="OpenAI Base URL",
        description="Base URL for OpenAI-compatible providers.",
    )
    model_mode: str = Field(
        default="default",
        title="Model Mode",
        description="Whether to use default or custom model assignments (default or custom).",
    )
    model_overrides: str = Field(
        default="{}",
        title="Model Overrides",
        description="JSON-encoded per-role provider/model overrides when model_mode is custom.",
    )
    providers_json: str = Field(
        default="[]",
        title="Providers",
        description="JSON-encoded list of configured provider entries for the settings UI.",
    )
    provider_status: str = Field(
        default="{}",
        title="Provider Status",
        description="JSON-encoded map of provider type → last connectivity-test status (ok|fail). Persisted so the Settings dots survive a reload.",
    )
    provider_status_details: str = Field(
        default="{}",
        title="Provider Status Details",
        description="JSON-encoded map of provider type → last connectivity-test detail (e.g. an HTTP code).",
    )

    @field_validator("harness")
    @classmethod
    def validate_harness(cls, v: str) -> str:
        options = _harness_options()
        if v not in options:
            available = ", ".join(options) or "none"
            raise ValueError(f"Unknown harness '{v}'. Available: {available}")
        return v

    @model_validator(mode='after')
    def apply_model_defaults(self) -> 'UserSettings':
        if self.planning_model is None:
            self.planning_model = PLANNING_MODEL_DEFAULTS.get(self.planning_provider.value)
        if self.coding_model is None:
            self.coding_model = CODING_MODEL_DEFAULTS.get(self.coding_provider.value)
        return self

    @model_validator(mode='before')
    def migrate_minds_cloud(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get('planning_provider') == 'minds_cloud':
                data['planning_provider'] = 'anthropic'
            if data.get('coding_provider') == 'minds_cloud':
                data['coding_provider'] = 'anthropic'
        return data
    def apply_model_defaults(self) -> 'UserSettings':
        if self.planning_model is None:
            self.planning_model = PLANNING_MODEL_DEFAULTS.get(self.planning_provider.value)
        if self.coding_model is None:
            self.coding_model = CODING_MODEL_DEFAULTS.get(self.coding_provider.value)
        return self

    @property
    def config_status(self) -> dict[str, Any]:
        """Whether ANY execution path exists for a chat turn.

        Coworker-architecture-aware (2026-07-04): ready when either
          1. an installed CLI coworker exists (Claude Code / Antigravity /
              Codex — run on the user's subscription login, no key), or
          2. the provider registry has an enabled entry with a key+models.

        The result is TTL-cached (60 s) to avoid redundant CLI/DB scans.
        The build stamp from ~/.cowork/server-build-stamp.json is included
        in every response.

        Imports are lazy to avoid the
        user_settings ⇄ harnesses import cycle.
        """
        global _config_status_cache, _config_status_cache_at

        now = time.monotonic()
        if _config_status_cache is not None and (now - _config_status_cache_at) < _CONFIG_STATUS_CACHE_TTL:
            return _config_status_cache

        google_oauth_configured = bool(self.google_oauth_client_id and self.google_oauth_client_secret)

        # 1. Installed CLI coworker?
        try:
            from cowork.harnesses.base import _registry
            for cls in _registry.values():
                find_cli = getattr(cls, "find_cli", None)
                if find_cli is None:
                    continue
                harness = cls()
                if harness.find_cli() is not None:
                    result = {
                        "config_ready": True,
                        "config_error": None,
                        "provider": "cli",
                        "provider_label": harness.label,
                        "model": "",
                        "build": _read_build_stamp(),
                        "google_oauth_configured": google_oauth_configured,
                    }
                    _config_status_cache = result
                    _config_status_cache_at = now
                    return result
        except Exception:
            pass  # registry unavailable during early boot — fall through

        # 2. Usable provider-registry entry?
        try:
            from cowork.db.session import get_open_session
            from cowork.services.provider_registry import ProviderRegistryService

            session = get_open_session()
            try:
                rows = ProviderRegistryService(session).list(include_disabled=False)
            finally:
                session.close()
            usable = next((r for r in rows if r.api_key_encrypted and r.models), None)
            if usable is not None:
                result = {
                    "config_ready": True,
                    "config_error": None,
                    "provider": usable.type,
                    "provider_label": usable.label,
                    "model": usable.models[0] if usable.models else "",
                    "build": _read_build_stamp(),
                    "google_oauth_configured": google_oauth_configured,
                }
                _config_status_cache = result
                _config_status_cache_at = now
                return result
        except Exception:
            pass

        result = {
            "config_ready": True,
            "config_error": "No coworker available — install a CLI agent (e.g. Claude Code) or add a model source in Settings.",
            "provider": "none",
            "provider_label": "None",
            "model": "",
            "build": _read_build_stamp(),
            "google_oauth_configured": google_oauth_configured,
        }
        _config_status_cache = result
        _config_status_cache_at = now
        return result

    @staticmethod
    def field_is_sensitive(field_name: str) -> bool:
        annotation = UserSettings.model_fields[field_name].annotation
        return SecretStr in get_args(annotation)

    @staticmethod
    def field_options(field_name: str) -> list[str] | None:
        field_info = UserSettings.model_fields[field_name]
        for meta in field_info.metadata:
            if isinstance(meta, _DynamicOptions):
                return meta.get()
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return [e.value for e in annotation]
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, Enum):
                return [e.value for e in arg]
        return None


_cache: UserSettings | None = None


def get_user_settings() -> UserSettings:
    global _cache
    if _cache is None:
        _cache = _load_from_db()
    return _cache


def invalidate_user_settings_cache() -> None:
    global _cache
    _cache = None


def _load_from_db() -> UserSettings:
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    session = get_open_session()
    try:
        return SettingService(session).load()
    finally:
        session.close()
