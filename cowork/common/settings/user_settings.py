from enum import Enum
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
    MINDS_CLOUD = "minds_cloud"

    @property
    def label(self) -> str:
        return PROVIDER_LABELS[self]

    @property
    def api_key_field(self) -> str:
        return _PROVIDER_KEY_FIELDS[self]


PROVIDER_LABELS: dict["Provider", str] = {
    Provider.ANTHROPIC: "Anthropic",
    Provider.OPENAI: "OpenAI",
    Provider.GEMINI: "Gemini",
    Provider.OPENAI_COMPATIBLE: "OpenAI-compatible",
    Provider.MINDS_CLOUD: "MindsHub",
}

_PROVIDER_KEY_FIELDS: dict["Provider", str] = {
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.OPENAI: "openai_api_key",
    Provider.GEMINI: "openai_api_key",
    Provider.OPENAI_COMPATIBLE: "openai_api_key",
    Provider.MINDS_CLOUD: "minds_api_key",
}

# Provider types as exposed to the UI (uses dashes, not underscores)
UI_PROVIDER_TYPES = ("minds-cloud", "anthropic", "openai", "gemini", "openai-compatible")

UI_PROVIDER_TYPE_LABELS: dict[str, str] = {
    "minds-cloud": "MindsHub",
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
    "minds-cloud": Provider.MINDS_CLOUD,
}


class _DynamicOptions:
    """Annotated metadata marker for fields whose valid options are resolved lazily from a callable."""

    def __init__(self, fn: Callable[[], list[str]]) -> None:
        self._fn = fn

    def get(self) -> list[str]:
        return self._fn()


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
    minds_api_key: SecretStr | None = Field(
        default=None,
        title="MindsHub API Key",
        description="API key for MindsHub. Required if using MindsHub as a provider.",
    )
    minds_url: str = Field(
        default="https://api.mindshub.ai/v1",
        title="MindsHub URL",
        description="Base URL for the MindsHub API.",
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

    def _provider_configured(self, provider: Provider) -> bool:
        # A provider is usable once it carries the credential it needs: an API
        # key for the hosted providers, or a base URL for an OpenAI-compatible
        # endpoint (key optional there). Mirrors the client's providerConfigured.
        if provider == Provider.OPENAI_COMPATIBLE:
            return bool(self.openai_base_url) or self.openai_api_key is not None
        return getattr(self, provider.api_key_field, None) is not None

    def _resolve_provider(self, preferred: Provider) -> Provider:
        # Keep the selected provider when it has credentials; otherwise fall
        # back to a configured one so a usable key still drives the agent even
        # when the selected provider was never set up (e.g. the default
        # MindsHub is selected but only an Anthropic key exists). Prefers
        # MindsHub, then any configured provider. Returns the original when
        # nothing is configured so the caller still surfaces the not-ready state.
        if self._provider_configured(preferred):
            return preferred
        if self._provider_configured(Provider.MINDS_CLOUD):
            return Provider.MINDS_CLOUD
        for p in (Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI, Provider.OPENAI_COMPATIBLE):
            if self._provider_configured(p):
                return p
        return preferred

    @property
    def effective_planning_provider(self) -> Provider:
        return self._resolve_provider(self.planning_provider)

    @property
    def effective_coding_provider(self) -> Provider:
        return self._resolve_provider(self.coding_provider)

    @property
    def config_status(self) -> dict[str, Any]:
        """Whether a usable planning provider is configured (after fallback)."""
        p = self.effective_planning_provider
        has_key = self._provider_configured(p)
        label = p.label
        # When we fell back, report the fallback provider's default model so the
        # status reflects what will actually run, not the stale selected model.
        model = (
            self.planning_model if p == self.planning_provider
            else PLANNING_MODEL_DEFAULTS.get(p.value, "")
        )
        return {
            "config_ready": has_key,
            "config_error": None if has_key else f"Configure {p.api_key_field} for {label}.",
            "provider": p.value,
            "provider_label": label,
            "model": model or "",
        }

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
