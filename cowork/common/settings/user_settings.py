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
        default="https://4nton.ai",
        title="Publish URL",
        description="URL for publishing artifacts.",
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

    def _has_key(self, p: Provider) -> bool:
        return getattr(self, p.api_key_field, None) is not None

    def _resolve_provider(self, preferred: Provider) -> Provider:
        """The provider actually usable for `preferred`: itself if its key is
        set, otherwise the first configured provider (managed MindsHub first).

        Mirrors the client's ``defaultModeProviderType`` so the readiness gate
        (``config_status``, surfaced at ``/health`` as ``config_ready`` — the
        signal the frontend's chat gate AND onboarding-vs-app routing read) and
        the agent's LLM client (``build_llm_client``) agree on what "configured"
        means — adding any key takes effect even if the stored
        ``planning_provider`` still points at a keyless provider. Returns
        ``preferred`` unchanged when nothing is configured."""
        if self._has_key(preferred):
            return preferred
        for p in (Provider.MINDS_CLOUD, Provider.ANTHROPIC, Provider.OPENAI):
            if self._has_key(p):
                return p
        return preferred

    @property
    def resolved_planning_provider(self) -> Provider:
        return self._resolve_provider(self.planning_provider)

    @property
    def resolved_coding_provider(self) -> Provider:
        return self._resolve_provider(self.coding_provider)

    @property
    def resolved_planning_model(self) -> str | None:
        p = self.resolved_planning_provider
        # Keep the user's chosen model when we didn't have to switch provider;
        # otherwise fall back to the resolved provider's default model so we
        # don't hand e.g. a Claude model id to the MindsHub gateway.
        return self.planning_model if p == self.planning_provider else PLANNING_MODEL_DEFAULTS.get(p.value)

    @property
    def resolved_coding_model(self) -> str | None:
        p = self.resolved_coding_provider
        return self.coding_model if p == self.coding_provider else CODING_MODEL_DEFAULTS.get(p.value)

    @property
    def config_status(self) -> dict[str, Any]:
        """Whether a usable provider is configured.

        Resolves the active planning provider to the first one that actually
        has a key (see ``_resolve_provider``) so this readiness signal matches
        what ``build_llm_client`` will actually run with."""
        p = self.resolved_planning_provider
        key_field = p.api_key_field
        has_key = getattr(self, key_field, None) is not None
        label = p.label
        return {
            "config_ready": has_key,
            "config_error": None if has_key else f"Configure {key_field} for {label}.",
            "provider": p.value,
            "provider_label": label,
            "model": self.resolved_planning_model or "",
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
