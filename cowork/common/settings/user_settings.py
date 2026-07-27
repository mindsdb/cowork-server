import json
from enum import Enum
from typing import Annotated, Any, Callable, get_args

from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator

from cowork.common.settings.app_settings import (
    CODING_MODEL_DEFAULTS,
    PLANNING_MODEL_DEFAULTS,
    ROUTER_MODEL_DEFAULTS,
    Settings,
    default_minds_url,
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
    Provider.MINDS_CLOUD: "MindsHub",
}

_PROVIDER_KEY_FIELDS: dict["Provider", str] = {
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.OPENAI: "openai_api_key",
    Provider.GEMINI: "gemini_api_key",
    Provider.OPENAI_COMPATIBLE: "openai_compatible_api_key",
    Provider.MINDS_CLOUD: "minds_api_key",
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


def _enabled_aware_default(
    provider_value: str,
    defaults: dict[str, str],
    enabled_map: dict[str, bool],
) -> str | None:
    """The provider's canonical default model, adjusted for availability.

    MindsHub marks a model the org's wallet can't currently pay for (or whose
    free allowance is exhausted) as ``enabled: false`` from ``/v1/models``, so
    blindly handing out the canonical default could be denied every turn. When
    the cached availability map (``minds_model_enabled``) marks the default as
    disabled, fall back to the first enabled model in the map — the map
    preserves the gateway's ``/v1/models`` ordering, which lists the
    free/baseline model first. Applies only to minds-cloud: direct (BYOK)
    providers have no such availability map.

    Deliberately conservative: an absent/empty map, a default missing from the
    map, or a map with nothing enabled all leave the canonical default
    untouched — degraded metadata must never change behavior.
    """
    default = defaults.get(provider_value)
    if provider_value != Provider.MINDS_CLOUD.value or not enabled_map:
        return default
    if default is None or enabled_map.get(default, True):
        return default
    for model_id, enabled in enabled_map.items():
        if enabled:
            return model_id
    return default


def _resolved_model(
    resolved_provider: "Provider",
    preferred_provider: "Provider",
    user_model: str | None,
    defaults: dict[str, str],
    enabled_map: dict[str, bool] | None = None,
) -> str | None:
    """Resolve a role's model given the readiness resolver's provider switch.

    The single load-bearing rule, shared by resolved_planning_model and
    resolved_coding_model so it can't drift between the two:

      - provider NOT switched → keep the user's chosen model.
      - provider switched → use the resolved provider's canonical default
        (availability-adjusted via _enabled_aware_default, so switching an
        account onto minds-cloud never lands on a locked model).
        NEVER fall back to the original provider's model — that would hand e.g.
        a Claude id to an openai-compatible / MindsHub endpoint (misrouting).
      - resolved provider has no canonical default (openai-compatible) → None,
        so config_status's model gate reports "select a model" rather than
        silently running a wrong model.
    """
    if resolved_provider == preferred_provider:
        return user_model
    return _enabled_aware_default(resolved_provider.value, defaults, enabled_map or {})

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
    from cowork.harnesses.base import available_harness_ids
    return available_harness_ids()


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
    gemini_api_key: SecretStr | None = Field(
        default=None,
        title="Gemini API Key",
        description="API key for Google Gemini (via its OpenAI-compatible endpoint). "
        "Falls back to the OpenAI key slot when unset.",
    )
    openai_compatible_api_key: SecretStr | None = Field(
        default=None,
        title="OpenAI-compatible API Key",
        description="API key for a custom OpenAI-compatible endpoint. "
        "Falls back to the OpenAI key slot when unset.",
    )
    minds_api_key: SecretStr | None = Field(
        default=None,
        title="MindsHub API Key",
        description="API key for MindsHub. Required if using MindsHub as a provider.",
    )
    minds_url: str = Field(
        default_factory=default_minds_url,
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
    # Router role: the cheap front-model that runs history summarization (and
    # later gates each turn, respond-vs-delegate). Selectable so a user can
    # point routing + summarization at a cheap model independently of the
    # coding (scratchpad) model. Falls back to the coding role in anton when
    # unset, so leaving these at defaults is behavior-preserving.
    router_provider: Provider = Field(
        default=Provider.ANTHROPIC,
        title="Routing & Summarization Provider",
        description="The provider for the routing/summarization model.",
    )
    router_model: str | None = Field(
        default=None,
        title="Routing & Summarization Model",
        description=(
            "The cheap model used for respond-vs-delegate routing and history "
            "summarization. Defaults to the recommended model for the selected "
            "provider (MindsHub → kimi; other providers → their smallest model)."
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
    nav_title: str = Field(
        default="",
        title="Nav Title",
        description="Sidebar title text. Empty uses the default, MindsHub.",
    )
    nav_title_color: str = Field(
        default="",
        title="Nav Title Color",
        description="Sidebar title color (hex). Empty follows the theme's default text color.",
    )
    nav_logo: str = Field(
        default="",
        title="Nav Logo",
        description="Sidebar logo image as a data URI. Empty shows no logo.",
    )
    show_theme_toggle: bool = Field(
        default=True,
        title="Show Theme Toggle",
        description="Show the floating light/dark theme toggle button.",
    )
    show_8bit_toggle: bool = Field(
        default=True,
        title="Show 8-Bit Toggle",
        description="Show the floating 8-bit style toggle button.",
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
    minds_model_enabled: str = Field(
        default="{}",
        title="MindsHub Model Availability",
        description=(
            "JSON-encoded map of MindsHub model id → enabled flag, cached from "
            "/v1/models whenever recommended-models fetches it live. Lets model "
            "defaults avoid locked models (wallet can't pay / free allowance "
            "spent) without a network call in the turn path."
        ),
    )

    # Memoized parse of `minds_model_enabled` (see `_minds_enabled_map`). Not a
    # settings field — never validated or serialized.
    _enabled_map_cache: dict[str, bool] | None = PrivateAttr(default=None)

    @field_validator("harness")
    @classmethod
    def validate_harness(cls, v: str) -> str:
        options = _harness_options()
        if v not in options:
            available = ", ".join(options) or "none"
            raise ValueError(f"Unknown harness '{v}'. Available: {available}")
        return v

    def _minds_enabled_map(self) -> dict[str, bool]:
        """The cached MindsHub model-availability map (id → enabled), or {}.

        Sourced from the ``minds_model_enabled`` setting, which the
        recommended-models endpoint refreshes from ``/v1/models`` on every
        settings load — so it tracks availability changes (e.g. adding credits
        re-enables a locked model on the next fetch) without any network call
        here.

        Parsed once per instance and memoized: this is called from
        ``apply_model_defaults`` and both ``resolved_*_model`` properties.
        """
        if self._enabled_map_cache is not None:
            return self._enabled_map_cache
        try:
            raw = json.loads(self.minds_model_enabled or "{}")
        except (ValueError, TypeError):
            raw = {}
        # Accept only real booleans. The map is written from real bools, but a
        # stringy value (corruption / a future writer) must not be misread —
        # ``bool("false")`` is True. A dropped entry is simply absent, which the
        # consumers already treat as "available", so this can't over-lock.
        result = (
            {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, bool)}
            if isinstance(raw, dict)
            else {}
        )
        self._enabled_map_cache = result
        return result

    @model_validator(mode='after')
    def apply_model_defaults(self) -> 'UserSettings':
        # Defaults are availability-aware for minds-cloud: when the canonical
        # default is locked (wallet can't pay / free allowance spent) it falls
        # back to the first enabled model instead of a guaranteed-denied
        # default. Only applies while the user hasn't picked a model (None) — an
        # explicit choice is never rewritten, and since nothing persists the
        # value assigned here, adding credits flips the default back to the
        # canonical model on the next settings load.
        enabled_map = self._minds_enabled_map()
        if self.planning_model is None:
            self.planning_model = _enabled_aware_default(
                self.planning_provider.value, PLANNING_MODEL_DEFAULTS, enabled_map
            )
        if self.coding_model is None:
            self.coding_model = _enabled_aware_default(
                self.coding_provider.value, CODING_MODEL_DEFAULTS, enabled_map
            )
        if self.router_model is None:
            self.router_model = _enabled_aware_default(
                self.coding_provider.value, ROUTER_MODEL_DEFAULTS, enabled_map
            )
        return self

    def _has_key(self, p: Provider) -> bool:
        # provider_api_key applies the gemini/openai-compatible → shared-openai
        # fallback, so a provider configured via EITHER its dedicated slot or the
        # legacy shared slot is correctly seen as keyed. (Raw getattr on the
        # dedicated slot would miss a gemini/oc user on the shared key.)
        return provider_api_key(self, p) is not None

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
        # Probe ALL providers (incl. gemini / openai-compatible, which have
        # dedicated key slots since the isolation change) — not just the legacy
        # minds/anthropic/openai trio — so a user who configured only a gemini
        # or openai-compatible key still resolves to a usable provider.
        for p in (
            Provider.MINDS_CLOUD,
            Provider.ANTHROPIC,
            Provider.OPENAI,
            Provider.GEMINI,
            Provider.OPENAI_COMPATIBLE,
        ):
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
        return _resolved_model(
            self.resolved_planning_provider,
            self.planning_provider,
            self.planning_model,
            PLANNING_MODEL_DEFAULTS,
            self._minds_enabled_map(),
        )

    @property
    def resolved_coding_model(self) -> str | None:
        return _resolved_model(
            self.resolved_coding_provider,
            self.coding_provider,
            self.coding_model,
            CODING_MODEL_DEFAULTS,
            self._minds_enabled_map(),
        )

    @property
    def resolved_router_provider(self) -> Provider:
        return self._resolve_provider(self.router_provider)

    @property
    def resolved_router_model(self) -> str | None:
        return _resolved_model(
            self.resolved_router_provider,
            self.router_provider,
            self.router_model,
            ROUTER_MODEL_DEFAULTS,
            self._minds_enabled_map(),
        )

    @property
    def config_status(self) -> dict[str, Any]:
        """Whether a usable provider is configured.

        Resolves the active planning provider to the first one that actually
        has a key (see ``_resolve_provider``) so this readiness signal matches
        what ``build_llm_client`` will actually run with — and reads the key via
        ``provider_api_key`` so gemini/openai-compatible still count when relying
        on the shared openai_api_key fallback."""
        p = self.resolved_planning_provider
        has_key = provider_api_key(self, p) is not None
        # Also require resolvable models. build_llm_client builds BOTH roles and
        # hands resolved_planning_model AND resolved_coding_model to the
        # providers; openai-compatible has no canonical default, so either role
        # can resolve to None and throw at runtime despite reading as "ready".
        # Gate on both so config_ready ⟹ the client can actually run.
        planning_model = self.resolved_planning_model
        coding_model = self.resolved_coding_model
        # openai-compatible needs a base URL. provider_base_url returns None for
        # an empty one (it must NOT silently fall back to api.openai.com — that
        # would leak the BYO key to OpenAI), so build_llm_client would hand the
        # key to the SDK's default host. Surface the misconfig instead. Checked
        # for whichever role actually resolves to openai-compatible.
        oc = Provider.OPENAI_COMPATIBLE
        needs_base = oc in (p, self.resolved_coding_provider)
        has_base = bool(self.openai_base_url) if needs_base else True
        label = p.label
        if not has_key:
            error = f"Configure an API key for {label}."
        elif not planning_model:
            error = f"Select a model for {label}."
        elif not coding_model:
            error = f"Select a coding model for {self.resolved_coding_provider.label}."
        elif not has_base:
            error = f"Set a base URL for {oc.label}."
        else:
            error = None
        return {
            "config_ready": (
                has_key and has_base and bool(planning_model) and bool(coding_model)
            ),
            "config_error": error,
            "provider": p.value,
            "provider_label": label,
            "model": planning_model or "",
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
