import json
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, get_args

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator

from cowork.common.settings.app_settings import (
    AGENT_ROLE_NAMES,
    CODING_MODEL_DEFAULTS,
    MINDS_FREE_MODEL,
    PLANNING_MODEL_DEFAULTS,
    ROLE_MODEL_DEFAULTS,
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


def _default_provider() -> "Provider":
    """The provider a fresh install starts on.

    Org mode runs MindsHub exclusively (BYOK in cloud is deferred, and the turn
    producer hands the pod a minted minds-cloud credential), so a tenant with
    nothing stored must not default to a BYOK provider it has no key for — that
    reads as "unconfigured" forever and sends every new org into onboarding.
    Desktop keeps its anthropic default.
    """
    return (
        Provider.MINDS_CLOUD
        if get_app_settings().tenancy_mode == "org"
        else Provider.ANTHROPIC
    )


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


def _defaults_with_declared(
    role: str, compiled: dict[str, str], declared: dict[str, str]
) -> dict[str, str]:
    """``compiled``, with the catalog's declared default over the minds-cloud slot.

    Only that slot moves. The direct providers are BYOK models that are not in
    MindsHub's catalog, so it has nothing to say about them and their defaults stay
    compiled in.

    The compiled value is not dead code underneath: it is what resolves before any
    ``/v1/models`` fetch has been persisted, which is every fresh install on its
    first message, and what resolves when the catalog is unreachable. Demoted, not
    retired.
    """
    if role not in declared:
        return compiled
    return {**compiled, Provider.MINDS_CLOUD.value: declared[role]}


def minds_role_start_models(
    *, declared: dict[str, str], enabled_map: dict[str, bool]
) -> list[str]:
    """The model each minds-cloud role starts on, in ``AGENT_ROLE_ORDER``.

    Ordered by ``ROLE_MODEL_DEFAULTS``, which is built in that order, so the list
    lines up with the ``recommendedPair`` slots without anyone counting them.

    What the picker is served as ``recommendedPair``, and it goes through the same
    two steps resolution does: the declared default replaces the compiled one, then
    availability adjusts it. Both steps matter to the picker, because the desktop
    writes these values back as explicit model pins when a save repoints a role
    onto MindsHub. Served unadjusted, it would show, and then pin, a model the
    wallet cannot pay for while turns ran something else.

    Calls ``_enabled_aware_default`` directly rather than going through
    ``_resolved_model(..., wallet_aware=True)`` the way the three
    ``resolved_*_model`` properties do. The two agree here because a default is
    only ever read where nothing is stored, and ``wallet_aware`` decides what
    happens to a STORED id the availability map does not affirm. There is no
    stored id on this path, so the extra step has nothing to act on.
    """
    return [
        _enabled_aware_default(
            Provider.MINDS_CLOUD.value,
            _defaults_with_declared(role, compiled, declared),
            enabled_map,
        )
        or ""
        for role, compiled in ROLE_MODEL_DEFAULTS.items()
    ]

def _first_available_model(enabled_map: dict[str, bool]) -> str | None:
    """First enabled model in map order, else the first served one, else None.

    The map lists the full ``/v1/models`` catalogue in the gateway's ranking, so
    its first enabled entry is the affordable baseline; when nothing is enabled the
    first key is still a served (locked) model — recoverable via top-up, unlike a
    retired id that 404s every turn. Empty map → None.
    """
    if not enabled_map:
        return None
    return next(
        (m for m, en in enabled_map.items() if en),
        next(iter(enabled_map)),
    )


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
    disabled — OR simply doesn't list it, meaning gone (renamed/retired), not
    degraded data — fall back to the first enabled model in the map. That map
    preserves the gateway's ``/v1/models`` ordering (a remote order we neither
    control nor pin), so this is the first model MindsHub itself ranks that the
    wallet can actually pay for: a free-tier wallet, with the paid aliases ahead
    of it marked disabled, lands on the free/baseline model. An empty/absent map
    (no tier data at all) leaves the default untouched. Applies only to
    minds-cloud: direct (BYOK) providers have no such availability map.

    Org mode requires the same positive evidence, but falls back to
    MINDS_FREE_MODEL instead, so a credit-less org isn't charged.
    """
    default = defaults.get(provider_value)
    if provider_value != Provider.MINDS_CLOUD.value:
        return default
    if get_app_settings().tenancy_mode == "org":
        if default is not None and enabled_map.get(default) is True:
            return default
        # Free model, unless a known catalogue no longer serves it — then it's
        # retired and would 404 every turn, so hand out a served model.
        if enabled_map and MINDS_FREE_MODEL not in enabled_map:
            served = _first_available_model(enabled_map)
            if served is not None:
                return served
        return MINDS_FREE_MODEL
    if not enabled_map:
        return default
    if default is None or enabled_map.get(default) is True:
        return default
    for model_id, enabled in enabled_map.items():
        if enabled:
            return model_id
    # Nothing affordable. A still-served (locked) default is kept — it 402s with a
    # top-up path. A default absent from this non-empty catalogue is retired and
    # would 404 every turn, so fall back to a served model instead.
    if default is not None and default not in enabled_map:
        served = _first_available_model(enabled_map)
        if served is not None:
            return served
    return default


def _resolved_model(
    resolved_provider: "Provider",
    preferred_provider: "Provider",
    user_model: str | None,
    defaults: dict[str, str],
    enabled_map: dict[str, bool] | None = None,
    *,
    wallet_aware: bool = False,
) -> str | None:
    """Resolve a role's model given the readiness resolver's provider switch.

    The single load-bearing rule, shared by every resolved_*_model property so
    it can't drift between the roles:

      - provider NOT switched → keep the user's chosen model (but see
        ``wallet_aware`` below).
      - provider switched → use the resolved provider's canonical default
        (availability-adjusted via _enabled_aware_default, so switching an
        account onto minds-cloud never lands on a locked model).
        NEVER fall back to the original provider's model — that would hand e.g.
        a Claude id to an openai-compatible / MindsHub endpoint (misrouting).
      - resolved provider has no canonical default (openai-compatible) → None,
        so config_status's model gate reports "select a model" rather than
        silently running a wrong model.

    ``wallet_aware`` (ENG-1632, now all three roles). A stored minds-cloud
    model the availability map marks ``enabled: false`` is guaranteed to be
    denied on every call — either because the wallet can't pay / allowance is
    spent, OR because the id isn't a MindsHub model at all (absent from a
    non-empty map counts as unavailable; see the probe note below). When a
    strictly-enabled model exists in the map, resolve to it instead of the
    doomed stored value; when nothing is enabled (fully drained account) or
    the map is absent/degraded, keep the stored value — degraded metadata
    must never change behavior. The stored row is never rewritten, so a
    topped-up wallet (``enabled: true`` on the next settings load) restores
    the stored model automatically.

    Originally auxiliary-roles-only (coding, router): they're invisible in
    default mode, so a stuck pin 402s every turn with no way for the user to
    see or fix it, surfacing as a spurious "internal error". Planning was
    deliberately exempted at the time — "an explicit choice is never
    rewritten" holds because planning is visible in the picker and has the
    pick-it-and-see-"Needs credits" lane (ENG-1248) — but that reasoning
    assumed the stored id was chosen for the CURRENT provider. MindsHub SSO
    sign-in breaks that assumption: it rewrites planning_provider to
    minds_cloud without touching the paired planning_model, so a pre-sign-in
    BYOK pick (e.g. an Anthropic id) survives untouched and 404s the gateway
    on every turn — never reaching the picker's "Needs credits" lane at all,
    since that lane is for a real MindsHub model the wallet can't currently
    afford, not a foreign id. Planning now gets the same fallback; the
    "Needs credits" lane is unaffected — it's a separate, still-visible
    per-model tag (``modelEnabled``) on the picker, not this resolution path.
    """
    if resolved_provider == preferred_provider and user_model:
        enabled = enabled_map or {}
        if wallet_aware and resolved_provider == Provider.MINDS_CLOUD and enabled:
            # Map keys are bare ids (/v1/models never emits the retired
            # "latest:" prefix), but login-era pins still carry it — strip it
            # for the probe or those pins silently escape the fallback.
            #
            # ABSENT from a non-empty map counts as unavailable, unlike
            # _enabled_aware_default's absent-means-available rule. The two
            # look inconsistent but probe different things: that rule probes
            # OUR canonical default (a guaranteed-served id — absence there
            # means an older gateway), this one probes a USER-STORED id that
            # can be anything (the drpconcepcion cohort stored a Gemini id
            # against minds-cloud and 404'd every aux call — an id the map
            # can never mark false because the gateway doesn't serve it).
            # The map is written from the full /v1/models catalogue, so
            # absence genuinely means "not served"; the non-empty guard keeps
            # the degraded-metadata rule intact.
            bare = user_model.removeprefix("latest:")
            if not enabled.get(bare, False):
                # First enabled entry in map order. The real guarantee here is
                # NOT "the gateway lists the free model first" (it doesn't, and
                # the ranking changes per deployment): it is that enablement
                # tracks affordability, so on a locked wallet only free-bucket
                # models are enabled and the first enabled entry is affordable
                # by construction. Embedding rows never
                # reach this map — filtered at construction in
                # fetch_minds_models (_is_embedding_row).
                fallback = next((mid for mid, en in enabled.items() if en), None)
                if fallback:
                    return fallback
                # Nothing affordable. A still-served (locked) pin is kept below. But
                # an id absent from this non-empty catalogue is retired/foreign and
                # 404s every turn with no recovery, so hand out a served model
                # instead of a dead id.
                if bare not in enabled:
                    return next(iter(enabled))
        return user_model
    # No explicit choice (or provider switched): the resolved provider's default,
    # availability-adjusted. openai-compatible has no default -> None, so the
    # "select a model" gate still fires for it.
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


class _OrgScoped:
    """Annotated marker: this setting is org-level config (admin-owned) rather
    than a personal preference. Untagged fields are per-user; SecretStr
    credentials are org regardless. Read by setting_is_org_scoped."""


ORG = _OrgScoped()


def _harness_options() -> list[str]:
    from cowork.harnesses.base import available_harness_ids
    return available_harness_ids()


def _coding_engine_options() -> list[str]:
    # Imported lazily so normal Cowork settings startup does not import or
    # launch a coding runtime. The registry only reports functional adapters;
    # future engines become additive here without UI conditionals.
    from cowork.coding.engines.registry import engine_registry

    return engine_registry.ids()


# ── .env ↔ DB setting aliases ────────────────────────────────────────
#
# One canonical map of DB setting key → its ANTON_* .env variable name, for
# every field that overlaps between AntonSettings (.env, read by the standalone
# ``anton`` CLI) and UserSettings (DB). This is the single source the .env→DB
# seed (``migrations``), the ``POST /settings/raw`` merge sync, and the client
# all derive from. The same map was previously hand-maintained in both
# ``migrations._ENV_TO_SETTING`` and the client's ``syncSettings.ts`` — the two
# had already drifted (the client copy silently omitted gemini, the OpenAI-
# compatible key, router_provider, memory_enabled, act_first, proactive_
# dashboards and publish_url), which is exactly the class of bug ENG-941/ENG-1125
# retires.
#
# planning_model / coding_model are DELIBERATELY absent (ENG-739): a model in
# .env is CLI-only and must never ride a bulk .env→DB sync, or a login /
# token-refresh would re-pin a picker choice from a stale ``latest:`` line.
#
# max_tool_rounds / max_continuations / max_turn_tokens are DELIBERATELY absent
# too, for the
# ENG-739 reason plus a harder failure mode: anton's own CoreSettings accepts
# any int, so a stale anton-CLI line like ANTON_MAX_TOOL_ROUNDS=1000 in the
# shared ~/.cowork/.env is valid for the CLI but fails UserSettings' bounds —
# and because sync_env_vars_to_db validates every mapped key and raises on the
# first failure, one such line would 400 every credential push / token
# refresh. Budgets enter the DB only via explicit writes (Settings UI / API).
#
# General inclusion rule: a key belongs here only if (a) every value anton's
# own settings accept for it is also valid for UserSettings, and (b)
# re-syncing a stale .env line can never override a choice the user made in
# the product. When in doubt, leave it out — .env lines still work for the
# standalone anton CLI.
#: Lowest non-zero per-turn spend ceiling a user may set (ENG-1286). Mirrored by
#: `BUDGET_FIELDS.maxTurnTokens.min` in cowork's `settingsTransform.js`; the two
#: are asserted equal in `tests/test_agent_budget_settings.py`.
TURN_CEILING_FLOOR = 750_000

SETTING_ENV_ALIASES: dict[str, str] = {
    "anthropic_api_key": "ANTON_ANTHROPIC_API_KEY",
    "openai_api_key": "ANTON_OPENAI_API_KEY",
    "openai_compatible_api_key": "ANTON_OPENAI_API_KEY_CUSTOM",
    "gemini_api_key": "ANTON_GEMINI_API_KEY",
    "minds_api_key": "ANTON_MINDS_API_KEY",
    "planning_provider": "ANTON_PLANNING_PROVIDER",
    "coding_provider": "ANTON_CODING_PROVIDER",
    "router_provider": "ANTON_ROUTER_PROVIDER",
    "minds_url": "ANTON_MINDS_URL",
    "openai_base_url": "ANTON_OPENAI_BASE_URL",
    "memory_enabled": "ANTON_MEMORY_ENABLED",
    "memory_mode": "ANTON_MEMORY_MODE",
    "episodic_memory": "ANTON_EPISODIC_MEMORY",
    "proactive_dashboards": "ANTON_PROACTIVE_DASHBOARDS",
    "act_first": "ANTON_ACT_FIRST",
    "publish_url": "ANTON_PUBLISH_URL",
    "artifact_autopublish_enabled": "ANTON_ARTIFACT_AUTOPUBLISH",
}

# Inverse view (ANTON_* .env var → DB setting key) for .env-first callers, i.e.
# the first-boot migration seed and the ``POST /settings/raw`` merge sync.
ENV_ALIAS_TO_SETTING: dict[str, str] = {v: k for k, v in SETTING_ENV_ALIASES.items()}


def normalize_provider_value(val: str, *, minds_key_present: bool) -> str:
    """Translate a .env / UI provider string to the DB ``Provider`` enum value.

    The single home for the hyphen→underscore canonicalization plus the
    "a Minds key is present, so ``openai-compatible`` really means
    ``minds_cloud``" heuristic. This was reimplemented in
    ``migrations._normalize_provider_value``, the client's ``syncSettings.ts``,
    and ``settingsTransform.js`` — three copies that could disagree, silently
    routing a provider to the wrong client (the inverse direction, DB→UI, is
    ``Provider.ui_value``).
    """
    canonical = val.replace("-", "_")
    if canonical == Provider.OPENAI_COMPATIBLE.value and minds_key_present:
        return Provider.MINDS_CLOUD.value
    return canonical


class UserSettings(Settings):
    # The recommended-model catalog and per-provider model defaults are
    # global, application-level config and live in app_settings
    # (RECOMMENDED_MODELS / RECOMMENDED_PAIR / *_MODEL_DEFAULTS).

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Only local reads process env; a shared server's injected provider
        # secrets would otherwise leak into every tenant's settings. Fail closed:
        # anything not local is DB-only.
        if get_app_settings().tenancy_mode == "local":
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)
        return (init_settings,)

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
    minds_url: Annotated[str, ORG] = Field(
        default_factory=default_minds_url,
        title="MindsHub URL",
        description="Base URL for the MindsHub API.",
    )
    planning_provider: Annotated[Provider, ORG] = Field(
        default_factory=_default_provider,
        title="Planning Provider",
        description="The provider to use for the reasoning/planning model.",
    )
    planning_model: str | None = Field(
        default=None,
        title="Planning Model",
        description="The reasoning model used for planning. Defaults to the recommended model for the selected provider.",
    )
    coding_provider: Annotated[Provider, ORG] = Field(
        default_factory=_default_provider,
        title="Coding Provider",
        description="The provider to use for the coding model.",
    )
    coding_model: str | None = Field(
        default=None,
        title="Coding Model",
        description="The coding model. Defaults to the recommended model for the selected provider.",
    )
    coding_agent_engine: Annotated[str, _DynamicOptions(_coding_engine_options)] = Field(
        default="codex",
        title="Coding Agent",
        description="The agent engine used by the separate Code workspace.",
    )
    coding_agent_model: str = Field(
        default="gpt",
        min_length=1,
        max_length=256,
        title="Coding Agent Model",
        description="The MindsHub Inference model used by the selected coding agent.",
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
    router_provider: Annotated[Provider, ORG] = Field(
        default_factory=_default_provider,
        title="Routing & Summarization Provider",
        description="The provider for the routing/summarization model.",
    )
    router_model: str | None = Field(
        default=None,
        title="Routing & Summarization Model",
        description=(
            "The cheap model used for history summarization. The pre-Anton route "
            "gate does NOT run on it — it takes the role's fast default for the "
            "provider (see resolved_gate_model), except on openai-compatible, "
            "where this is the only model there is. Defaults to the recommended "
            "model for the selected provider (MindsHub → MindsHub Air; other "
            "providers → their smallest model)."
        ),
    )
    harness: Annotated[str, _DynamicOptions(_harness_options), ORG] = Field(
        default="anton",
        title="Harness",
        description="The AI harness used to generate responses.",
    )
    channels_harness: Annotated[str, _DynamicOptions(_harness_options), ORG] = Field(
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
    show_coding_mode_toggle: bool = Field(
        default=True,
        title="Show Coding Mode Toggle",
        description="Show the floating coding-mode toggle button.",
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
    # `hub_workspace_id`, not `workspace_id`: in this repo `workspace` already
    # means a filesystem location (the per-conversation private directory, the
    # project tree, the paths on AntonSettings), and that meaning is load-bearing
    # in dozens of places. This field holds a MindsHub Workspace uuid, an
    # org-internal container that owns hub resources and lives in the auth
    # service. The two concepts share nothing but a word.
    #
    # Untagged, so it writes per-user: `SettingService._new_row` files an
    # untagged key at (org, user) scope in org mode and in the single global row
    # on a desktop install, which is the right answer for both. It carries no
    # ORG marker deliberately: which workspace a person is looking at is their
    # own preference, not org configuration an admin sets for everyone.
    hub_workspace_id: str = Field(
        default="",
        title="Active MindsHub Workspace",
        description=(
            "Which MindsHub workspace this person is working in, as a uuid. "
            "Empty means no pick has been made, and readers fall back to the "
            "organization's default workspace. Interim storage: the shared "
            "per-user preference the console reads has no route in auth yet, and "
            "this field is what a follow-up migrates onto it."
        ),
    )
    coding_mode_enabled: bool = Field(
        default=False,
        title="Enable Coding Mode",
        description=(
            "Show a per-task harness/model picker and allow launching tasks in "
            "an external coding CLI (e.g. Claude Code) instead of the in-app chat."
        ),
    )
    # Which harnesses appear as options in the per-task harness picker
    # (Coding Mode's composer pill). All default true — an account that never
    # visits this setting sees every harness it's otherwise eligible for.
    # Anton has no enable flag: it's the default agent and always offered —
    # a picker with every harness disabled would have nothing to run.
    # `claude-code` isn't a cowork.harnesses.base-registered harness (it runs
    # the `claude` CLI entirely client-side in the Electron app, never
    # through this server), so it has no `available_harness_ids()` entry to
    # validate against — this flag is the only server-side notion of it.
    harness_hermes_enabled: bool = Field(
        default=True,
        title="Enable Hermes in the Harness Picker",
        description="Offer Hermes as a per-task harness choice in Coding Mode.",
    )
    harness_claude_code_enabled: bool = Field(
        default=True,
        title="Enable Claude Code in the Harness Picker",
        description="Offer Claude Code as a per-task harness choice in Coding Mode.",
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
    history_compaction_enabled: bool = Field(
        default=True,
        title="History Compaction",
        description="Replay anton's compacted summary + recent tail instead of full history each turn.",
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
    # ── Advanced agent budgets ──
    # Overlaid onto AntonSettings per conversation (anton_harness.harness);
    # anton's own CLI defaults (25/3) are lower — Cowork deliberately runs
    # with more headroom so long tasks finish without a mid-task check-in.
    # Defaults come from app settings so hosted deployments (operator-paid
    # inference, no Settings UI on web) can lower them via env without
    # touching per-user rows.
    max_tool_rounds: Annotated[int, ORG] = Field(
        default_factory=lambda: get_app_settings().default_max_tool_rounds,
        ge=5,
        le=500,
        title="Max Steps per Task",
        description=(
            "How many actions (running code, reading files, searching) the agent "
            "may take on one request before it pauses and checks in with you. "
            "Raise it so big tasks can finish in one go; lower it to keep a "
            "tighter leash on time and cost. Applies to the Anton agent and, for "
            "Cowork sessions, replaces the ANTON_MAX_TOOL_ROUNDS environment "
            "variable."
        ),
    )
    max_continuations: Annotated[int, ORG] = Field(
        default_factory=lambda: get_app_settings().default_max_continuations,
        ge=0,
        le=25,
        title="Max Auto-Continues",
        description=(
            "When the agent stops but its work looks unfinished, Cowork can send "
            "it back to complete the job — this caps how many times. Raise it "
            "for hands-off thoroughness; set 0 to stop after the first attempt "
            "(you'll still get a summary of what's missing). Applies to the "
            "Anton agent and, for Cowork sessions, replaces the "
            "ANTON_MAX_CONTINUATIONS environment variable."
        ),
    )
    max_turn_tokens: Annotated[int, ORG] = Field(
        default_factory=lambda: get_app_settings().default_max_turn_tokens,
        # Plain contiguous range. "No limit" in the UI is the TOP of it
        # (50_000_000), not a sentinel.
        #
        # "No limit" is EFFECTIVELY, not literally, true — and the bound is
        # closer than it looks. A turn makes about
        # `max_tool_rounds x (max_continuations + 1)` LLM calls, which at THIS
        # repo's defaults (50 x 6) is ~306 calls, so 50M is reached at ~163k per
        # call — below the ~190k context a long conversation carries. At the
        # maxima a user can set (500 x 25) it is ~13,000 calls and ~3.8k per
        # call. So the step cap does NOT always land first; it merely always has
        # so far. The largest turn in 30 days of production was 8.26M, because
        # real turns end and compaction intervenes long before that shape.
        # Do not restate this as "the ceiling can never fire at max" — an
        # earlier version of this comment did, using anton's own 25x3 defaults
        # rather than Cowork's, which put the threshold at 480k per call and
        # made it look unreachable.
        #
        # A 0-means-unlimited sentinel was built
        # and then removed — it needed a hole in the range, a validator to guard
        # the hole, and a special case in the client clamp, all to solve
        # discoverability that the checkbox solves on its own. It also collided
        # with `max_continuations` next door, where 0 means literally zero.
        #
        # The floor is 750_000, not a rounder-looking 100_000. A turn's first
        # LLM call costs roughly the conversation's context — ~190k on a long
        # one — so a ceiling smaller than a couple of calls stops the turn
        # before it has done anything. Measured against anton: 100_000
        # dispatched ZERO tools and still spent 400_000. anton now guarantees at
        # least one tool round regardless, so this floor is a usability bound
        # rather than a safety one: it is the lowest value where a 190k-context
        # turn still gets several rounds, and it sits just above the p75
        # external turn (736k), so "the minimum" means "cut me off around the
        # 75th percentile".
        ge=TURN_CEILING_FLOOR,
        le=50_000_000,
        title="Max Tokens per Task",
        description=(
            "The most tokens the agent may spend on one request before it "
            "pauses and checks in with you. Tokens are the unit your plan's "
            "monthly allowance is measured in — including tokens re-read from "
            "cache — so a task that gets stuck can burn a large share of the "
            "month without finishing. Raise it if you routinely give the agent "
            "big jobs; lower it to cap what any single request can cost. "
            "Applies to the Anton agent and, for Cowork sessions, replaces the "
            "ANTON_MAX_TURN_TOKENS environment variable."
        ),
    )
    publish_url: Annotated[str, ORG] = Field(
        default="",
        title="Publish URL",
        description="Base URL for publishing artifacts. When empty, derived from the MindsHub endpoint (api[.env].mindshub.ai → view[.env].mindshub.ai, else prod); set explicitly to override.",
    )
    artifact_autopublish_enabled: Annotated[bool, ORG] = Field(
        # On by default, deliberately. It was drafted as off because publishing a
        # fullstack artifact resolved datasource secrets from a process-global
        # vault, so one org's artifact could ship another's credentials. That is
        # closed: the vault is org-keyed now (`vault_for_scope` ->
        # `scoped_storage_root(..., store="data-vault")`), and a missing scope
        # RAISES on an org deployment rather than falling back to the shared root,
        # so the leak cannot reappear silently.
        #
        # Inert outside org mode: `autopublish_project_artifacts` returns on the
        # scope guard before it ever reads this, so a desktop install is
        # unaffected by the default either way.
        default=True,
        title="Auto-publish artifacts",
        description=(
            "In org deployments, publish every artifact the agent creates or changes "
            "to the viewer automatically, restricted to the author's organization. "
            "On by default. ORG-scoped, so an organization can still turn it off "
            "for itself. It uploads user content without an explicit user action, "
            "which is the reason to keep the per-org switch. The env override that "
            "actually "
            "bypasses that is ARTIFACT_AUTOPUBLISH_ENABLED, the field name itself: "
            "UserSettings is a pydantic-settings model, so any field with no DB row "
            "falls back to the environment, process-global, for every tenant of the "
            "deployment. (The ANTON_ARTIFACT_AUTOPUBLISH alias below is NOT that "
            "override — SETTING_ENV_ALIASES only drives the .env→DB seed, which org "
            "deployments skip entirely; see dev_setup._migrate_env_to_db_if_local.)"
        ),
    )
    openai_base_url: Annotated[str, ORG] = Field(
        default="",
        title="OpenAI Base URL",
        description="Base URL for OpenAI-compatible providers.",
    )
    model_mode: Annotated[str, ORG] = Field(
        default="default",
        title="Model Mode",
        description="Whether to use default or custom model assignments (default or custom).",
    )
    model_overrides: Annotated[str, ORG] = Field(
        default="{}",
        title="Model Overrides",
        description="JSON-encoded per-role provider/model overrides when model_mode is custom.",
    )
    providers_json: Annotated[str, ORG] = Field(
        default="[]",
        title="Providers",
        description="JSON-encoded list of configured provider entries for the settings UI.",
    )
    provider_status: Annotated[str, ORG] = Field(
        default="{}",
        title="Provider Status",
        description="JSON-encoded map of provider type → last connectivity-test status (ok|fail). Persisted so the Settings dots survive a reload.",
    )
    provider_status_details: Annotated[str, ORG] = Field(
        default="{}",
        title="Provider Status Details",
        description="JSON-encoded map of provider type → last connectivity-test detail (e.g. an HTTP code).",
    )
    minds_model_enabled: Annotated[str, ORG] = Field(
        default="{}",
        title="MindsHub Model Availability",
        description=(
            "JSON-encoded map of MindsHub model id → enabled flag, cached from "
            "/v1/models whenever recommended-models fetches it live. Lets model "
            "defaults avoid locked models (wallet can't pay / free allowance "
            "spent) without a network call in the turn path."
        ),
    )

    minds_role_defaults: Annotated[str, ORG] = Field(
        default="{}",
        title="MindsHub Role Defaults",
        description=(
            "JSON-encoded map of agent role -> the model id MindsHub's catalog "
            "declares as that role's default, cached from /v1/models whenever "
            "recommended-models fetches it live. Lets the default a new user "
            "starts on move by config, without a client release and without a "
            "network call in the turn path."
        ),
    )

    # Memoized parse of `minds_model_enabled` (see `_minds_enabled_map`). Not a
    # settings field — never validated or serialized.
    _enabled_map_cache: dict[str, bool] | None = PrivateAttr(default=None)
    # Same, for `minds_role_defaults` (see `_minds_role_default_map`).
    _role_default_cache: dict[str, str] | None = PrivateAttr(default=None)

    @field_validator("harness")
    @classmethod
    def validate_harness(cls, v: str) -> str:
        options = _harness_options()
        if v not in options:
            available = ", ".join(options) or "none"
            raise ValueError(f"Unknown harness '{v}'. Available: {available}")
        return v

    @field_validator("coding_agent_engine")
    @classmethod
    def validate_coding_agent_engine(cls, value: str) -> str:
        options = _coding_engine_options()
        if value not in options:
            available = ", ".join(options) or "none"
            raise ValueError(f"Unknown coding agent '{value}'. Available: {available}")
        return value

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

    def _minds_role_default_map(self) -> dict[str, str]:
        """The cached agent-role -> model-id map MindsHub's catalog declares, or {}.

        Sourced from the ``minds_role_defaults`` setting, which the
        recommended-models endpoint refreshes from ``/v1/models`` on every
        settings load, so moving a default in the catalog reaches this install on
        its next settings load with no client release.

        Parsed once per instance and memoized, for the same reason
        ``_minds_enabled_map`` is: this is read once per role by
        ``apply_model_defaults`` and again by each ``resolved_*_model``.
        """
        if self._role_default_cache is not None:
            return self._role_default_cache
        try:
            raw = json.loads(self.minds_role_defaults or "{}")
        except (ValueError, TypeError):
            raw = {}
        # Both halves must be real non-empty strings. A role is looked up by name
        # and its value is handed to the gateway as a model id, so a stringified
        # null or a number here would resolve a role onto a model that cannot
        # exist, and the compiled fallback below is strictly better than that.
        result = (
            {
                k: v
                for k, v in raw.items()
                if isinstance(k, str) and isinstance(v, str) and k in AGENT_ROLE_NAMES and v
            }
            if isinstance(raw, dict)
            else {}
        )
        self._role_default_cache = result
        return result

    def _defaults_for_role(self, role: str, compiled: dict[str, str]) -> dict[str, str]:
        """``compiled``, with the catalog's declared default over the minds-cloud slot.

        The one place the remote declaration is preferred over the compiled table,
        so the six resolution sites cannot drift on which wins.
        """
        return _defaults_with_declared(role, compiled, self._minds_role_default_map())

    @model_validator(mode='after')
    def apply_model_defaults(self) -> 'UserSettings':
        # Defaults are availability-aware for minds-cloud: when the canonical
        # default is locked (wallet can't pay / free allowance spent) it falls
        # back to the first enabled model instead of a guaranteed-denied
        # default. Only applies while the user hasn't picked a model (None) — an
        # explicit choice is never rewritten, and since nothing persists the
        # value assigned here, adding credits flips the default back to the
        # canonical model on the next settings load.
        #
        # NOT collapsed into the wallet-aware branch of _resolved_model
        # (ENG-1632), although the two apply the same helper: this validator
        # MUTATES the stored fields at construction time, and downstream
        # consumers depend on that pre-fill — e.g. build_llm_client's effort
        # guard compares the stored model to the resolved one, and a user with
        # no coding_model row only keeps their reasoning effort because this
        # fill makes the two equal. Removing the fill here would silently
        # strip effort for every no-row user (pinned by
        # test_effort_survives_when_no_model_row_is_stored).
        #
        # Known wart, deliberately preserved as-is: the ROUTER default below
        # derives from coding_provider while resolved_router_model resolves
        # against router_provider — split-provider configs disagree between
        # the two. Tracked on ENG-1632 as a follow-up; changing it here would
        # alter resolution for existing split configs.
        enabled_map = self._minds_enabled_map()
        if self.planning_model is None:
            self.planning_model = _enabled_aware_default(
                self.planning_provider.value,
                self._defaults_for_role("planning", PLANNING_MODEL_DEFAULTS),
                enabled_map,
            )
        if self.coding_model is None:
            self.coding_model = _enabled_aware_default(
                self.coding_provider.value,
                self._defaults_for_role("coding", CODING_MODEL_DEFAULTS),
                enabled_map,
            )
        if self.router_model is None:
            self.router_model = _enabled_aware_default(
                self.coding_provider.value,
                self._defaults_for_role("router", ROUTER_MODEL_DEFAULTS),
                enabled_map,
            )
        return self

    def _has_key(self, p: Provider) -> bool:
        # Org mode mints a per-turn key, so minds-cloud is usable with nothing
        # stored; else a fresh org resolves to the pod's ambient ANTHROPIC key.
        if p is Provider.MINDS_CLOUD and get_app_settings().tenancy_mode == "org":
            return True
        # An openai-compatible endpoint's credential is its base URL — a local
        # model server usually wants no key at all, and the Settings UI lets one
        # be saved that way. Judged keyless, the resolver below walks straight
        # past it to the hosted gateway, which is the one place these prompts
        # must not go.
        if p is Provider.OPENAI_COMPATIBLE and (self.openai_base_url or "").strip():
            return True
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
        # wallet_aware (ENG-1632 follow-up): the docstring on _resolved_model
        # originally kept planning exempt from this because a bad pick is
        # "visible in the picker" — the user can see it and fix it. That
        # holds for a genuinely unaffordable MindsHub model (the ENG-1248
        # "Needs credits" lane still surfaces those separately, unaffected by
        # this flag). It does NOT hold for a model that isn't a MindsHub
        # model at all: MindsHub SSO sign-in rewrites planning_provider to
        # minds_cloud without touching the paired planning_model, so a BYOK
        # pick from before sign-in (e.g. an Anthropic id) survives untouched
        # and gets sent straight to the gateway on the next turn, which
        # 404s ("That model isn't available") on every single message
        # instead of surfacing anywhere the user would see it first. Falling
        # back to an actually-available MindsHub model — same self-healing
        # coding/router already get — beats a hard failure on every turn.
        return _resolved_model(
            self.resolved_planning_provider,
            self.planning_provider,
            self.planning_model,
            self._defaults_for_role("planning", PLANNING_MODEL_DEFAULTS),
            self._minds_enabled_map(),
            wallet_aware=True,
        )

    @property
    def resolved_coding_model(self) -> str | None:
        # wallet_aware: the coding role (completion verifier, scratchpad) is
        # invisible in default mode — a wallet-locked pin here 402s every turn
        # with no way for the user to see or fix it (ENG-1632).
        return _resolved_model(
            self.resolved_coding_provider,
            self.coding_provider,
            self.coding_model,
            self._defaults_for_role("coding", CODING_MODEL_DEFAULTS),
            self._minds_enabled_map(),
            wallet_aware=True,
        )

    @property
    def resolved_router_provider(self) -> Provider:
        return self._resolve_provider(self.router_provider)

    @property
    def resolved_router_model(self) -> str | None:
        # wallet_aware: same rationale as resolved_coding_model — the router
        # role (history summarization; the route gate resolves its own model,
        # see resolved_gate_model) is invisible in default mode (ENG-1632).
        return _resolved_model(
            self.resolved_router_provider,
            self.router_provider,
            self.router_model,
            self._defaults_for_role("router", ROUTER_MODEL_DEFAULTS),
            self._minds_enabled_map(),
            wallet_aware=True,
        )

    @property
    def resolved_gate_model(self) -> str | None:
        """The model the pre-Anton route gate runs on (ENG-1851).

        Deliberately NOT the user's ``router_model``. The gate sits ahead of
        every turn with a budget measured in seconds, so it has one requirement
        the router role's other consumer (history summarization) does not:
        speed. A user who picks a large model for "routing and summarization"
        is choosing it for the summaries; applied to the gate it times out on
        every turn and bills for the attempt. So the gate takes the role's
        *default* for the resolved provider — the catalog's declared default
        for minds-cloud, the compiled one for the direct providers — which are
        all chosen to be fast, availability-adjusted like every other default.

        openai-compatible is the exception: it has no canonical model (it is a
        BYO endpoint), so the user's own pick is the only model that exists
        there and the gate uses it. The Settings UI carries the speed note.
        """
        provider = self.resolved_router_provider
        if provider is Provider.OPENAI_COMPATIBLE:
            return self.resolved_router_model
        return _enabled_aware_default(
            provider.value,
            self._defaults_for_role("router", ROUTER_MODEL_DEFAULTS),
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
        # _has_key treats minds-cloud as keyed in org mode (per-turn mint).
        has_key = self._has_key(p)
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
        # A bare `SecretStr` has no type args; `SecretStr | None` carries it in
        # the union args. Handle both so a future non-optional credential is
        # still encrypted + classified as org, not stored as plaintext.
        return annotation is SecretStr or SecretStr in get_args(annotation)

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


def setting_is_org_scoped(key: str) -> bool:
    """True for org-level config, False for a personal preference. channel.*
    keys and SecretStr credentials are org structurally; other fields are org
    iff tagged with the ORG marker. Only decides where a write lands, not read
    fallback."""
    if key.startswith("channel."):
        return True
    field = UserSettings.model_fields.get(key)
    if field is None:
        return False
    if UserSettings.field_is_sensitive(key):
        return True
    return any(isinstance(m, _OrgScoped) for m in field.metadata)


_current_scope: ContextVar["TenantScope | None"] = ContextVar("settings_scope", default=None)


@contextmanager
def use_settings_scope(scope: "TenantScope"):
    """Bind the tenant scope get_user_settings() resolves against for the
    duration of the block — set at the turn boundary (the detached producer) and
    per request, so deep readers don't each need the scope threaded in."""
    token = _current_scope.set(scope)
    try:
        yield
    finally:
        _current_scope.reset(token)


def current_settings_scope() -> "TenantScope | None":
    """The ambient tenant scope bound by use_settings_scope (None outside a
    bound request/turn) — for org-keying non-settings resources on the same
    boundary, e.g. the in-process harness memory root."""
    return _current_scope.get()


def get_user_settings(scope: "TenantScope | None" = None) -> UserSettings:
    """Resolved settings for a scope: explicit arg, else ambient
    (use_settings_scope), else LOCAL_SCOPE. Unscoped resolves global rows only,
    never another org's data. Loads fresh every call — no process-global cache."""
    from cowork.db.scoped import LOCAL_SCOPE

    scope = scope or _current_scope.get() or LOCAL_SCOPE
    return _load_from_db(scope)


def invalidate_user_settings_cache() -> None:
    # No cache anymore (get_user_settings loads fresh); kept as a no-op so
    # post-write callers don't need to change.
    pass


def _load_from_db(scope: "TenantScope") -> UserSettings:
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    session = get_open_session()
    try:
        return SettingService(session, scope).load()
    finally:
        session.close()
