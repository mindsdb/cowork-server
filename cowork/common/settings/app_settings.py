import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cowork.common.paths import cowork_home


# ── Global model catalog ───────────────────────────────────────────────
# Recommended models and per-provider model defaults are global,
# application-level configuration — the same for every user — so they live
# here rather than as per-user fields on UserSettings.
#
# minds-cloud model names are owned by MindsHub, not this repo. The list is
# resolved at runtime from MindsHub's OpenAI-compatible `/v1/models` endpoint
# (see cowork.services.providers.fetch_minds_models) and supplied by the
# /settings/recommended-models endpoint. It is intentionally left empty here
# so no aliases are hand-maintained — the working default pair lives in
# RECOMMENDED_PAIR / *_MODEL_DEFAULTS below. MindsHub aliases are bare
# (``sonnet``); the older ``latest:`` prefix still resolves but is deprecated.
RECOMMENDED_MODELS: dict[str, list[str]] = {
    "minds-cloud": [],
    "anthropic": ["claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-5.5", "gpt-5.5-mini", "o3", "o4-mini"],
    # Live-overlaid from Google's OpenAI-compatible /models when a Gemini key is
    # configured (see recommended_models endpoint); this static list is only the
    # fallback for the pre-key onboarding pick and offline loads. The old
    # gemini-2.5-* ids and the never-real "gemini-3-flash-preview" all 404 for
    # new users, so the fallback lists only current ids (ENG-1145).
    "gemini": ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite"],
    "openai-compatible": [],
}

# Keyed by the Provider enum *value* (the string) rather than the enum
# itself, so this module stays free of a circular import with user_settings,
# which owns the Provider enum.
# gemini has concrete recommended models (see RECOMMENDED_MODELS); openai-
# compatible is BYO-endpoint with no canonical model, so it deliberately has no
# entry here. Consequence in resolved_*_model: the user's own model is kept ONLY
# while openai-compatible is the explicitly selected provider; on a *switch* to
# it the lookup misses → None (not the prior provider's model), which trips
# config_status's model gate ("select a model") rather than misrouting.
#
# The one model MindsHub's free monthly allowance covers; every other alias
# bills the wallet. It is also every minds-cloud role default below, so the
# name is declared here rather than beside the org-mode fallback that used to
# be its only reader.
MINDS_FREE_MODEL = "mindshub_air"

# Single source for the per-role defaults: PLANNING/CODING/ROUTER_MODEL_
# DEFAULTS and RECOMMENDED_PAIR below are all derived from this table.
MODEL_ROLE_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "planning": "claude-sonnet-4-6",
        "coding": "claude-haiku-4-5-20251001",
        "router": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "planning": "gpt-5.5",
        "coding": "gpt-5.5-mini",
        "router": "gpt-5.5-mini",
    },
    "gemini": {
        # All three roles use the one id confirmed to work on a fresh
        # free-tier Google key; other ids were zero-quota/404 for new keys
        # but stay reachable via the picker.
        "planning": "gemini-3.6-flash",
        "coding": "gemini-3.6-flash",
        "router": "gemini-3.6-flash",
    },
    # Every role defaults to the free model, so a user who has picked nothing
    # can complete a whole turn without the wallet being charged for any of it.
    # The two invisible roles are the reason this matters: a user sees and can
    # change the planning model, but the coding role (completion verifier,
    # scratchpad) and the router role (respond-vs-delegate, history
    # summarization) run unseen, so a paid default there is denied on an empty
    # wallet with nothing the user can look at to explain why. Wallet-aware
    # resolution downstream is the safety net for a stored pin; this is the
    # value a fresh account starts from, which is also the state where no
    # availability map has been fetched yet and there is nothing to be aware of.
    "minds_cloud": {
        "planning": MINDS_FREE_MODEL,
        "coding": MINDS_FREE_MODEL,
        # Not chosen on price: this role gates every turn, and a slow model here
        # is measurably worse than no router at all. Unmeasured on that axis.
        "router": MINDS_FREE_MODEL,
    },
}
# The agent roles a model is resolved for, in the order the picker's
# `recommendedPair` lists them. One declaration, because three places used to
# spell it out: this order, the derived dicts below, and the pair overlay in the
# recommended-models endpoint. Mirrors the role enum in the MindsHub model
# policy's JSON Schema; the two are a cross-repo contract and a role in one and
# not the other is silently dropped at the parse in fetch_minds_models.
# `test_role_default_dicts_match_the_source_table` holds it to the table above.
AGENT_ROLE_ORDER: tuple[str, ...] = ("planning", "coding", "router")
AGENT_ROLE_NAMES: frozenset[str] = frozenset(AGENT_ROLE_ORDER)

# The table above, pivoted: role -> provider -> model. Anything that resolves a
# role it was handed the name of reads this, so no caller has to line a role up
# against a positional list of the three dicts below.
ROLE_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    role: {p: r[role] for p, r in MODEL_ROLE_DEFAULTS.items()} for role in AGENT_ROLE_ORDER
}
PLANNING_MODEL_DEFAULTS: dict[str, str] = ROLE_MODEL_DEFAULTS["planning"]
CODING_MODEL_DEFAULTS: dict[str, str] = ROLE_MODEL_DEFAULTS["coding"]
ROUTER_MODEL_DEFAULTS: dict[str, str] = ROLE_MODEL_DEFAULTS["router"]

# Per-provider default model tuple served to the picker as `recommendedPair`.
# Order: AGENT_ROLE_ORDER; values derived from MODEL_ROLE_DEFAULTS above.
# openai-compatible has no canonical default, so it's a literal special case.
# The frontend falls back to the coding slot when the 3rd is absent, so an
# older client still works.
RECOMMENDED_PAIR: dict[str, tuple[str, ...]] = {
    **{
        provider.replace("_", "-"): tuple(roles[role] for role in AGENT_ROLE_ORDER)
        for provider, roles in MODEL_ROLE_DEFAULTS.items()
    },
    "openai-compatible": ("",) * len(AGENT_ROLE_ORDER),
}

# Reasoning-effort capability for direct (BYOK) provider models. minds-cloud
# advertises its levels live via MindsHub's `/v1/models`; direct Anthropic/OpenAI
# have no such endpoint, so the levels are hand-maintained here. Keyed by exact
# model id → {"efforts": [<display order>], "default": <one of efforts>}. A model
# absent from this map (e.g. claude-haiku) is treated as not supporting effort —
# the UI hides the picker for it. Levels mirror what each provider accepts:
# Anthropic via output_config={"effort": ...}; OpenAI via reasoning_effort /
# reasoning={"effort": ...}.
#
# Anthropic effort ladder (per the Claude API reference): default is "high";
# "max" is supported on Opus 4.6+ and Sonnet 4.6 (not Haiku/older Sonnets);
# "xhigh" was added in Opus 4.7, so only Opus 4.7/4.8 carry it. Haiku 4.5 has no
# effort support and is intentionally absent.
DIRECT_EFFORT_CATALOG: dict[str, dict] = {
    "claude-opus-4-8":   {"efforts": ["low", "medium", "high", "xhigh", "max"], "default": "high"},
    "claude-opus-4-7":   {"efforts": ["low", "medium", "high", "xhigh", "max"], "default": "high"},
    "claude-opus-4-6":   {"efforts": ["low", "medium", "high", "max"], "default": "high"},
    "claude-sonnet-4-6": {"efforts": ["low", "medium", "high", "max"], "default": "high"},
    "gpt-5.5":      {"efforts": ["minimal", "low", "medium", "high"], "default": "medium"},
    "gpt-5.5-mini": {"efforts": ["minimal", "low", "medium", "high"], "default": "medium"},
    "o3":      {"efforts": ["low", "medium", "high"], "default": "medium"},
    "o4-mini": {"efforts": ["low", "medium", "high"], "default": "medium"},
}


# ── Environment-aware MindsHub URLs ─────────────────────────────────
# The URL pattern is:
#   prod:    api.mindshub.ai    / view.mindshub.ai
#   staging: api.staging.mindshub.ai / view.staging.mindshub.ai
#   local:   same as dev (local dev typically targets the dev env)


# The only non-prod environments that have MindsHub sub-domains. Anything
# else (unset, 'local', 'prod', a typo like 'stagging', or an ambient ENV
# such as the POSIX shell's ENV=~/.kshrc) resolves to prod rather than being
# interpolated into a bogus hostname like api.<garbage>.mindshub.ai.
_KNOWN_ENV_SLUGS = ("staging", "dev")


def _env_slug() -> str:
    """Return the env slug for URL construction, or '' for prod.

    Only the known non-prod slugs in ``_KNOWN_ENV_SLUGS`` produce a sub-domain;
    every other value (unset, 'local', 'prod', typos, or an ambient ENV from
    the shell) resolves to '' (production). Desktop installs never set ENV, so
    they correctly default to prod. Cloud deploys set ENV explicitly.
    """
    env = os.environ.get("ENV", "").lower()
    return env if env in _KNOWN_ENV_SLUGS else ""


def default_minds_api_host() -> str:
    """Environment-aware MindsHub API host (no path)."""
    slug = _env_slug()
    return f"https://api.{slug}.mindshub.ai" if slug else "https://api.mindshub.ai"


def default_minds_url() -> str:
    """Environment-aware MindsHub API URL (with /v1 path)."""
    return f"{default_minds_api_host()}/v1"


def default_turn_minds_api_host() -> str:
    """MindsHub API host for the POD's turn — this deployment's OWN inference.

    Per-PR envs are the one case ``default_minds_api_host`` gets wrong: they all
    run ENV=development, so it yields dev's host, but each has its own inference
    AND its own auth database — a key minted here is unknown to dev's auth (401).
    Their namespace carries the slug, so derive from that (downward API, not a
    Host header: a crafted Host would send the pod, and the minted key, to an
    attacker's endpoint).

    Everything else — dev, staging, prod, desktop — keeps the ENV-slug host,
    which is already correct (notably prod, which has no slug at all).
    """
    ns = (os.environ.get("POD_NAMESPACE") or os.environ.get("NAMESPACE") or "").strip()
    if ns.startswith("pr-"):
        return f"https://api-{ns}.dev.mindshub.ai"
    return default_minds_api_host()


def default_publish_url() -> str:
    """Environment-aware MindsHub publish/view URL."""
    slug = _env_slug()
    return f"https://view.{slug}.mindshub.ai" if slug else "https://view.mindshub.ai"


def _env_file_chain() -> list[str]:
    """The ``.env`` search path (pydantic-settings is "last wins").

    ``<COWORK_HOME>/.env`` is the current global config, with a local ``.env``
    highest for dev overrides. The legacy ``~/.anton/.env`` is a fallback for
    un-migrated installs — but ONLY for the default (prod) home. An isolated
    build (``COWORK_HOME`` set) must NOT inherit that prod-era file: a path var
    living there (``DATABASE_URI``, ``MASTER_KEY_PATH``, ``COWORK_PROJECTS_DIR``,
    …) would resolve every build back onto the same DB/paths and defeat the
    isolation (the exact ENG-324 shared-DB failure this exists to prevent).

    COWORK_HOME is read at import; the desktop app sets it before the server
    process starts, so an isolated build reads its own .env.
    """
    files = [str(cowork_home() / ".env"), ".env"]
    if not os.environ.get("COWORK_HOME"):
        # Prod (default home) still consults the legacy file, ordered BEFORE
        # <COWORK_HOME>/.env so the migrated file wins (fresh over stale).
        files.insert(0, str(Path.home() / ".anton" / ".env"))
    return files


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file_chain(),
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        extra="ignore",
    )


class DatabaseSettings(Settings):
    uri: str = Field(
        default_factory=lambda: f"sqlite:///{cowork_home() / 'cowork.db'}",
        description="The database connection URI",
    )  # DATABASE_URI

    # Connection pool configurations
    max_overflow: int = Field(
        default=20, description="The maximum overflow size of the database connection pool"
    )  # DATABASE_MAX_OVERFLOW
    pool_pre_ping: bool = Field(default=True, description="Whether to enable pool pre-ping")  # DATABASE_POOL_PRE_PING
    pool_recycle: int = Field(default=300, description="The pool recycle time in seconds")  # DATABASE_POOL_RECYCLE
    pool_size: int = Field(default=20, description="The size of the database connection pool")  # DATABASE_POOL_SIZE
    pool_timeout: int = Field(default=300, description="The pool timeout in seconds")  # DATABASE_POOL_TIMEOUT

    # Query timeout configurations
    query_timeout: int = Field(default=300, description="The query timeout in seconds")  # DATABASE_QUERY_TIMEOUT
    statement_timeout: int = Field(
        default=300000, description="The statement timeout in milliseconds"
    )  # DATABASE_STATEMENT_TIMEOUT


class ProjectSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "projects"),
        validation_alias=AliasChoices("COWORK_PROJECTS_DIR", "PROJECTS_ROOT_DIR"),
        description=(
            "Root directory where project folders are stored. In org mode, only "
            "this path's final component is kept; the parent directory is always "
            "COWORK_HOME, so one organization stays one subtree there."
        ),
    )  # PROJECT_ROOT_DIR or COWORK_PROJECTS_DIR or PROJECTS_ROOT_DIR


class FileSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "files"),
        validation_alias=AliasChoices("COWORK_FILES_DIR", "FILES_ROOT_DIR"),
        description="Root directory where uploaded files are stored",
    )  # FILE_ROOT_DIR or COWORK_FILES_DIR or FILES_ROOT_DIR


class StorageSettings(Settings):
    # Org mode only: stores live under <shared_root>/<org_id>/<store>/ (one
    # mountable subtree per org). Local mode never reads this; in org mode the
    # per-store *_DIR overrides are inert.
    shared_root: str = Field(
        default_factory=lambda: str(cowork_home()),
        validation_alias=AliasChoices("COWORK_SHARED_DIR", "STORAGE_SHARED_ROOT"),
        description="Root of the org-keyed shared storage tree (org mode only)",
    )


class SkillSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "skills"),
        validation_alias=AliasChoices("COWORK_SKILLS_DIR", "SKILLS_ROOT_DIR"),
        description=(
            "Root directory where agentskills.io-format skill folders are stored. "
            "In org mode, only this path's final component is kept; the parent "
            "directory is always COWORK_HOME, so one organization stays one "
            "subtree there."
        ),
    )  # COWORK_SKILLS_DIR or SKILLS_ROOT_DIR


class ConnectorSettings(Settings):
    vault_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "data-vault"),
        validation_alias=AliasChoices("COWORK_VAULT_DIR", "CONNECTOR_VAULT_DIR"),
        description="Root directory for the local data vault (saved connector credentials)",
    )


class OAuthSettings(Settings):
    google_drive_client_id: str = Field(default="", validation_alias=AliasChoices("GOOGLE_DRIVE_CLIENT_ID"))
    google_drive_client_secret: str = Field(default="", validation_alias=AliasChoices("GOOGLE_DRIVE_CLIENT_SECRET"))

    google_calendar_client_id: str = Field(default="", validation_alias=AliasChoices("GOOGLE_CALENDAR_CLIENT_ID"))
    google_calendar_client_secret: str = Field(default="", validation_alias=AliasChoices("GOOGLE_CALENDAR_CLIENT_SECRET"))

    gmail_client_id: str = Field(default="", validation_alias=AliasChoices("GMAIL_CLIENT_ID"))
    gmail_client_secret: str = Field(default="", validation_alias=AliasChoices("GMAIL_CLIENT_SECRET"))

    google_ads_client_id: str = Field(default="", validation_alias=AliasChoices("GOOGLE_ADS_CLIENT_ID"))
    google_ads_client_secret: str = Field(default="", validation_alias=AliasChoices("GOOGLE_ADS_CLIENT_SECRET"))

    google_analytics_client_id: str = Field(default="", validation_alias=AliasChoices("GOOGLE_ANALYTICS_CLIENT_ID"))
    google_analytics_client_secret: str = Field(default="", validation_alias=AliasChoices("GOOGLE_ANALYTICS_CLIENT_SECRET"))

    linear_client_id: str = Field(default="", validation_alias=AliasChoices("LINEAR_CLIENT_ID"))
    linear_client_secret: str = Field(default="", validation_alias=AliasChoices("LINEAR_CLIENT_SECRET"))

    github_client_id: str = Field(default="", validation_alias=AliasChoices("GITHUB_CLIENT_ID"))
    github_client_secret: str = Field(default="", validation_alias=AliasChoices("GITHUB_CLIENT_SECRET"))

    # Browser-side key for the Google Picker widget (drive.file scope only
    # grants access to files the user explicitly picks via this UI).
    google_picker_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_PICKER_API_KEY"))

    server_origin: str = Field(
        default="http://127.0.0.1:26866",
        validation_alias=AliasChoices("COWORK_SERVER_ORIGIN"),
        description="Public base URL of this server, used to build OAuth redirect URIs",
    )
    state_path: str = Field(
        default_factory=lambda: str(cowork_home() / "oauth_state.json"),
        description="Path to the file used to persist pending OAuth state",
    )


class MemorySettings(Settings):
    # populate_by_name: callers construct MemorySettings(root_dir=...) directly.
    model_config = SettingsConfigDict(populate_by_name=True)

    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "memory"),
        validation_alias=AliasChoices("COWORK_MEMORY_DIR", "MEMORY_ROOT_DIR"),
        description=(
            "Root directory for all memory files. In org mode, only this path's "
            "final component is kept; the parent directory is always COWORK_HOME, "
            "so one organization stays one subtree there."
        ),
    )


class StreamSettings(Settings):
    backend: str = Field(
        default="file",
        validation_alias=AliasChoices("COWORK_STREAM_BACKEND"),
        description="Turn-stream buffer backend: 'file' (desktop / single-instance cloud) or 'redis' (multi-instance cloud, WIP)",
    )
    dir: str = Field(
        default_factory=lambda: str(cowork_home() / "streams"),
        validation_alias=AliasChoices("COWORK_STREAMS_DIR"),
        description="Root directory for file-backed turn-stream buffers",
    )


class TurnQueueSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="COWORK_TURN_")

    backend: str = Field(
        default="inprocess",
        description="Turn-queue backend: 'inprocess' (single-instance, default) or 'remote' (Redis-backed, multi-instance).",
    )

    @property
    def is_remote(self) -> bool:
        return self.backend == "remote"

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used when backend is 'remote'.",
    )
    jobs_stream: str = Field(
        default="scratchpad:requests",
        description=(
            "Key prefix for turn job streams when backend is 'remote'. Each conversation "
            "gets its own stream at '{prefix}:{conversation_id}', and '{prefix}:queues' is "
            "the set of conversations that have one. The controller locks a conversation "
            "before reading its stream, so a job for a busy pod is never delivered and "
            "never blocks another conversation."
        ),
    )
    reply_idle_timeout_seconds: float = Field(
        default=600.0,
        description=(
            "Fail a remote turn after this many seconds with no reply for it on the reply "
            "stream (worker down, crashed, or wedged). Generous on purpose: the reply "
            "protocol has no heartbeat, so a long tool run legitimately produces no reply "
            "for minutes — tighten it once the pod sends one. <= 0 disables the bound, "
            "which means an unresponsive worker leaves the turn spinning forever."
        ),
    )  # COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS
    auth_internal_base_url: str = Field(
        default="",
        description="Base URL of the auth service's internal API, used to mint per-turn MindsHub keys.",
    )  # COWORK_TURN_AUTH_INTERNAL_BASE_URL
    auth_internal_secret: str = Field(
        default="",
        description="Shared secret sent as X-Internal-Auth when minting per-turn MindsHub keys.",
    )  # COWORK_TURN_AUTH_INTERNAL_SECRET
    turn_key_ttl_seconds: int = Field(
        default=1200,
        description="TTL, in seconds, of the minted per-turn MindsHub key (20 min; keep within auth's turn_key_max_ttl_seconds).",
    )  # COWORK_TURN_TURN_KEY_TTL_SECONDS
    minds_base_url: str = Field(
        default="",
        description=(
            "Explicit MindsHub inference base URL (OpenAI-compatible, incl. /v1) the pod's "
            "turn calls. Overrides the env-slug default (default_minds_api_host); required for "
            "per-PR / non-standard envs whose host the slug logic cannot derive. Empty = derive."
        ),
    )  # COWORK_TURN_MINDS_BASE_URL
    minds_coding_model: str = Field(
        default="",
        description=(
            "MindsHub model alias for the pod's coding calls (completion verifier + nested "
            "scratchpad calls). The pod always runs on minds-cloud, so this must be a minds "
            "alias the env serves. Empty = the minds-cloud coding default (CODING_MODEL_DEFAULTS)."
        ),
    )  # COWORK_TURN_MINDS_CODING_MODEL


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    port: int = Field(
        default=26866,
        # One name per context: the desktop app hands the sidecar its port as
        # COWORK_SERVER_PORT; k8s/cloud sets COWORK_LISTEN_PORT,
        # which wins because k8s auto-injects the former as a tcp:// URI.
        validation_alias=AliasChoices("COWORK_LISTEN_PORT", "COWORK_SERVER_PORT"),
        description="The port to run the server on",
    )

    @field_validator("port", mode="before")
    @classmethod
    def _discard_k8s_injected_port(cls, v: object) -> object:
        if isinstance(v, str) and v.startswith("tcp://"):
            return 26866
        return v
    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("COWORK_SERVER_HOST"),
        description="The host to run the server on",
    )

    # Port the Vite renderer dev server listens on — included in the default
    # CORS allowed origins so `make dev` / `make watch` work out of the box.
    renderer_port: int = Field(
        default=5173,
        validation_alias=AliasChoices("COWORK_RENDERER_PORT", "VITE_RENDERER_PORT"),
        description="Vite dev server port (used to build default CORS allowed origins).",
    )

    # CORS allowed origins.  When empty the validator below fills in localhost
    # on both configured ports.  Packaged Electron loads from file:// with
    # webSecurity:false so no Origin header is sent — not needed here.
    # Override for cloud/VPC:  COWORK_ALLOWED_ORIGINS='["https://app.example.com"]'
    # Use ["*"] only when an ingress controller enforces origin filtering upstream.
    allowed_origins: list[str] = Field(
        default=[],
        validation_alias=AliasChoices("COWORK_ALLOWED_ORIGINS"),
        description=(
            "CORS allowed origins (JSON array). "
            "Defaults to localhost on COWORK_LISTEN_PORT and COWORK_RENDERER_PORT."
        ),
    )

    @model_validator(mode="after")
    def _default_allowed_origins(self) -> "AppSettings":
        if not self.allowed_origins:
            self.allowed_origins = [
                f"http://localhost:{self.port}",
                f"http://127.0.0.1:{self.port}",
                f"http://localhost:{self.renderer_port}",
                f"http://127.0.0.1:{self.renderer_port}",
            ]
        return self

    require_auth: bool = Field(
        default=False,
        validation_alias=AliasChoices("COWORK_REQUIRE_AUTH"),
        description=(
            "Require a bearer token on all API requests (except /health). "
            "Set COWORK_AUTH_TOKEN to a fixed token, or leave it empty to "
            "auto-generate one on first startup (written back to ~/.cowork/.env)."
        ),
    )
    auth_token: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_AUTH_TOKEN"),
        description=(
            "Bearer token clients must send as 'Authorization: Bearer <token>'. "
            "Only checked when COWORK_REQUIRE_AUTH=true. Auto-generated if empty."
        ),
    )
    tenancy_mode: Literal["local", "org"] = Field(
        default="local",
        validation_alias=AliasChoices("COWORK_TENANCY_MODE"),
        description=(
            "Deployment tenancy mode. 'local' (default): single-user desktop "
            "sidecar — request auth is the shared bearer token above. 'org': "
            "multi-tenant cloud deployment behind the auth gateway — requests "
            "carry trusted identity headers (X-User-Id / X-Organization-Id) "
            "from which a per-request principal is built."
        ),
    )
    pod_scratch_dir: str = Field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "cowork"),
        validation_alias=AliasChoices("COWORK_POD_SCRATCH_DIR"),
        description=(
            "Org mode only (see cowork.common.paths.pod_local_only): root for "
            "scratch and deployment-local state that carries no org_id segment "
            "and so must never sit on the shared COWORK_HOME tree. Covers the "
            "connector probe's plaintext credential env files, publish's "
            "state.json, and the anton harness's temporary data-vault "
            "directory. Local mode never reads this field; those stores keep "
            "resolving under COWORK_HOME exactly as before. Defaults to the "
            "container's own temp directory, which is never the shared EFS "
            "mount and is gone on pod restart."
        ),
    )
    ask_user_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("COWORK_ASK_USER_ENABLED"),
        description=(
            "Whether the agent may ask interactive multiple-choice questions "
            "(the `ask_user` tool). Off by default because the renderer must "
            "ship first: the frontend and this server are versioned "
            "independently, and a client that does not know the "
            "`response.ask_user` event drops it silently, leaving the agent "
            "apparently hung until the question times out. Turn on only after "
            "the frontend is rolled out."
        ),
    )
    surface_override: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_SURFACE"),
        description=(
            "Deployer-declared surface for trace attribution (ENG-1459), "
            "overriding the tenancy-based inference. Inference reads org "
            "tenancy as 'web' and anything else as 'desktop', which is right "
            "for the two surfaces that matter and wrong for deployments that "
            "are neither: the hub snapshot instances being deprecated run "
            "local tenancy but are not desktops, and the enterprise container "
            "is self-hosted. Those pass their own value here rather than "
            "inflating the desktop population they are measured against. "
            "Plain str, not a Literal, for the same reason as the channel "
            "above: an invalid value must never fail settings load over "
            "telemetry. It does NOT fall back to inference — an unrecognised "
            "value logs a warning and leaves the surface absent (#357 review). "
            "Inferring would be actively wrong for the deployments this "
            "override exists for: a typo from a hub snapshot or the enterprise "
            "container would relabel it 'desktop' and inflate the very "
            "baseline web is measured against, silently. Absent is honestly "
            "unknown; guessed is not. Empty (default) = infer."
        ),
    )
    install_channel_override: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_INSTALL_CHANNEL"),
        description=(
            "Deployer-declared install channel for trace attribution "
            "(ENG-1279), overriding inference from tenancy/pip metadata. "
            "Needed where inference is wrong from inside the process: hub "
            "snapshot instances run local tenancy with a PyPI-installed "
            "cowork-server inside their docker image, so they pass 'hosted' "
            "here. Plain str, not a Literal — an invalid value must degrade "
            "to inference (build_info validates and warns), never fail "
            "settings load over telemetry. Empty (default) = infer."
        ),
    )
    identity_enforce: Literal["audit", "enforce"] = Field(
        default="audit",
        validation_alias=AliasChoices("COWORK_IDENTITY_ENFORCE"),
        description=(
            "Org-mode identity enforcement. 'enforce' (default): requests without "
            "identity headers are rejected with 401. 'audit': they are logged and "
            "allowed through instead — an explicit opt-out for local debugging "
            "against org mode, not a rollout stage; every real deployment sets "
            "'enforce' in its Helm values regardless of this default."
        ),
    )
    owner: str = Field(
        default=os.environ.get("COWORK_SERVER_OWNER", ""),
        description=(
            "Opaque per-install owner token echoed at /health. The desktop app passes the "
            "token it generated and only adopts a server whose /health owner matches, so one "
            "OS user's app never adopts another user's sidecar on a shared loopback port "
            "(ENG-439). Empty means the server advertises no owner and is not adoptable."
        ),
    )  # COWORK_SERVER_OWNER

    log_level: str = Field(default="WARNING", description="The logging level")  # LOG_LEVEL

    master_key: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_MASTER_KEY"),
        description=(
            "Fernet master key material (urlsafe-base64) used to encrypt "
            "sensitive settings at rest. When set, it is used directly and "
            "master_key_path is ignored. Required for stateless/cloud deploys "
            "so the key survives pod restarts (a per-pod file key would orphan "
            "everything encrypted before the restart). Empty (desktop default) "
            "reads or generates the key at master_key_path."
        ),
    )  # COWORK_MASTER_KEY

    master_key_path: str = Field(
        default_factory=lambda: str(cowork_home() / ".master_key"),
        description="Path to the Fernet master key file used to encrypt sensitive settings",
    )  # MASTER_KEY_PATH

    public_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_PUBLIC_BASE_URL", "COWORK_SERVER_ORIGIN"),
        description=(
            "Public HTTPS base URL of this server (e.g. https://cowork.example.com), "
            "used to build channel webhook URLs for setWebhook-style registration. "
            "Empty when the server is not publicly reachable."
        ),
    )  # COWORK_PUBLIC_BASE_URL

    conversation_link_template: str = Field(
        default="",
        validation_alias=AliasChoices("COWORK_CONVERSATION_LINK_TEMPLATE"),
        description=(
            "Link template appended to channel replies whose turn ran tools, "
            "with a {conversation_id} placeholder. Empty disables the link."
        ),
    )  # COWORK_CONVERSATION_LINK_TEMPLATE

    channels_harness: str = Field(
        default="anton",
        validation_alias=AliasChoices("COWORK_CHANNELS_HARNESS"),
        description=(
            "Harness that serves channel conversations (e.g. 'anton', 'hermes'). "
            "Applies to NEW channel conversations only — existing ones stay pinned "
            "to the harness that first served them. Independent of the UI harness "
            "selection, which never applies to channels."
        ),
    )  # COWORK_CHANNELS_HARNESS

    # Deployment-level defaults for the per-user agent tool budgets. Users who
    # set the corresponding UserSettings override these; users who don't get
    # these values. Hosted deployments (where inference cost lands on the
    # operator and the Settings UI is unreachable — sidebar entries are
    # desktop-only) can lower them without touching per-user rows; commented
    # entries live in deployment/cowork-server/values-{prod,staging}.yaml.
    # NOTE: get_app_settings() is @lru_cache'd, so changing the COWORK_DEFAULT_*
    # env requires a process restart — "I changed the env and nothing happened"
    # means the old value is cached for the life of the process.
    default_max_tool_rounds: int = Field(
        default=50,
        ge=5,
        le=500,
        validation_alias=AliasChoices("COWORK_DEFAULT_MAX_TOOL_ROUNDS"),
        description="Default for the per-user 'Max Steps per Task' agent budget.",
    )  # COWORK_DEFAULT_MAX_TOOL_ROUNDS
    default_max_continuations: int = Field(
        default=5,
        ge=0,
        le=25,
        validation_alias=AliasChoices("COWORK_DEFAULT_MAX_CONTINUATIONS"),
        description="Default for the per-user 'Max Auto-Continues' agent budget.",
    )  # COWORK_DEFAULT_MAX_CONTINUATIONS
    # Unlike the two above, this one does NOT deliberately run looser than
    # anton's own default — it matches it (ENG-1286's 1,250,000). The measured
    # per-turn distribution that set that number came from Cowork traffic, so
    # it already reflects these looser round budgets; raising it here would
    # loosen a ceiling against the very population it was sized on.
    default_max_turn_tokens: int = Field(
        default=1_250_000,
        # Deployment-level default. Kept at ge=750_000 rather than mirroring
        # UserSettings' "0 = unlimited": an operator turning the guard off for a
        # whole org silently is exactly the outcome the per-user sentinel exists
        # to make deliberate. An operator who really wants that sets a huge
        # number, which is at least visible in the value.
        ge=750_000,
        le=50_000_000,
        validation_alias=AliasChoices("COWORK_DEFAULT_MAX_TURN_TOKENS"),
        description="Default for the per-user 'Max Tokens per Task' agent budget.",
    )  # COWORK_DEFAULT_MAX_TURN_TOKENS

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE_*
    project: ProjectSettings = Field(default_factory=ProjectSettings)  # PROJECT_*
    file: FileSettings = Field(default_factory=FileSettings)  # FILE_*
    storage: StorageSettings = Field(default_factory=StorageSettings)  # STORAGE_*
    skill: SkillSettings = Field(default_factory=SkillSettings)  # SKILL_*
    connector: ConnectorSettings = Field(default_factory=ConnectorSettings)  # CONNECTOR_*
    memory: MemorySettings = Field(default_factory=MemorySettings)  # MEMORY_*


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
