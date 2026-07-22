import os
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
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"],
    "openai-compatible": [],
}

RECOMMENDED_PAIR: dict[str, tuple[str, str]] = {
    "minds-cloud": ("sonnet", "haiku"),
    "anthropic": ("claude-sonnet-4-6", "claude-haiku-4-5-20251001"),
    "openai": ("gpt-5.5", "gpt-5.5-mini"),
    "gemini": ("gemini-2.5-pro", "gemini-2.5-flash"),
    "openai-compatible": ("", ""),
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
PLANNING_MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.5",
    "gemini": "gemini-2.5-pro",
    "minds_cloud": "sonnet",
}
CODING_MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.5-mini",
    "gemini": "gemini-2.5-flash",
    "minds_cloud": "haiku",
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
#   dev:     api.dev.mindshub.ai    / view.dev.mindshub.ai
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
        description="Root directory where project folders are stored",
    )  # PROJECT_ROOT_DIR or COWORK_PROJECTS_DIR or PROJECTS_ROOT_DIR


class FileSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "files"),
        validation_alias=AliasChoices("COWORK_FILES_DIR", "FILES_ROOT_DIR"),
        description="Root directory where uploaded files are stored",
    )  # FILE_ROOT_DIR or COWORK_FILES_DIR or FILES_ROOT_DIR


class SkillSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "skills"),
        validation_alias=AliasChoices("COWORK_SKILLS_DIR", "SKILLS_ROOT_DIR"),
        description="Root directory where agentskills.io-format skill folders are stored",
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
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "memory"),
        description="Root directory for all memory files",
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
    identity_enforce: Literal["audit", "enforce"] = Field(
        default="audit",
        validation_alias=AliasChoices("COWORK_IDENTITY_ENFORCE"),
        description=(
            "Org-mode identity enforcement. 'audit' (default): requests without "
            "identity headers are logged and allowed through. 'enforce': they "
            "are rejected with 401. Flip to 'enforce' once the audit log shows "
            "all legitimate identity-less callers are handled."
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

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE_*
    project: ProjectSettings = Field(default_factory=ProjectSettings)  # PROJECT_*
    file: FileSettings = Field(default_factory=FileSettings)  # FILE_*
    skill: SkillSettings = Field(default_factory=SkillSettings)  # SKILL_*
    connector: ConnectorSettings = Field(default_factory=ConnectorSettings)  # CONNECTOR_*
    memory: MemorySettings = Field(default_factory=MemorySettings)  # MEMORY_*


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
