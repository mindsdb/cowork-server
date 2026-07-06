import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
# RECOMMENDED_PAIR / *_MODEL_DEFAULTS below. NOTE: MindsHub's router resolves
# only the canonical ``latest:<alias>`` form; a bare alias (``sonnet``/``haiku``)
# is rejected with an uncaught HTTP 500 (ENG-577). The pairs below are kept bare
# for display, and callers that hit the router normalize via
# ``providers.canonical_minds_model`` (the live ``/v1/models`` ids are already
# canonical). If these defaults ever reach the router unnormalized, prefix them.
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Global config lives in ~/.cowork/.env now; ~/.anton/.env is kept as
        # a fallback for un-migrated installs. Order matters: pydantic-settings
        # is "last wins", so ~/.cowork/.env must come AFTER ~/.anton/.env (fresh
        # over stale), with local ".env" highest for dev overrides.
        env_file=[str(Path.home() / ".anton" / ".env"), str(Path.home() / ".cowork" / ".env"), ".env"],
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        extra="ignore",
    )


class DatabaseSettings(Settings):
    uri: str = Field(
        default=f"sqlite:///{str(Path.home() / ".cowork" / "cowork.db")}", description="The database connection URI"
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
        default=str(Path.home() / ".cowork" / "projects"),
        validation_alias=AliasChoices("COWORK_PROJECTS_DIR", "PROJECTS_ROOT_DIR"),
        description="Root directory where project folders are stored",
    )  # PROJECT_ROOT_DIR or COWORK_PROJECTS_DIR or PROJECTS_ROOT_DIR


class FileSettings(Settings):
    root_dir: str = Field(
        default=str(Path.home() / ".cowork" / "files"),
        validation_alias=AliasChoices("COWORK_FILES_DIR", "FILES_ROOT_DIR"),
        description="Root directory where uploaded files are stored",
    )  # FILE_ROOT_DIR or COWORK_FILES_DIR or FILES_ROOT_DIR


class SkillSettings(Settings):
    root_dir: str = Field(
        default=str(Path.home() / ".cowork" / "skills"),
        validation_alias=AliasChoices("COWORK_SKILLS_DIR", "SKILLS_ROOT_DIR"),
        description="Root directory where agentskills.io-format skill folders are stored",
    )  # COWORK_SKILLS_DIR or SKILLS_ROOT_DIR


class ConnectorSettings(Settings):
    vault_dir: str = Field(
        default=str(Path.home() / ".cowork" / "data-vault"),
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

    server_origin: str = Field(
        default="http://127.0.0.1:26866",
        validation_alias=AliasChoices("COWORK_SERVER_ORIGIN"),
        description="Public base URL of this server, used to build OAuth redirect URIs",
    )
    state_path: str = Field(
        default=str(Path.home() / ".cowork" / "oauth_state.json"),
        description="Path to the file used to persist pending OAuth state",
    )


class MemorySettings(Settings):
    root_dir: str = Field(
        default=str(Path.home() / ".cowork" / "memory"),
        description="Root directory for all memory files",
    )


class StreamSettings(Settings):
    backend: str = Field(
        default="file",
        validation_alias=AliasChoices("COWORK_STREAM_BACKEND"),
        description="Turn-stream buffer backend: 'file' (desktop / single-instance cloud) or 'redis' (multi-instance cloud, WIP)",
    )
    dir: str = Field(
        default=str(Path.home() / ".cowork" / "streams"),
        validation_alias=AliasChoices("COWORK_STREAMS_DIR"),
        description="Root directory for file-backed turn-stream buffers",
    )


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    port: int = Field(
        default=26866,
        validation_alias=AliasChoices("COWORK_SERVER_PORT"),
        description="The port to run the server on",
    )
    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("COWORK_SERVER_HOST"),
        description="The host to run the server on",
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
        default=str(Path.home() / ".cowork" / ".master_key"),
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
