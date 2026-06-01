from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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


class ConnectorSettings(Settings):
    vault_dir: str = Field(
        default=str(Path.home() / ".cowork" / "data-vault"),
        validation_alias=AliasChoices("COWORK_VAULT_DIR", "CONNECTOR_VAULT_DIR"),
        description="Root directory for the local data vault (saved connector credentials)",
    )


class OAuthSettings(Settings):
    google_client_id: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_CLIENT_ID"),
        description="Google OAuth client ID",
    )
    google_client_secret: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_CLIENT_SECRET"),
        description="Google OAuth client secret",
    )
    server_origin: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias=AliasChoices("COWORK_SERVER_ORIGIN"),
        description="Public base URL of this server, used to build OAuth redirect URIs",
    )
    state_path: str = Field(
        default=str(Path.home() / ".cowork" / "oauth_state.json"),
        description="Path to the file used to persist pending OAuth state",
    )


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    port: int = Field(
        default=int(os.environ.get("COWORK_SERVER_PORT", os.environ.get("SERVER_PORT", 26866))),
        description="The port to run the server on"
    )
    host: str = Field(
        default=os.environ.get("COWORK_SERVER_HOST", os.environ.get("SERVER_HOST", "127.0.0.1")),
        description="The host to run the server on"
    )

    log_level: str = Field(default="WARNING", description="The logging level")  # LOG_LEVEL

    master_key_path: str = Field(
        default=str(Path.home() / ".cowork" / ".master_key"),
        description="Path to the Fernet master key file used to encrypt sensitive settings",
    )  # MASTER_KEY_PATH

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE_*
    project: ProjectSettings = Field(default_factory=ProjectSettings)  # PROJECT_*
    file: FileSettings = Field(default_factory=FileSettings)  # FILE_*
    connector: ConnectorSettings = Field(default_factory=ConnectorSettings)  # CONNECTOR_*


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
