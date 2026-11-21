"""Request bodies of the custom MCP connection routes: the desktop app's, then the web's."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cowork.services.connectors.custom_mcp import is_connection_id

_MAX_TOOLS = 500
_MAX_TOOL_NAME = 128


def _https(value: str | None) -> str | None:
    if value is not None and not value.startswith("https://"):
        raise ValueError("must be an https URL")
    return value


class CheckUrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class CustomMcpSettings(BaseModel):
    """Held settings and token of one custom MCP connection; every field is
    optional on an update, and None leaves the held value as it is."""

    model_config = ConfigDict(extra="forbid")

    server_url: str | None = Field(default=None, max_length=2048)
    label: str | None = Field(default=None, min_length=1, max_length=128)
    resource: str | None = Field(default=None, max_length=2048)
    authorization_endpoint: str | None = Field(default=None, max_length=2048)
    token_endpoint: str | None = Field(default=None, max_length=2048)
    revocation_endpoint: str | None = Field(default=None, max_length=2048)
    client_id: str | None = Field(default=None, max_length=512)
    allowed_tools: list[str] | None = Field(default=None, max_length=_MAX_TOOLS)
    access_token: str | None = Field(default=None, max_length=16384)
    expires_at: str | None = Field(default=None, max_length=64)

    @field_validator("resource", "authorization_endpoint", "token_endpoint", "revocation_endpoint")
    @classmethod
    def _endpoints_are_https(cls, value: str | None) -> str | None:
        return _https(value)

    @field_validator("allowed_tools")
    @classmethod
    def _tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not name or len(name) > _MAX_TOOL_NAME for name in value):
            raise ValueError(f"tool names must be 1 to {_MAX_TOOL_NAME} characters")
        return value


class CreateCustomMcpRequest(CustomMcpSettings):
    """A new connection: its id (drawn by the app), its label and server URL."""

    id: str
    label: str = Field(min_length=1, max_length=128)
    server_url: str = Field(min_length=1, max_length=2048)

    @field_validator("id")
    @classmethod
    def _connection_id(cls, value: str) -> str:
        if not is_connection_id(value):
            raise ValueError("not a custom MCP connection id")
        return value


class AllowedToolsRequest(BaseModel):
    tools: list[str] = Field(max_length=_MAX_TOOLS)

    @field_validator("tools")
    @classmethod
    def _tool_names(cls, value: list[str]) -> list[str]:
        if any(not name or len(name) > _MAX_TOOL_NAME for name in value):
            raise ValueError(f"tool names must be 1 to {_MAX_TOOL_NAME} characters")
        return value


#: auth's own bounds on the web routes' fields, so a body auth would refuse never leaves.
_MAX_LOGIN_HOSTS = 4
_MAX_LOGIN_HOST = 260
#: auth sets no list bound, but its largest member ceiling (768 KiB) holds at most this many
#: of its smallest stored definitions (62 bytes), so a longer list can never save.
_MAX_CLOUD_SWITCHES = 768 * 1024 // 62


class CloudReconnectRequest(BaseModel):
    """A sign-in again with the stored client, after the member confirmed its login hosts."""

    model_config = ConfigDict(extra="forbid")

    confirmed_login_hosts: list[str] = Field(default_factory=list, max_length=_MAX_LOGIN_HOSTS)

    @field_validator("confirmed_login_hosts")
    @classmethod
    def _hosts(cls, value: list[str]) -> list[str]:
        if any(not host or len(host) > _MAX_LOGIN_HOST for host in value):
            raise ValueError(f"login hosts must be 1 to {_MAX_LOGIN_HOST} characters")
        return value


class CloudConnectRequest(CloudReconnectRequest):
    """A server a member adds on web; auth checks the URL and runs the sign-in.

    ``client_secret`` is relayed to auth once and never logged, kept or returned.
    """

    server_url: str = Field(min_length=1, max_length=2048)
    label: str = Field(min_length=1, max_length=128)
    client_id: str = Field(default="", max_length=512)
    client_secret: str = Field(default="", max_length=512, repr=False)


class CloudSwitchesRequest(BaseModel):
    """The stored tools to switch on, by name, and the snapshot revision they were chosen from."""

    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(max_length=_MAX_CLOUD_SWITCHES)
    tools_revision: int = Field(ge=0, strict=True)

    @field_validator("tools")
    @classmethod
    def _tool_names(cls, value: list[str]) -> list[str]:
        """Each name 1 to 128 characters and named once, as auth's switch save requires."""
        if any(not name or len(name) > _MAX_TOOL_NAME for name in value):
            raise ValueError(f"tool names must be 1 to {_MAX_TOOL_NAME} characters")
        if len(set(value)) != len(value):
            raise ValueError("a tool is named more than once")
        return value


class CloudFinishRequest(BaseModel):
    """The one-time completion id auth put in the finish page's address fragment."""

    model_config = ConfigDict(extra="forbid")

    completion: str = Field(min_length=1, max_length=64)
