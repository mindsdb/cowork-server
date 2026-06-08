from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class InstallationStatus(str, Enum):
    disconnected = "disconnected"
    active = "active"
    error = "error"


class TriggerRule(str, Enum):
    always = "always"
    mention_only = "mention_only"
    regex = "regex"


class EventDirection(str, Enum):
    inbound = "inbound"
    outbound = "outbound"


class EventStatus(str, Enum):
    received = "received"
    routed = "routed"
    delivered = "delivered"
    failed = "failed"
    duplicate = "duplicate"


class ChannelInstallationResponse(BaseModel):
    id: UUID
    channel_type: str
    display_name: str
    enabled: bool
    status: InstallationStatus
    created_at: datetime | None = None
    modified_at: datetime | None = None


class BindingCreateRequest(BaseModel):
    channel_type: str
    external_group_id: str
    external_thread_id: str | None = None
    display_name: str | None = None
    trigger_rule: TriggerRule = TriggerRule.always
    trigger_pattern: str | None = None
    anton_project_id: UUID | None = None
    anton_conversation_id: UUID | None = None


class BindingUpdateRequest(BaseModel):
    # External target identity is fixed at creation; only routing + linkage are
    # editable. All fields optional — only provided ones are applied.
    display_name: str | None = None
    trigger_rule: TriggerRule | None = None
    trigger_pattern: str | None = None
    anton_project_id: UUID | None = None
    anton_conversation_id: UUID | None = None


class BindingResponse(BaseModel):
    id: UUID
    channel_type: str
    external_group_id: str
    external_thread_id: str | None = None
    display_name: str | None = None
    trigger_rule: TriggerRule
    trigger_pattern: str | None = None
    anton_project_id: UUID | None = None
    anton_conversation_id: UUID | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None


class ChannelSessionResponse(BaseModel):
    id: UUID
    binding_id: UUID
    external_session_key: str
    anton_session_id: str | None = None
    last_message_at: datetime | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None


class ChannelEventResponse(BaseModel):
    id: UUID
    channel_type: str
    external_message_id: str | None = None
    direction: EventDirection
    status: EventStatus
    dedupe_key: str | None = None
    error: str | None = None
    created_at: datetime | None = None


class CredentialFieldSpec(BaseModel):
    """A plugin credential field, as advertised to the UI. No values here."""

    name: str
    label: str
    secret: bool
    required: bool
    description: str | None = None


class PluginCapabilities(BaseModel):
    """Capability flags the UI uses to decide which forms/buttons to show."""

    supports_webhook_ingress: bool = False
    supports_webhook_setup: bool = False
    supports_teardown: bool = False
    supports_oauth: bool = False
    supports_direct_credentials: bool = True
    supports_custom_ack: bool = False


class PluginResponse(BaseModel):
    channel_type: str
    display_name: str
    credentials: list[CredentialFieldSpec]
    has_oauth: bool = False
    webhook_paths: list[str] = []
    capabilities: PluginCapabilities


class CredentialValue(BaseModel):
    """Masked credential state. Secret values are never returned: ``value`` is
    null for secret fields (only ``is_set`` is meaningful); non-secret fields
    may echo their stored ``value``."""

    is_set: bool
    value: str | None = None


class ChannelConfigResponse(BaseModel):
    channel_type: str
    configured: bool
    fields: dict[str, CredentialValue]


class ChannelConfigUpdateRequest(BaseModel):
    """Field name → raw value. Only provided fields are written; omitted fields
    keep their existing stored value (no magic sentinel needed). Secret values
    are accepted here and stored encrypted, but never returned."""

    values: dict[str, str]


class ChannelStatusItem(BaseModel):
    channel_type: str
    display_name: str
    enabled: bool
    status: InstallationStatus
    configured: bool


class ChannelStatusResponse(BaseModel):
    plugin_count: int
    installation_count: int
    channels: list[ChannelStatusItem]


class ChannelReloadResponse(BaseModel):
    channel_type: str
    active: bool


class ChannelLifecycleResponse(BaseModel):
    channel_type: str
    action: str  # "setup" | "teardown"
    active: bool
    detail: str
