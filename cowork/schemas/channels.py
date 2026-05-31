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
