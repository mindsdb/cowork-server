from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ChannelInstallation(BaseSQLModel, table=True):
    __tablename__ = "channel_installations"
    __table_args__ = (sa.UniqueConstraint("channel_type", name="uq_channel_installations_type"),)

    channel_type: str = Field(description="Stable adapter name: telegram | slack | discord | whatsapp")
    display_name: str = Field(description="Human-facing channel label for the UI")
    enabled: bool = Field(default=False, description="Whether the adapter should be started")
    status: str = Field(
        default="disconnected",
        description="Last known adapter state: disconnected | active | error",
    )


class ChannelBinding(BaseSQLModel, table=True):
    __tablename__ = "channel_bindings"
    __table_args__ = (
        sa.UniqueConstraint(
            "channel_type",
            "external_group_id",
            "external_thread_key",
            name="uq_channel_bindings_target",
        ),
    )

    channel_type: str = Field(description="Stable adapter name")
    external_group_id: str = Field(description="Platform conversation id (chat/channel id)")
    external_thread_id: str | None = Field(
        default=None,
        description="Optional sub-context (Slack thread, forum post); None = the conversation as a whole",
    )
    # Non-null mirror of external_thread_id used solely for the uniqueness
    # constraint: SQLite treats NULLs as distinct, so a nullable
    # external_thread_id can't stop two whole-channel bindings for the same
    # group. Derived server-side as ``external_thread_id or "__default__"``;
    # callers should not set it directly.
    external_thread_key: str = Field(
        default="__default__",
        index=True,
        description="Routing key for uniqueness; external_thread_id when set, else '__default__'",
    )
    display_name: str | None = Field(default=None, description="Human-facing label for the bound chat")
    trigger_rule: str = Field(default="always", description="always | mention_only | regex")
    trigger_pattern: str | None = Field(default=None, description="Regex source when trigger_rule = regex")
    anton_project_id: UUID | None = Field(
        default=None,
        foreign_key="projects.id",
        description="Project context this channel routes into",
    )
    anton_conversation_id: UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        description="Conversation this binding is pinned to, when one external chat == one conversation",
    )


class ChannelSession(BaseSQLModel, table=True):
    __tablename__ = "channel_sessions"
    __table_args__ = (
        sa.UniqueConstraint("binding_id", "external_session_key", name="uq_channel_sessions_key"),
    )

    binding_id: UUID = Field(foreign_key="channel_bindings.id", description="Parent binding")
    external_session_key: str = Field(
        description="Platform-side session identity (e.g. chat id, or chat id + thread id)",
    )
    anton_session_id: str | None = Field(
        default=None,
        description="Anton runtime session/conversation handle this maps to",
    )
    last_message_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC time of the most recent message routed through this session",
    )


class ChannelEvent(BaseSQLModel, table=True):
    __tablename__ = "channel_events"

    channel_type: str = Field(index=True, description="Stable adapter name")
    external_message_id: str | None = Field(
        default=None,
        description="Platform-side message id, when the platform provides one",
    )
    direction: str = Field(description="inbound | outbound")
    status: str = Field(description="received | routed | delivered | failed | duplicate")
    # Bounded de-dup key (platforms redeliver webhooks); looked up before routing.
    dedupe_key: str | None = Field(default=None, index=True, description="Key used to drop redeliveries")
    error: str | None = Field(default=None, description="Failure detail; never contains secrets")
