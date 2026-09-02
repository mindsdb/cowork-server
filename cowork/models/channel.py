from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ChannelInstallation(BaseSQLModel, table=True):
    __tablename__ = "channel_installations"
    # One per channel_type per org (or global if org_id is NULL).
    # Mirrors settings' org/global partial-index split (models/setting.py).
    __table_args__ = (
        sa.Index(
            "uq_channel_installations_type_global",
            "channel_type",
            unique=True,
            sqlite_where=sa.text("org_id IS NULL"),
            postgresql_where=sa.text("org_id IS NULL"),
        ),
        sa.Index(
            "uq_channel_installations_type_org",
            "channel_type",
            "org_id",
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL"),
        ),
        # Pre-scope webhook-routing key (Slack team_id, etc.), looked up before
        # any org exists — unique on its own, not per-org. NULL until discovered.
        sa.Index(
            "uq_channel_installations_external_account",
            "channel_type",
            "external_account_id",
            unique=True,
            sqlite_where=sa.text("external_account_id IS NOT NULL"),
            postgresql_where=sa.text("external_account_id IS NOT NULL"),
        ),
    )

    channel_type: str = Field(description="Stable adapter name: telegram | slack | discord | whatsapp")
    display_name: str = Field(description="Human-facing channel label for the UI")
    enabled: bool = Field(default=False, description="Whether the adapter should be started")
    status: str = Field(
        default="disconnected",
        description="Last known adapter state: disconnected | active | error",
    )
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    external_account_id: str | None = Field(
        default=None,
        max_length=255,
        description="Platform account id used to route an inbound webhook to its "
        "installation before any org scope exists (Slack team_id, Discord application_id, "
        "WhatsApp phone_number_id, ...); NULL until setup discovers it",
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
    external_thread_key: str = Field(
        default="__default__",
        index=True,
        description="Routing key for uniqueness; external_thread_id when set, else '__default__'",
    )
    display_name: str | None = Field(default=None, description="Human-facing label for the bound chat")
    trigger_rule: str = Field(default="always", description="always | mention_only | regex")
    trigger_pattern: str | None = Field(default=None, description="Regex source when trigger_rule = regex")
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")
    instructions: str | None = Field(
        default=None,
        description="Operator instructions (persona, tone, scope) for turns served through this binding",
    )
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
    # Dedupe key per installation (org_id anchors the index).
    # Prevents org A's redelivered event from deduping against org B's.
    __table_args__ = (
        sa.Index(
            "uq_channel_events_inbound_dedupe_global",
            "channel_type",
            "dedupe_key",
            unique=True,
            sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NULL"),
            postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NULL"),
        ),
        sa.Index(
            "uq_channel_events_inbound_dedupe_org",
            "channel_type",
            "org_id",
            "dedupe_key",
            unique=True,
            sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NOT NULL"),
            postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NOT NULL"),
        ),
    )

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
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
