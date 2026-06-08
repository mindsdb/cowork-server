"""channels

Revision ID: b7c1d2e3f4a5
Revises: 93375a6617f4
Create Date: 2026-05-31 23:30:45.328376

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "b8a2f4c6d9e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "channel_installations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("channel_type", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'disconnected'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("channel_type", name="uq_channel_installations_type"),
    )

    op.create_table(
        "channel_bindings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("channel_type", sa.String(64), nullable=False),
        sa.Column("external_group_id", sa.String(255), nullable=False),
        sa.Column("external_thread_id", sa.String(255), nullable=True),
        sa.Column("external_thread_key", sa.String(255), nullable=False, server_default=sa.text("'__default__'")),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("trigger_rule", sa.String(32), nullable=False, server_default=sa.text("'always'")),
        sa.Column("trigger_pattern", sa.Text(), nullable=True),
        sa.Column("anton_project_id", sa.Uuid(), nullable=True),
        sa.Column("anton_conversation_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["anton_project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["anton_conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "channel_type", "external_group_id", "external_thread_key",
            name="uq_channel_bindings_target",
        ),
    )
    op.create_index(op.f("ix_channel_bindings_external_thread_key"), "channel_bindings", ["external_thread_key"], unique=False)

    op.create_table(
        "channel_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("binding_id", sa.Uuid(), nullable=False),
        sa.Column("external_session_key", sa.String(512), nullable=False),
        sa.Column("anton_session_id", sa.String(255), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["binding_id"], ["channel_bindings.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("binding_id", "external_session_key", name="uq_channel_sessions_key"),
    )

    op.create_table(
        "channel_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("channel_type", sa.String(64), nullable=False),
        sa.Column("external_message_id", sa.String(255), nullable=True),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("dedupe_key", sa.String(255), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_channel_events_channel_type"), "channel_events", ["channel_type"], unique=False)
    op.create_index(op.f("ix_channel_events_dedupe_key"), "channel_events", ["dedupe_key"], unique=False)
    op.create_index(
        "uq_channel_events_inbound_dedupe",
        "channel_events",
        ["channel_type", "dedupe_key"],
        unique=True,
        sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL"),
        postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_channel_events_inbound_dedupe", table_name="channel_events")
    op.drop_index(op.f("ix_channel_events_dedupe_key"), table_name="channel_events")
    op.drop_index(op.f("ix_channel_events_channel_type"), table_name="channel_events")
    op.drop_table("channel_events")
    op.drop_table("channel_sessions")
    op.drop_index(op.f("ix_channel_bindings_external_thread_key"), table_name="channel_bindings")
    op.drop_table("channel_bindings")
    op.drop_table("channel_installations")
