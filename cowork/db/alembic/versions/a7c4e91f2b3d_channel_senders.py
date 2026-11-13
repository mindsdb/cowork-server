"""channel_senders, channel_pairing_codes and the bindings' capability switches

Who may talk to the bot in each external chat, the one-time codes that let a
sender in, and four per-binding switches for what an allowed sender's turns
may use. No sender row is written here, so every existing chat starts with
nobody allowed. The switches default to false through a server default,
which SQLite needs to add a NOT NULL column.

Revision ID: a7c4e91f2b3d
Revises: 1d5fe417cc0a
Create Date: 2026-11-13 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "a7c4e91f2b3d"
down_revision: Union[str, Sequence[str], None] = "1d5fe417cc0a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SWITCHES = ("allow_scratchpad", "allow_connectors", "allow_memory_writes", "allow_file_replies")
_SENDER_KEY = ["channel_type", "external_group_id", "external_sender_id"]


def _has_table(table_name: str) -> bool:
    """Whether the table exists on the migration's connection."""
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _binding_columns() -> set[str]:
    """The column names channel_bindings has now."""
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns("channel_bindings")}


def upgrade() -> None:
    """Add the sender and pairing code tables and the binding switches.

    Each part is guarded: a database first built from the models already has it.
    """
    existing = _binding_columns()
    for name in _SWITCHES:
        if name not in existing:
            op.add_column(
                "channel_bindings",
                sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.false()),
            )

    if not _has_table("channel_senders"):
        op.create_table(
            "channel_senders",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("channel_type", sa.String(), nullable=False),
            sa.Column("external_group_id", sa.String(), nullable=False),
            sa.Column("external_sender_id", sa.String(length=255), nullable=False),
            sa.Column("is_group", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("display_name", sa.String(length=64), nullable=True),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("org_id", sa.String(length=36), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_channel_senders_org_id", "channel_senders", ["org_id"])
        op.create_index(
            "uq_channel_senders_chat_sender_global",
            "channel_senders",
            _SENDER_KEY,
            unique=True,
            sqlite_where=sa.text("org_id IS NULL"),
            postgresql_where=sa.text("org_id IS NULL"),
        )
        op.create_index(
            "uq_channel_senders_chat_sender_org",
            "channel_senders",
            [*_SENDER_KEY, "org_id"],
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL"),
        )

    if not _has_table("channel_pairing_codes"):
        op.create_table(
            "channel_pairing_codes",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("channel_type", sa.String(), nullable=False),
            sa.Column("code_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("org_id", sa.String(length=36), nullable=True),
            sa.Column("created_by", sa.String(length=36), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("code_hash"),
        )
        op.create_index("ix_channel_pairing_codes_channel_type", "channel_pairing_codes", ["channel_type"])
        op.create_index("ix_channel_pairing_codes_org_id", "channel_pairing_codes", ["org_id"])


def downgrade() -> None:
    """Drop the two tables and the binding switches."""
    if _has_table("channel_pairing_codes"):
        op.drop_index("ix_channel_pairing_codes_org_id", table_name="channel_pairing_codes")
        op.drop_index("ix_channel_pairing_codes_channel_type", table_name="channel_pairing_codes")
        op.drop_table("channel_pairing_codes")
    if _has_table("channel_senders"):
        op.drop_index("uq_channel_senders_chat_sender_org", table_name="channel_senders")
        op.drop_index("uq_channel_senders_chat_sender_global", table_name="channel_senders")
        op.drop_index("ix_channel_senders_org_id", table_name="channel_senders")
        op.drop_table("channel_senders")
    existing = _binding_columns()
    with op.batch_alter_table("channel_bindings") as batch:
        for name in _SWITCHES:
            if name in existing:
                batch.drop_column(name)
