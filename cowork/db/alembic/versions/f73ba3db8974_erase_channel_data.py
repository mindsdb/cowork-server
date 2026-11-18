"""Erase channel data and unpin channel rows from conversations

Channels are gone from the server. This erases what they stored: the bot
credentials and agent choice in `settings`, and every row of the four channel
tables. It also drops the foreign key from `channel_bindings` to
`conversations`, so a binding row cannot block a conversation delete.

The tables and `conversations.channel_type` stay for now: during a rolling
deploy the previous release still reads and writes them. A later release drops
the empty tables. `channel_type` stays as the record of which conversations a
chat app started.

Revision ID: f73ba3db8974
Revises: f6a2c4e8b1d3
Create Date: 2026-11-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f73ba3db8974"
down_revision: Union[str, Sequence[str], None] = "f6a2c4e8b1d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# b7c1d2e3f4a5_channels.py creates this FK unnamed. Postgres auto-names it,
# so the drop takes that name from the inspector; the downgrade restores it
# under the default `<table>_<column>_fkey`. SQLite reports no name, and its
# batch rebuild addresses the key through a naming convention (as d5b28c4a91e7).
_TABLE = "channel_bindings"
_PG_FK = "channel_bindings_anton_conversation_id_fkey"
_NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
_SQLITE_FK = "fk_channel_bindings_anton_conversation_id_conversations"
# Children before parents: channel_sessions points at channel_bindings.
_CHANNEL_TABLES = ("channel_events", "channel_sessions", "channel_bindings", "channel_installations")


def _conversation_fk() -> dict | None:
    """The binding's foreign key to conversations, as the inspector reports
    it, or None. A database created from the models after this revision never
    had the key."""
    for fk in sa.inspect(op.get_bind()).get_foreign_keys(_TABLE):
        if fk["referred_table"] == "conversations" and fk["constrained_columns"] == ["anton_conversation_id"]:
            return fk
    return None


def upgrade() -> None:
    """Upgrade schema. The erased rows and secrets are not kept anywhere."""
    op.execute(sa.text("DELETE FROM settings WHERE key LIKE 'channel.%' OR key = 'channels_harness'"))
    for table in _CHANNEL_TABLES:
        op.execute(sa.text(f"DELETE FROM {table}"))
    fk = _conversation_fk()
    if fk is None:
        return
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(_TABLE, schema=None, naming_convention=_NAMING) as batch:
            batch.drop_constraint(_SQLITE_FK, type_="foreignkey")
    else:
        op.drop_constraint(fk["name"], _TABLE, type_="foreignkey")


def downgrade() -> None:
    """Downgrade schema. Restores the foreign key; the erased rows stay erased.
    A binding the previous release wrote after the upgrade may name a
    conversation deleted since, so those are dropped first."""
    op.execute(sa.text(
        "DELETE FROM channel_sessions WHERE binding_id IN ("
        " SELECT id FROM channel_bindings WHERE anton_conversation_id IS NOT NULL"
        " AND anton_conversation_id NOT IN (SELECT id FROM conversations))"
    ))
    op.execute(sa.text(
        "DELETE FROM channel_bindings WHERE anton_conversation_id IS NOT NULL"
        " AND anton_conversation_id NOT IN (SELECT id FROM conversations)"
    ))
    if _conversation_fk() is not None:
        return
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(_TABLE, schema=None, naming_convention=_NAMING) as batch:
            batch.create_foreign_key(_SQLITE_FK, "conversations", ["anton_conversation_id"], ["id"])
    else:
        op.create_foreign_key(_PG_FK, _TABLE, "conversations", ["anton_conversation_id"], ["id"])
