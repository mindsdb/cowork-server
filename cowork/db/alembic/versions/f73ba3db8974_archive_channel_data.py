"""Keep the removed channels' data as an archive, without their bot passwords

Channels are gone from the server. Their rows stay where they are, as a record
nobody reads: the four channel tables, their non-secret settings and
`conversations.channel_type`. What goes is every bot password: every
`channel.*` row in `settings` except the few fields the plugins marked
non-secret, so a secret field this list never named is erased too. It also
drops the foreign key from `channel_bindings` to
`conversations`, so a binding row cannot block a conversation delete.

The tables stay: during a rolling deploy the previous release still reads and
writes them, and a later release decides how to keep their rows before it
drops them.

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


# The `channel.<type>.<field>` keys services/channels.py stored unencrypted,
# for the fields the last plugins declared `secret=False`. Every other
# `channel.` key is erased: deny by default, so a secret field an older
# release had, or one this list missed, cannot survive.
_NON_SECRET_KEYS = (
    "channel.discord.public_key",
    "channel.telegram.bot_username",
    "channel.whatsapp.phone_number_id",
    "channel.whatsapp.verify_token",
)

# b7c1d2e3f4a5_channels.py creates this FK unnamed. Postgres auto-names it,
# so the drop takes that name from the inspector; the downgrade restores it
# under the default `<table>_<column>_fkey`. SQLite reports no name, and its
# batch rebuild addresses the key through a naming convention (as d5b28c4a91e7).
_TABLE = "channel_bindings"
_PG_FK = "channel_bindings_anton_conversation_id_fkey"
_NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
_SQLITE_FK = "fk_channel_bindings_anton_conversation_id_conversations"


def _conversation_fk() -> dict | None:
    """The binding's foreign key to conversations, as the inspector reports
    it, or None. A database created from the models after this revision never
    had the key."""
    for fk in sa.inspect(op.get_bind()).get_foreign_keys(_TABLE):
        if fk["referred_table"] == "conversations" and fk["constrained_columns"] == ["anton_conversation_id"]:
            return fk
    return None


def upgrade() -> None:
    """Upgrade schema. The erased passwords are not kept anywhere."""
    op.execute(
        sa.text("DELETE FROM settings WHERE key LIKE 'channel.%' AND key NOT IN :kept").bindparams(
            sa.bindparam("kept", value=list(_NON_SECRET_KEYS), expanding=True)
        )
    )
    fk = _conversation_fk()
    if fk is None:
        return
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(_TABLE, schema=None, naming_convention=_NAMING) as batch:
            batch.drop_constraint(_SQLITE_FK, type_="foreignkey")
    else:
        op.drop_constraint(fk["name"], _TABLE, type_="foreignkey")


def downgrade() -> None:
    """Downgrade schema. Restores the foreign key; the erased passwords stay
    erased. A binding may name a conversation deleted after the upgrade, so
    that link is cleared first; the binding row itself is kept."""
    op.execute(sa.text(
        "UPDATE channel_bindings SET anton_conversation_id = NULL WHERE anton_conversation_id IS NOT NULL"
        " AND anton_conversation_id NOT IN (SELECT id FROM conversations)"
    ))
    if _conversation_fk() is not None:
        return
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(_TABLE, schema=None, naming_convention=_NAMING) as batch:
            batch.create_foreign_key(_SQLITE_FK, "conversations", ["anton_conversation_id"], ["id"])
    else:
        op.create_foreign_key(_PG_FK, _TABLE, "conversations", ["anton_conversation_id"], ["id"])
