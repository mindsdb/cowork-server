"""Indexes and seq backfill backing message-history pagination.

Revision ID: 3e4b5f7586d3
Revises: e2262a14c001
Create Date: 2026-09-15 00:00:00.000000

Three gaps, all on the message-history read path:

`message_events.message_id` had no index at all, so every event lookup for
a message was an unindexed scan.

`messages.seq` was added by a1c3e5f7b9d2 with ``server_default="0"`` and no
backfill, so every row written before that migration shares seq 0. Cursor
pagination keys on seq, and with every legacy row tied at 0 the tiebreak
falls through to a random UUID — a paginated read of a legacy conversation
comes back grouped by role instead of in history order, disagreeing with
the unbounded read of the same rows. The backfill below renumbers each
affected conversation by the order those rows already sort in
(``created_at``, ``seq``, user-before-assistant, ``id`` — ConversationService's
_MESSAGE_ORDER), so seq becomes a dense per-conversation ordinal and the two
read paths agree permanently.

Cursor pagination then needs ``(conversation_id, seq)`` to be an index seek.
An earlier draft of this migration created ``(conversation_id, created_at,
seq)``, which cannot serve that query at all: created_at sits between the
equality column and the sort column, so the planner sorts the whole
conversation into a temp B-tree for every page.

`ix_messages_conversation_created` stays; it's still the index behind
`last_message_at`/`list_conversations`'s created_at-only queries.

Not reversible in the data sense: downgrade drops the indexes, but the
backfilled seq values stay. They are a normalization of an ordering the
rows already had, so leaving them is harmless — restoring the original
all-zero column would reintroduce the bug this fixes.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3e4b5f7586d3"
down_revision: Union[str, Sequence[str], None] = "e2262a14c001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_MESSAGE_EVENTS_INDEX = "ix_message_events_message_id"
_MESSAGES_SEQ_INDEX = "ix_messages_conversation_seq"
# An earlier draft of this same revision created a (conversation_id,
# created_at, seq) index under this name. Dropped on upgrade so a checkout
# that already ran that draft locally doesn't keep an index nothing uses.
_SUPERSEDED_SEQ_INDEX = "ix_messages_conversation_created_seq"


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return index_name in {ix["name"] for ix in inspector.get_indexes(table_name)}


def _backfill_message_seq() -> None:
    """Renumber seq per conversation for conversations whose rows are tied.

    Only conversations that actually carry duplicate seq values are touched,
    so a database written entirely after a1c3e5f7b9d2 is a no-op.
    """
    bind = op.get_bind()
    tied = bind.execute(
        sa.text(
            """
            SELECT conversation_id
            FROM messages
            GROUP BY conversation_id
            HAVING COUNT(*) <> COUNT(DISTINCT seq)
            """
        )
    ).scalars().all()
    for conversation_id in tied:
        rows = bind.execute(
            sa.text(
                """
                SELECT id
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at, seq,
                         CASE WHEN role = 'user' THEN 0 ELSE 1 END, id
                """
            ),
            {"cid": conversation_id},
        ).scalars().all()
        for position, message_id in enumerate(rows):
            bind.execute(
                sa.text("UPDATE messages SET seq = :seq WHERE id = :id"),
                {"seq": position, "id": message_id},
            )


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.create_index(_MESSAGE_EVENTS_INDEX, "message_events", ["message_id"])
    _backfill_message_seq()
    if _has_index("messages", _SUPERSEDED_SEQ_INDEX):
        op.drop_index(_SUPERSEDED_SEQ_INDEX, table_name="messages")
    if not _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.create_index(_MESSAGES_SEQ_INDEX, "messages", ["conversation_id", "seq"])


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.drop_index(_MESSAGES_SEQ_INDEX, table_name="messages")
    if _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.drop_index(_MESSAGE_EVENTS_INDEX, table_name="message_events")
