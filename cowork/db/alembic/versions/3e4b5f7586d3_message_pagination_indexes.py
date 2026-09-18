"""Indexes and seq backfill backing message-history pagination.

Revision ID: 3e4b5f7586d3
Revises: e2262a14c001
Create Date: 2026-09-15 00:00:00.000000

Three gaps, all on the message-history read path:

`message_events.message_id` had no index at all, so every event lookup for
a message was an unindexed scan.

`messages.seq` was added by a1c3e5f7b9d2 with ``server_default="0"`` and no
backfill, so every row written before that migration shares seq 0. Every read
path orders by seq (ConversationService._MESSAGE_ORDER), and with every legacy
row tied at 0 the tiebreak falls through to a random UUID — a legacy
conversation comes back grouped by role instead of in history order. The
backfill below renumbers each affected conversation into a dense ordinal so
seq is a total order per conversation.

It orders by ``seq`` before ``created_at`` on purpose. A conversation that
spans this migration holds legacy rows (all seq 0) alongside newer rows with
real seq values, and on SQLite those newer rows carry created_at in two
different formats — microsecond-precision where the writer passed a datetime,
second-precision where the column default fired — which SQLite compares
lexicographically. Leading on created_at there puts an answer before the
question it replies to, and this migration would bake that into seq
permanently, because afterwards the conversation's seqs are distinct and the
``HAVING`` filter never selects it again. Leading on seq keeps the legacy
block (all 0) first, ordered among itself by created_at, which is safe
because every pre-seq row was written by the same path and shares one format.

Deploy notes, and an unclosed race. Every pod runs migrations itself at boot
(`cowork/server.py`'s lifespan -> `run_dev_setup` in `cowork/dev_setup.py` ->
`run_schema_migrations`), so on a rolling deploy this runs while old pods are
still serving turns, and `run_schema_migrations` wraps the whole chain in one
`engine.begin()`.

A message written to a legacy conversation while the backfill is running can
still land in the wrong place: `ConversationService._next_seq` computes
max(seq)+1 in its own SELECT, so a writer that read the column before the
renumbering commits inserts seq=1 against rows that are now 0..N-1, and the
`HAVING` filter never selects that conversation again to repair it. The window
is the gap between one conversation's SELECT and its UPDATE.

A table-level LOCK does NOT close this, and an earlier version of this
migration that took SHARE ROW EXCLUSIVE made it worse: that mode does not
conflict with the writer's plain SELECT, so the stale read still happened and
the INSERT merely queued behind the lock, turning an insert that would have
committed before the backfill (and so been renumbered correctly) into a
stranded one. Closing it properly needs the seq allocation itself serialized —
a row lock in `_next_seq` plus a unique `(conversation_id, seq)` so a collision
raises instead of silently reordering — or the backfill run with writers
actually stopped rather than merely blocked. Both are larger than this change.

Run this with writers quiesced if the deployment allows it. On a large
`messages` table the index build also blocks writes for its duration.

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
                ORDER BY seq, created_at,
                         CASE WHEN role = 'user' THEN 0 ELSE 1 END, id
                """
            ),
            {"cid": conversation_id},
        ).scalars().all()
        updates = [{"new_seq": position, "row_id": row_id} for position, row_id in enumerate(rows)]
        if updates:
            bind.execute(
                sa.text("UPDATE messages SET seq = :new_seq WHERE id = :row_id"), updates
            )


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.create_index(_MESSAGE_EVENTS_INDEX, "message_events", ["message_id"])
    _backfill_message_seq()
    if not _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.create_index(_MESSAGES_SEQ_INDEX, "messages", ["conversation_id", "seq"])


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.drop_index(_MESSAGES_SEQ_INDEX, table_name="messages")
    if _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.drop_index(_MESSAGE_EVENTS_INDEX, table_name="message_events")
