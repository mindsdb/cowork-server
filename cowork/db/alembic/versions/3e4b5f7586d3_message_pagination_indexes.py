"""Indexes backing message-history pagination and event lookup.

Revision ID: 3e4b5f7586d3
Revises: e2262a14c001
Create Date: 2026-09-15 00:00:00.000000

Two independent gaps: `message_events.message_id` had no index at all
(every event lookup for a message was an unindexed scan), and cursor-based
keyset pagination over `messages` (ordered by conversation_id, created_at,
seq) needs a composite index that includes `seq` to be an index seek
rather than a scan within same-`created_at` groups. Additive only —
`ix_messages_conversation_created` stays; it's still the index behind
`last_message_at`/`list_conversations`'s `created_at`-only queries.
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
_MESSAGES_SEQ_INDEX = "ix_messages_conversation_created_seq"


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return index_name in {ix["name"] for ix in inspector.get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.create_index(_MESSAGE_EVENTS_INDEX, "message_events", ["message_id"])
    if not _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.create_index(
            _MESSAGES_SEQ_INDEX, "messages", ["conversation_id", "created_at", "seq"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index("messages", _MESSAGES_SEQ_INDEX):
        op.drop_index(_MESSAGES_SEQ_INDEX, table_name="messages")
    if _has_index("message_events", _MESSAGE_EVENTS_INDEX):
        op.drop_index(_MESSAGE_EVENTS_INDEX, table_name="message_events")
