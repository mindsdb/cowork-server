"""index messages(conversation_id, created_at) for last-activity derivation

Revision ID: c1d9f3a7e6b2
Revises: a1c3e5f7b9d2
Create Date: 2026-07-22 00:00:00.000000

Supports ENG-961: a conversation's "last activity" is derived as
MAX(message.created_at) per conversation (the stored conversation.modified_at
never moves on a turn). This composite index turns that aggregate into an
index seek rather than a per-conversation scan.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1d9f3a7e6b2"
down_revision: Union[str, Sequence[str], None] = "a1c3e5f7b9d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX_NAME = "ix_messages_conversation_created"


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return index_name in {ix["name"] for ix in inspector.get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_index("messages", _INDEX_NAME):
        op.create_index(_INDEX_NAME, "messages", ["conversation_id", "created_at"])


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index("messages", _INDEX_NAME):
        op.drop_index(_INDEX_NAME, table_name="messages")
