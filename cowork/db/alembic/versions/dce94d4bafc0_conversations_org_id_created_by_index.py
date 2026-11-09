"""index conversations(org_id, created_by) for a person's own conversations

Revision ID: dce94d4bafc0
Revises: 6f1b8d3a5c27
Create Date: 2026-11-06 00:00:00.000000

In org mode every artifact request reads the ids of the conversations the
caller created in the organization (ConversationService.owned_ids). With an
index on org_id alone, that read visits every conversation of the
organization to keep the caller's. This composite index reads the caller's
rows alone.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "dce94d4bafc0"
down_revision: Union[str, Sequence[str], None] = "6f1b8d3a5c27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX_NAME = "ix_conversations_org_id_created_by"


def _has_index(*, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return index_name in {ix["name"] for ix in inspector.get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    # Bootstrap can create SQLModel metadata, this index included, before
    # Alembic adopts the schema.
    if not _has_index(table_name="conversations", index_name=_INDEX_NAME):
        op.create_index(_INDEX_NAME, "conversations", ["org_id", "created_by"])


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index(table_name="conversations", index_name=_INDEX_NAME):
        op.drop_index(_INDEX_NAME, table_name="conversations")
