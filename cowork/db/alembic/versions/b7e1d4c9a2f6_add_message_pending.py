"""add message pending flag

Adds ``messages.pending`` (ENG-1231): the user message is now persisted at turn
start so a mid-turn refresh/reconnect still shows the question. The flag marks it
in-flight — included in the UI view (get_messages) but excluded from replayed LLM
history (get_ordered_messages) until the turn ends — and is cleared on terminal.

Backfills to false, so every existing row is treated as a finalized message.

Revision ID: b7e1d4c9a2f6
Revises: c8e1a4f7b2d9
Create Date: 2026-08-03 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7e1d4c9a2f6"
down_revision: Union[str, Sequence[str], None] = "c8e1a4f7b2d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("messages", "pending"):
        op.add_column(
            "messages",
            sa.Column("pending", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_column("messages", "pending"):
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_column("pending")
