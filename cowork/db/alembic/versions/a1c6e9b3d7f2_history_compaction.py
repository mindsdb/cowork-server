"""conversations.history_summary + history_summary_cutoff_id

Persist anton's compacted summary of a conversation's older turns so the
next turn can replay summary + tail instead of resending full history.
No FK on `history_summary_cutoff_id` — a stale/missing id falls back to
full history rather than blocking message deletion.

Revision ID: a1c6e9b3d7f2
Revises: f7d2b9e4a1c6
Create Date: 2026-07-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1c6e9b3d7f2"
down_revision: Union[str, Sequence[str], None] = "f7d2b9e4a1c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("conversations", "history_summary"):
        op.add_column("conversations", sa.Column("history_summary", sa.Text(), nullable=True))
    if not _has_column("conversations", "history_summary_cutoff_id"):
        op.add_column(
            "conversations", sa.Column("history_summary_cutoff_id", sa.Uuid(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "history_summary_cutoff_id"):
            batch_op.drop_column("history_summary_cutoff_id")
        if _has_column("conversations", "history_summary"):
            batch_op.drop_column("history_summary")
