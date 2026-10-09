"""add conversation last_turn_ended_by

Revision ID: a7c3e9f1b2d4
Revises: 3e4b5f7586d3
Create Date: 2026-10-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c3e9f1b2d4"
down_revision: Union[str, Sequence[str], None] = "3e4b5f7586d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("conversations", "last_turn_ended_by"):
        op.add_column(
            "conversations", sa.Column("last_turn_ended_by", sa.String(length=32), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "last_turn_ended_by"):
            batch_op.drop_column("last_turn_ended_by")
