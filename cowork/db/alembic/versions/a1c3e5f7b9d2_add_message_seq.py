"""add message seq

Revision ID: a1c3e5f7b9d2
Revises: f7d2b9e4a1c6
Create Date: 2026-07-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1c3e5f7b9d2"
down_revision: Union[str, Sequence[str], None] = "f7d2b9e4a1c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("messages", "seq"):
        op.add_column(
            "messages",
            sa.Column("seq", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_column("messages", "seq"):
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_column("seq")
