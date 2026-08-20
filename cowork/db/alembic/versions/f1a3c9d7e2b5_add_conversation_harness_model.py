"""add conversation harness and model

Revision ID: f1a3c9d7e2b5
Revises: c3f8a2b6d1e4
Create Date: 2026-08-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a3c9d7e2b5"
down_revision: Union[str, Sequence[str], None] = "c3f8a2b6d1e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("conversations", "harness"):
        op.add_column("conversations", sa.Column("harness", sa.String(), nullable=True))
    if not _has_column("conversations", "model"):
        op.add_column("conversations", sa.Column("model", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "model"):
            batch_op.drop_column("model")
        if _has_column("conversations", "harness"):
            batch_op.drop_column("harness")
