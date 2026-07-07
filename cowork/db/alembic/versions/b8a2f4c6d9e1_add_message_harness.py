"""add message harness

Revision ID: b8a2f4c6d9e1
Revises: 93375a6617f4
Create Date: 2026-06-04 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8a2f4c6d9e1"
down_revision: str | Sequence[str] | None = "93375a6617f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("messages", "harness"):
        op.add_column("messages", sa.Column("harness", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    if _has_column("messages", "harness"):
        with op.batch_alter_table("messages") as batch_op:
            batch_op.drop_column("harness")
