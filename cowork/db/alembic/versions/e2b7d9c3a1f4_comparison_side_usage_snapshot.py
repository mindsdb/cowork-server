"""Model comparisons: keep each side's last-read cost for the history list.

Revision ID: e2b7d9c3a1f4
Revises: c5e1a8d4f2b6
"""

from alembic import op
import sqlalchemy as sa

revision = "e2b7d9c3a1f4"
down_revision = "c5e1a8d4f2b6"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    # A database created before Alembic already has every model column.
    if not _has_column("comparison_sides", "usage_snapshot"):
        op.add_column("comparison_sides", sa.Column("usage_snapshot", sa.JSON(), nullable=True))


def downgrade() -> None:
    if _has_column("comparison_sides", "usage_snapshot"):
        with op.batch_alter_table("comparison_sides") as batch_op:
            batch_op.drop_column("usage_snapshot")
