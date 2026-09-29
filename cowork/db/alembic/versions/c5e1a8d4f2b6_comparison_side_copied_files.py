"""Model comparisons: remember what a side's project copy held.

Revision ID: c5e1a8d4f2b6
Revises: a7c4e9b2d1f3
"""

from alembic import op
import sqlalchemy as sa

revision = "c5e1a8d4f2b6"
down_revision = "a7c4e9b2d1f3"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    # A database created before Alembic already has every model column.
    if not _has_column("comparison_sides", "copied_files"):
        op.add_column("comparison_sides", sa.Column("copied_files", sa.JSON(), nullable=True))


def downgrade() -> None:
    if _has_column("comparison_sides", "copied_files"):
        with op.batch_alter_table("comparison_sides") as batch_op:
            batch_op.drop_column("copied_files")
