"""Model comparisons: remember where Continue put a side's work.

Revision ID: f4a8c2e6b9d1
Revises: e2b7d9c3a1f4
"""

from alembic import op
import sqlalchemy as sa

revision = "f4a8c2e6b9d1"
down_revision = "e2b7d9c3a1f4"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    # A database created before Alembic already has every model column.
    if not _has_column("comparison_sides", "carried_folder"):
        op.add_column("comparison_sides", sa.Column("carried_folder", sa.String(length=255), nullable=True))


def downgrade() -> None:
    if _has_column("comparison_sides", "carried_folder"):
        with op.batch_alter_table("comparison_sides") as batch_op:
            batch_op.drop_column("carried_folder")
