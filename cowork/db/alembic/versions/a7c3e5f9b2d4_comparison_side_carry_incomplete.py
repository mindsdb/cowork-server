"""Model comparisons: remember that Continue left some of a side's work behind.

Revision ID: a7c3e5f9b2d4
Revises: f4a8c2e6b9d1
"""

from alembic import op
import sqlalchemy as sa

revision = "a7c3e5f9b2d4"
down_revision = "f4a8c2e6b9d1"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    # A database created before Alembic already has every model column.
    if not _has_column("comparison_sides", "carry_incomplete"):
        op.add_column(
            "comparison_sides",
            sa.Column("carry_incomplete", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    if _has_column("comparison_sides", "carry_incomplete"):
        with op.batch_alter_table("comparison_sides") as batch_op:
            batch_op.drop_column("carry_incomplete")
