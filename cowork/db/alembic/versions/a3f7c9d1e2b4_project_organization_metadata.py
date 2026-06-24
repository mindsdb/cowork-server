"""project organization metadata

Adds server-side project organization columns (pinned, sort_order, archived,
last_selected_at) so list organization persists and follows the user across
devices. Backward compatible: every column has a server default, so existing
rows (and pre-Alembic databases stamped at the initial revision) upgrade
cleanly without a data backfill.

Revision ID: a3f7c9d1e2b4
Revises: fbe3964c2030
Create Date: 2026-06-24 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f7c9d1e2b4"
down_revision: Union[str, Sequence[str], None] = "fbe3964c2030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        if not _has_column("projects", "pinned"):
            batch_op.add_column(
                sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false"))
            )
        if not _has_column("projects", "sort_order"):
            batch_op.add_column(
                sa.Column("sort_order", sa.Integer(), nullable=False, server_default=sa.text("0"))
            )
        if not _has_column("projects", "archived"):
            batch_op.add_column(
                sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.text("false"))
            )
        if not _has_column("projects", "last_selected_at"):
            batch_op.add_column(
                sa.Column("last_selected_at", sa.DateTime(timezone=True), nullable=True)
            )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        if _has_column("projects", "last_selected_at"):
            batch_op.drop_column("last_selected_at")
        if _has_column("projects", "archived"):
            batch_op.drop_column("archived")
        if _has_column("projects", "sort_order"):
            batch_op.drop_column("sort_order")
        if _has_column("projects", "pinned"):
            batch_op.drop_column("pinned")
