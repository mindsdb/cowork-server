"""schedules.requires_browser: defer browser-dependent runs while the app sleeps

A schedule that works live tabs (the morning digest) must not fire into a
dead bridge — its due slot defers and catches up on launch, honestly.

Revision ID: c7d4e1f3a8b5
Revises: b4e9d3a2c6f1
Create Date: 2026-07-23 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7d4e1f3a8b5"
down_revision: Union[str, Sequence[str], None] = "b4e9d3a2c6f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("schedules", "requires_browser"):
        op.add_column(
            "schedules",
            sa.Column("requires_browser", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_column("schedules", "requires_browser"):
        with op.batch_alter_table("schedules") as batch_op:
            batch_op.drop_column("requires_browser")
