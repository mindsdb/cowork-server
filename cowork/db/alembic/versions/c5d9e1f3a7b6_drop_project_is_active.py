"""drop project is_active

Removes the ``projects.is_active`` column. ``is_active`` was a second,
redundant notion of the "active" project that the task/conversation creation
path never actually read — it could silently disagree with the client's
selection and with ``last_selected_at``, causing tasks to land in the wrong
project. The unified model keeps a single server-side signal,
``last_selected_at`` (added in revision a3f7c9d1e2b4), used only as the
fallback for headless/scheduled runs.

Backward compatible: dropping the column is safe because nothing reads it any
more. The downgrade re-adds it with its original server default so the schema
round-trips, but the value is no longer maintained by application code.

Revision ID: c5d9e1f3a7b6
Revises: a3f7c9d1e2b4
Create Date: 2026-06-24 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c5d9e1f3a7b6"
down_revision: Union[str, Sequence[str], None] = "a3f7c9d1e2b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        if _has_column("projects", "is_active"):
            batch_op.drop_column("is_active")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        if not _has_column("projects", "is_active"):
            batch_op.add_column(
                sa.Column(
                    "is_active",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.text("true"),
                )
            )
