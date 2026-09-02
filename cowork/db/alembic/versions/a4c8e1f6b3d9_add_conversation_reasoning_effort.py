"""add conversation reasoning_effort

Revision ID: a4c8e1f6b3d9
Revises: f1a3c9d7e2b5
Create Date: 2026-08-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4c8e1f6b3d9"
# Keep original parent f1a3c9d7e2b5 (already on staging). The channel
# org-scoping migrations (d4a7c2e9f1b3, e5b8d3f0a2c7) must NOT be inserted
# before this; they form a separate migration branch. See ENG-1684.
down_revision: Union[str, Sequence[str], None] = "f1a3c9d7e2b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("conversations", "reasoning_effort"):
        op.add_column("conversations", sa.Column("reasoning_effort", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "reasoning_effort"):
            batch_op.drop_column("reasoning_effort")
