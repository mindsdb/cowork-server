"""add conversation verifier latch

Revision ID: b7e3f2a19c48
Revises: a4c8e1f6b3d9
Create Date: 2026-09-01 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7e3f2a19c48"
down_revision: Union[str, Sequence[str], None] = "a4c8e1f6b3d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("conversations", "verifier_latch"):
        op.add_column("conversations", sa.Column("verifier_latch", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "verifier_latch"):
            batch_op.drop_column("verifier_latch")
