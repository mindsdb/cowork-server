"""add channel_bindings.instructions

Per-chat operator instructions (persona, tone, scope) appended to the
channel-mode system prompt for turns served through that binding.

Revision ID: e8b3c5d7a9f1
Revises: d5f3a8c1e6b2
Create Date: 2026-07-08 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8b3c5d7a9f1"
down_revision: Union[str, Sequence[str], None] = "d5f3a8c1e6b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("channel_bindings", sa.Column("instructions", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("channel_bindings", "instructions")
