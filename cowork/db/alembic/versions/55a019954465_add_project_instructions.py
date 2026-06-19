"""add project instructions

Revision ID: 55a019954465
Revises: c4e7a1b9d2f0
Create Date: 2026-06-19 15:28:23.246712

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55a019954465'
down_revision: Union[str, Sequence[str], None] = 'c4e7a1b9d2f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("projects", sa.Column("instructions", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("projects", "instructions")
