"""merge artifact-platform-server with staging migration heads

Revision ID: fbe3964c2030
Revises: a1b2c3d4e5f6, d5f3a8c1e6b2
Create Date: 2026-06-23 17:51:54.039001

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fbe3964c2030'
down_revision: Union[str, Sequence[str], None] = ('a1b2c3d4e5f6', 'd5f3a8c1e6b2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
