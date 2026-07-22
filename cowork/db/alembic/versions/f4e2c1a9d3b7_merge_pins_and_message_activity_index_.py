"""merge pins and message-activity-index heads

Two migrations branched off a1c3e5f7b9d2 independently and both landed on
staging: d2e8f1a4c7b9 (pins per-user/per-org uniqueness) and c1d9f3a7e6b2
(messages last-activity index, ENG-961 / #221). That leaves the graph with two
heads, so ``alembic upgrade head`` errors out. This is an empty merge revision
that rejoins them into a single head — it touches no schema (each branch's own
upgrade already did its work) and rewrites neither migration, so DBs/branches
that already have either one stay consistent. The two changes are independent
(the messages index vs the pins index), so their apply order is irrelevant.

Revision ID: f4e2c1a9d3b7
Revises: c1d9f3a7e6b2, d2e8f1a4c7b9
Create Date: 2026-07-22 19:46:16.050435

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4e2c1a9d3b7'
down_revision: Union[str, Sequence[str], None] = ('c1d9f3a7e6b2', 'd2e8f1a4c7b9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
