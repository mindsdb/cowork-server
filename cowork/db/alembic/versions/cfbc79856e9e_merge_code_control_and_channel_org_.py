"""merge code-control and channel-org-scoping heads

Two migrations branched off a4c8e1f6b3d9 independently and both landed on
staging: e2c4a6f8b1d3 (tenant-namespaced Code control-plane records) and
d4a7c2e9f1b3 -> e5b8d3f0a2c7 (channel installation and event org scoping).
That leaves the graph with two heads, so ``alembic upgrade head`` errors out
and every code path that calls it fails, including server startup. This is an
empty merge revision that rejoins them into a single head -- it touches no
schema (each branch's own upgrade already did its work) and rewrites neither
branch's down_revision, so databases already stamped at either head stay
consistent. The two branches touch disjoint tables (code_control_records vs
channel_installations and channel_events), so their apply order is irrelevant.

Revision ID: cfbc79856e9e
Revises: e2c4a6f8b1d3, e5b8d3f0a2c7
Create Date: 2026-09-02 22:47:12.685176

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cfbc79856e9e'
down_revision: Union[str, Sequence[str], None] = ('e2c4a6f8b1d3', 'e5b8d3f0a2c7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
