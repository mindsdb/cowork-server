"""comparison_sides: clear the usage snapshots that count plan usage as cost

The history list serves each side's ``usage_snapshot`` as it is. A snapshot
stored before the usage service answered with paid usage only holds the list
price and tokens of calls a plan window or the Free allowance covered, so the
history shows plan usage as cost. A snapshot is rewritten only when a read of
its side succeeds, and a side whose read now fails keeps its snapshot for good.
Clearing every snapshot makes the history show no figure for a side until its
next read stores one from paid usage only.

Revision ID: c33e840ac409
Revises: a7c3e9f1b5d2
Create Date: 2026-11-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c33e840ac409"
down_revision: Union[str, Sequence[str], None] = "a7c3e9f1b5d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Clear every stored snapshot. The usage service's next read of a side
    stores it again, from paid usage only."""
    op.execute(sa.text("UPDATE comparison_sides SET usage_snapshot = NULL WHERE usage_snapshot IS NOT NULL"))


def downgrade() -> None:
    """Nothing to undo: the cleared snapshots held figures the history must
    not show, and the next read of a side stores a new one."""
