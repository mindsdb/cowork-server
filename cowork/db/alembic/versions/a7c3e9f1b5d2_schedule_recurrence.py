"""schedules: keep which days a weekly or monthly task runs on

A weekly task can run on several days, and a monthly one on a day of the
month or on a weekday of the month, such as the first Monday. Existing rows
keep NULL: a weekly task without a rule runs on its next run's weekday, as
before.

Revision ID: a7c3e9f1b5d2
Revises: d4e8a1c6b3f2
Create Date: 2026-11-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c3e9f1b5d2"
down_revision: Union[str, Sequence[str], None] = "d4e8a1c6b3f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = "schedules"
_COLUMN = "recurrence"


def _has_column() -> bool:
    """Whether the column exists. A database created from the models before
    Alembic already has it."""
    return _COLUMN in {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column():
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema. Every rule is dropped, so weekly tasks run on their
    next run's weekday. Monthly tasks are paused: an older server can't step
    them, and running them on another cadence would run them far more often
    than the user chose."""
    if not _has_column():
        return
    op.execute(sa.text(
        "UPDATE schedules SET enabled = :off, paused_at = COALESCE(paused_at, CURRENT_TIMESTAMP) "
        "WHERE cadence = 'monthly'"
    ).bindparams(off=False))
    op.drop_column(_TABLE, _COLUMN)
