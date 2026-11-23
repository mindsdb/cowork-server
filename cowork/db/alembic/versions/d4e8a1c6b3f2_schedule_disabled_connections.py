"""schedules: disabled_connections

A scheduled task can leave out connections it doesn't need, so an unattended
run can't use them: the same {engine, name} list a chat turn mutes them with.

Revision ID: d4e8a1c6b3f2
Revises: fabea0eac5e2
Create Date: 2026-11-23 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d4e8a1c6b3f2"
down_revision: Union[str, Sequence[str], None] = "fabea0eac5e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Guarded: a database first built from the models already has it.
    """
    if "disabled_connections" not in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedules")}:
        op.add_column("schedules", sa.Column("disabled_connections", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema. Tasks go back to using every connection."""
    if "disabled_connections" in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedules")}:
        op.drop_column("schedules", "disabled_connections")
