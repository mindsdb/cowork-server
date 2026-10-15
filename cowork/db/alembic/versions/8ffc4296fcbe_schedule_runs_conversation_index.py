"""schedule_runs: index conversation_id

A conversation list says which of its conversations are a schedule's runs by
looking their ids up in schedule_runs. The index keeps that one lookup per
page, however long the schedules' history.

Revision ID: 8ffc4296fcbe
Revises: 4552aea8bb5f
Create Date: 2026-10-15 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "8ffc4296fcbe"
down_revision: Union[str, Sequence[str], None] = "4552aea8bb5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "ix_schedule_runs_conversation_id"


def upgrade() -> None:
    """Upgrade schema.

    Guarded: a database first built from the models (pre-Alembic desktops)
    already has it.
    """
    if _INDEX not in {ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes("schedule_runs")}:
        op.create_index(_INDEX, "schedule_runs", ["conversation_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(_INDEX, table_name="schedule_runs")
