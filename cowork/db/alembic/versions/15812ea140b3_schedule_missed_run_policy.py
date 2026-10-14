"""schedules: missed-run policy and the failure streak behind auto-pause

missed_run_policy chooses what a recurring schedule does with slots that
passed while it could not run. consecutive_failures counts cron occurrences
in a row that settled failed, and auto_paused_at records when that streak
paused the schedule.

Revision ID: 15812ea140b3
Revises: bb263d8b3b42
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "15812ea140b3"
down_revision: Union[str, Sequence[str], None] = "bb263d8b3b42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Guarded per column: a database first built from the models (pre-Alembic
    desktops) already has them.
    """
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedules")}
    with op.batch_alter_table("schedules", schema=None) as batch:
        if "missed_run_policy" not in existing:
            batch.add_column(
                sa.Column("missed_run_policy", sa.String(), nullable=False, server_default="run_latest_once")
            )
        if "consecutive_failures" not in existing:
            batch.add_column(sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"))
        if "auto_paused_at" not in existing:
            batch.add_column(sa.Column("auto_paused_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("schedules", schema=None) as batch:
        batch.drop_column("auto_paused_at")
        batch.drop_column("consecutive_failures")
        batch.drop_column("missed_run_policy")
