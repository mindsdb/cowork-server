"""Index the latest finished run lookup.

Revision ID: a9e2b6c4d810
Revises: 8c3e1f6a2b47
"""
from alembic import op
import sqlalchemy as sa

revision = "a9e2b6c4d810"
down_revision = "8c3e1f6a2b47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    names = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("schedule_runs")}
    if "ix_schedule_runs_schedule_finished" not in names:
        op.create_index("ix_schedule_runs_schedule_finished", "schedule_runs", ["schedule_id", "finished_at", "started_at"])


def downgrade() -> None:
    op.drop_index("ix_schedule_runs_schedule_finished", table_name="schedule_runs")
