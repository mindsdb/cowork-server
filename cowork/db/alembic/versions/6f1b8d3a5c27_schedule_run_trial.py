"""schedule_runs: trial, for a run-now that tries a task out

A trial run is started from Run now with ``trial=true``, typically right after
the task is created. It leaves a one-off scheduled for its own time, and
neither the freshness guard nor the failure streak counts it, so trying a task
out never costs it a scheduled run. Existing runs are not trials.

Revision ID: 6f1b8d3a5c27
Revises: a9e2b6c4d810
"""
from alembic import op
import sqlalchemy as sa

revision = "6f1b8d3a5c27"
down_revision = "a9e2b6c4d810"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded: a database first built from the models already has it.
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedule_runs")}
    if "trial" not in existing:
        with op.batch_alter_table("schedule_runs") as batch:
            batch.add_column(sa.Column("trial", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    with op.batch_alter_table("schedule_runs") as batch:
        batch.drop_column("trial")
