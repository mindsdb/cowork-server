"""Claim each cron slot with its run row, and lease the run to its process.

`slot_at` names the cron slot a run fires for, and a unique index on
(schedule_id, slot_at) makes the run's insert the claim, so replicas that poll
together start one run. `lease_expires_at` is renewed while the run lives, so
any replica can tell a run whose process died from one still running.

Revision ID: c41f7a9e2b58
Revises: a3734c9e5210
"""
from alembic import op
import sqlalchemy as sa

revision = "c41f7a9e2b58"
down_revision = "a3734c9e5210"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedule_runs")}
    with op.batch_alter_table("schedule_runs") as batch:
        if "slot_at" not in columns:
            batch.add_column(sa.Column("slot_at", sa.DateTime(timezone=True), nullable=True))
        if "lease_expires_at" not in columns:
            batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    indexes = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("schedule_runs")}
    if "uq_schedule_runs_slot" not in indexes:
        op.create_index(
            "uq_schedule_runs_slot",
            "schedule_runs",
            ["schedule_id", "slot_at"],
            unique=True,
            sqlite_where=sa.text("slot_at IS NOT NULL"),
            postgresql_where=sa.text("slot_at IS NOT NULL"),
        )


def downgrade() -> None:
    op.drop_index("uq_schedule_runs_slot", table_name="schedule_runs")
    with op.batch_alter_table("schedule_runs") as batch:
        batch.drop_column("lease_expires_at")
        batch.drop_column("slot_at")
