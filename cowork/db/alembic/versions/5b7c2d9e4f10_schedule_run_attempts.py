"""Record each run's attempt at its slot and how its execution ended.

A cron run that failed before its agent acted is tried again, up to three
attempts, each its own run row. `for_slot_at` keeps the slot a run was for
after the run gives the slot back, `attempt` numbers the attempts that count,
and `retry_of` points a retry at the slot's first attempt. `outcome` says what
is known about the run's execution, and `execution_started_at` when its turn
was handed to the agent.

Revision ID: 5b7c2d9e4f10
Revises: 2a366c1f4b9f
"""
from alembic import op
import sqlalchemy as sa

revision = "5b7c2d9e4f10"
down_revision = "2a366c1f4b9f"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("for_slot_at", lambda: sa.Column("for_slot_at", sa.DateTime(timezone=True), nullable=True)),
    ("attempt", lambda: sa.Column("attempt", sa.Integer(), nullable=True)),
    ("outcome", lambda: sa.Column("outcome", sa.String(), nullable=True)),
    ("execution_started_at", lambda: sa.Column("execution_started_at", sa.DateTime(timezone=True), nullable=True)),
)


def upgrade() -> None:
    # Guarded: a database first built from the models already has them.
    inspector = sa.inspect(op.get_bind())
    columns = {c["name"] for c in inspector.get_columns("schedule_runs")}
    with op.batch_alter_table("schedule_runs") as batch:
        for name, column in _COLUMNS:
            if name not in columns:
                batch.add_column(column())
        if "retry_of" not in columns:
            batch.add_column(sa.Column("retry_of", sa.Uuid(), nullable=True))
            batch.create_foreign_key(
                "fk_schedule_runs_retry_of", "schedule_runs", ["retry_of"], ["id"], ondelete="SET NULL",
            )
    indexes = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("schedule_runs")}
    if "ix_schedule_runs_for_slot" not in indexes:
        op.create_index("ix_schedule_runs_for_slot", "schedule_runs", ["schedule_id", "for_slot_at"])


def downgrade() -> None:
    op.drop_index("ix_schedule_runs_for_slot", table_name="schedule_runs")
    with op.batch_alter_table("schedule_runs") as batch:
        batch.drop_constraint("fk_schedule_runs_retry_of", type_="foreignkey")
        batch.drop_column("retry_of")
        for name, _ in reversed(_COLUMNS):
            batch.drop_column(name)
