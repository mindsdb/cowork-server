"""schedule reliability: failure + missed-run policy

Adds failure-policy + missed-run-policy state to ``schedules`` and an
idempotency key + attempt count to ``schedule_runs``:

  schedules.missed_run_policy    — skip | run_once | catch_up
  schedules.consecutive_failures — failure streak (drives auto-pause)
  schedules.health               — ok | failing | paused
  schedule_runs.idempotency_key  — occurrence-stable key (dedup guard)
  schedule_runs.attempts         — in-process retries the run took

A unique constraint on (schedule_id, idempotency_key) is the DB-level guard
that stops a scheduled occurrence from double-firing.

Revision ID: c2f4a6b8d0e1
Revises: fbe3964c2030
Create Date: 2026-06-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c2f4a6b8d0e1"
down_revision: Union[str, Sequence[str], None] = "fbe3964c2030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _columns(table_name: str) -> set[str]:
    if not _has_table(table_name):
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _unique_constraints(table_name: str) -> set[str]:
    if not _has_table(table_name):
        return set()
    return {
        uc.get("name")
        for uc in sa.inspect(op.get_bind()).get_unique_constraints(table_name)
    }


def upgrade() -> None:
    """Upgrade schema."""
    schedule_cols = _columns("schedules")
    if schedule_cols:
        with op.batch_alter_table("schedules") as batch_op:
            if "missed_run_policy" not in schedule_cols:
                batch_op.add_column(
                    sa.Column(
                        "missed_run_policy",
                        sa.String(length=16),
                        nullable=False,
                        server_default="skip",
                    )
                )
            if "consecutive_failures" not in schedule_cols:
                batch_op.add_column(
                    sa.Column(
                        "consecutive_failures",
                        sa.Integer(),
                        nullable=False,
                        server_default=sa.text("0"),
                    )
                )
            if "health" not in schedule_cols:
                batch_op.add_column(
                    sa.Column(
                        "health",
                        sa.String(length=16),
                        nullable=False,
                        server_default="ok",
                    )
                )

    run_cols = _columns("schedule_runs")
    if run_cols:
        with op.batch_alter_table("schedule_runs") as batch_op:
            if "idempotency_key" not in run_cols:
                batch_op.add_column(
                    sa.Column("idempotency_key", sa.String(length=255), nullable=True)
                )
            if "attempts" not in run_cols:
                batch_op.add_column(
                    sa.Column(
                        "attempts",
                        sa.Integer(),
                        nullable=False,
                        server_default=sa.text("1"),
                    )
                )
            if "uq_schedule_run_idempotency" not in _unique_constraints("schedule_runs"):
                batch_op.create_unique_constraint(
                    "uq_schedule_run_idempotency",
                    ["schedule_id", "idempotency_key"],
                )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("schedule_runs"):
        run_cols = _columns("schedule_runs")
        with op.batch_alter_table("schedule_runs") as batch_op:
            if "uq_schedule_run_idempotency" in _unique_constraints("schedule_runs"):
                batch_op.drop_constraint("uq_schedule_run_idempotency", type_="unique")
            if "attempts" in run_cols:
                batch_op.drop_column("attempts")
            if "idempotency_key" in run_cols:
                batch_op.drop_column("idempotency_key")

    if _has_table("schedules"):
        schedule_cols = _columns("schedules")
        with op.batch_alter_table("schedules") as batch_op:
            if "health" in schedule_cols:
                batch_op.drop_column("health")
            if "consecutive_failures" in schedule_cols:
                batch_op.drop_column("consecutive_failures")
            if "missed_run_policy" in schedule_cols:
                batch_op.drop_column("missed_run_policy")
