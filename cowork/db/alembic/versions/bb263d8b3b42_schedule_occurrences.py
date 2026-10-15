"""schedule_occurrences: one claimable row per firing; runs become its attempts

A schedule fired by inserting nothing durable first: the poll checked for a
running run and then started one, so two pollers (or a manual press racing a
cron slot) could both pass the check. An occurrence row is claimed by insert,
unique per (schedule_id, occurrence_key), with at most one active occurrence
per schedule. Each ScheduleRun is now one numbered attempt at an occurrence,
unique per (occurrence_id, attempt). Runs written before this revision keep
NULL in both columns.

Revision ID: bb263d8b3b42
Revises: 3e4b5f7586d3
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "bb263d8b3b42"
down_revision: Union[str, Sequence[str], None] = "3e4b5f7586d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_RUN_FK = "fk_schedule_runs_occurrence_id_schedule_occurrences"
_RUN_UQ = "uq_schedule_runs_occurrence_attempt"
_RUN_IX = "ix_schedule_runs_occurrence_id"


def upgrade() -> None:
    """Upgrade schema.

    Guarded per object: a database first built from the models (pre-Alembic
    desktops) already has the table and columns this revision adds.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "schedule_occurrences" not in inspector.get_table_names():
        op.create_table(
            "schedule_occurrences",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("schedule_id", sa.Uuid(), nullable=False),
            sa.Column("occurrence_key", sa.String(length=128), nullable=False),
            sa.Column("trigger_source", sa.String(), nullable=False),
            sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("attempts", sa.Integer(), nullable=False),
            sa.Column("max_attempts", sa.Integer(), nullable=False),
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedules.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("schedule_id", "occurrence_key", name="uq_schedule_occurrences_key"),
        )
        inspector = sa.inspect(bind)
    occurrence_indexes = {ix["name"] for ix in inspector.get_indexes("schedule_occurrences")}
    if "uq_schedule_occurrences_one_active" not in occurrence_indexes:
        op.create_index(
            "uq_schedule_occurrences_one_active",
            "schedule_occurrences",
            ["schedule_id"],
            unique=True,
            sqlite_where=sa.text("status = 'active'"),
            postgresql_where=sa.text("status = 'active'"),
        )
    if "ix_schedule_occurrences_status_next_attempt" not in occurrence_indexes:
        op.create_index(
            "ix_schedule_occurrences_status_next_attempt",
            "schedule_occurrences",
            ["status", "next_attempt_at"],
        )

    run_columns = {c["name"] for c in inspector.get_columns("schedule_runs")}
    if "occurrence_id" not in run_columns:
        with op.batch_alter_table("schedule_runs", schema=None) as batch:
            batch.add_column(sa.Column("occurrence_id", sa.Uuid(), nullable=True))
            batch.add_column(sa.Column("attempt", sa.Integer(), nullable=True))
            batch.create_foreign_key(
                _RUN_FK, "schedule_occurrences", ["occurrence_id"], ["id"], ondelete="CASCADE"
            )
            batch.create_unique_constraint(_RUN_UQ, ["occurrence_id", "attempt"])
            batch.create_index(_RUN_IX, ["occurrence_id"])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("schedule_runs", schema=None) as batch:
        batch.drop_index(_RUN_IX)
        batch.drop_constraint(_RUN_UQ, type_="unique")
        batch.drop_constraint(_RUN_FK, type_="foreignkey")
        batch.drop_column("attempt")
        batch.drop_column("occurrence_id")
    op.drop_index("ix_schedule_occurrences_status_next_attempt", table_name="schedule_occurrences")
    op.drop_index("uq_schedule_occurrences_one_active", table_name="schedule_occurrences")
    op.drop_table("schedule_occurrences")
