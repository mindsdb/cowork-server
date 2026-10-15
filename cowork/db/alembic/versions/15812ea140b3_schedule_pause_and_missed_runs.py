"""schedules: explicit pause, missed one-offs, the failure streak, skipped periods

paused_at records a pause the user asked for, and missed_at a one-off whose
time passed by too much to run on its own; `enabled` alone could not tell
those apart from a one-off that has run. consecutive_failures counts cron
occurrences in a row that settled failed, and auto_paused_at records when
that streak paused the schedule. schedule_skipped_periods summarizes slots
passed over while the app was not running, one row per stretch.

Existing disabled schedules are sorted into those states: a recurring one
was paused by the user; a one-off with runs has run; a one-off counted as
missed, with no runs, was missed; any other one-off was paused.

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

    Guarded per object: a database first built from the models (pre-Alembic
    desktops) already has them.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("schedules")}
    with op.batch_alter_table("schedules", schema=None) as batch:
        if "consecutive_failures" not in existing:
            batch.add_column(sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"))
        if "auto_paused_at" not in existing:
            batch.add_column(sa.Column("auto_paused_at", sa.DateTime(timezone=True), nullable=True))
        if "paused_at" not in existing:
            batch.add_column(sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True))
        if "missed_at" not in existing:
            batch.add_column(sa.Column("missed_at", sa.DateTime(timezone=True), nullable=True))

    if "schedule_skipped_periods" not in inspector.get_table_names():
        op.create_table(
            "schedule_skipped_periods",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("schedule_id", sa.Uuid(), nullable=False),
            sa.Column("first_slot", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_slot", sa.DateTime(timezone=True), nullable=False),
            sa.Column("slot_count", sa.Integer(), nullable=False),
            sa.Column("reason", sa.String(), nullable=False),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedules.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_schedule_skipped_periods_schedule_id", "schedule_skipped_periods", ["schedule_id"],
        )

    when = sa.func.coalesce(sa.column("modified_at"), sa.func.now())
    no_runs = sa.text("NOT EXISTS (SELECT 1 FROM schedule_runs r WHERE r.schedule_id = schedules.id)")
    schedules = sa.table(
        "schedules",
        sa.column("enabled", sa.Boolean()),
        sa.column("cadence", sa.String()),
        sa.column("missed_runs", sa.Integer()),
        sa.column("modified_at", sa.DateTime(timezone=True)),
        sa.column("paused_at", sa.DateTime(timezone=True)),
        sa.column("missed_at", sa.DateTime(timezone=True)),
    )
    disabled = sa.and_(
        schedules.c.enabled == sa.false(),
        schedules.c.paused_at.is_(None),
        schedules.c.missed_at.is_(None),
    )
    op.execute(
        schedules.update()
        .where(disabled, schedules.c.cadence != "once")
        .values(paused_at=when)
    )
    op.execute(
        schedules.update()
        .where(disabled, schedules.c.cadence == "once", no_runs, schedules.c.missed_runs > 0)
        .values(missed_at=when)
    )
    op.execute(
        schedules.update()
        .where(disabled, schedules.c.cadence == "once", no_runs, schedules.c.missed_runs == 0)
        .values(paused_at=when)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_schedule_skipped_periods_schedule_id", table_name="schedule_skipped_periods")
    op.drop_table("schedule_skipped_periods")
    with op.batch_alter_table("schedules", schema=None) as batch:
        batch.drop_column("missed_at")
        batch.drop_column("paused_at")
        batch.drop_column("auto_paused_at")
        batch.drop_column("consecutive_failures")
