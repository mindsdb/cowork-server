"""schedules: explicit pause, missed one-offs, the failure streak

paused_at records a pause the user asked for, and missed_at a one-off whose
time passed by too much to run on its own; `enabled` alone could not tell
those apart from a one-off that has run. consecutive_failures counts cron
slots in a row that settled failed, and auto_paused_at records when that
streak paused the schedule.

Existing disabled schedules are sorted into those states: a recurring one
was paused by the user; a one-off with cron runs has run; a one-off counted as
missed, with no cron runs, was missed; any other one-off was paused. Manual
runs did not disable one-offs on the old server, so they do not establish done.

Revision ID: 8c3e1f6a2b47
Revises: 5b7c2d9e4f10
"""
from alembic import op
import sqlalchemy as sa

revision = "8c3e1f6a2b47"
down_revision = "5b7c2d9e4f10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded per object: a database first built from the models already has them.
    inspector = sa.inspect(op.get_bind())
    existing = {c["name"] for c in inspector.get_columns("schedules")}
    with op.batch_alter_table("schedules") as batch:
        if "consecutive_failures" not in existing:
            batch.add_column(sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"))
        if "auto_paused_at" not in existing:
            batch.add_column(sa.Column("auto_paused_at", sa.DateTime(timezone=True), nullable=True))
        if "paused_at" not in existing:
            batch.add_column(sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True))
        if "missed_at" not in existing:
            batch.add_column(sa.Column("missed_at", sa.DateTime(timezone=True), nullable=True))

    when = sa.func.coalesce(sa.column("modified_at"), sa.func.now())
    no_cron_runs = sa.text(
        "NOT EXISTS (SELECT 1 FROM schedule_runs r WHERE r.schedule_id = schedules.id AND NOT r.is_manual)"
    )
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
        .where(disabled, schedules.c.cadence == "once", no_cron_runs, schedules.c.missed_runs > 0)
        .values(missed_at=when)
    )
    op.execute(
        schedules.update()
        .where(disabled, schedules.c.cadence == "once", no_cron_runs, schedules.c.missed_runs == 0)
        .values(paused_at=when)
    )


def downgrade() -> None:
    with op.batch_alter_table("schedules") as batch:
        batch.drop_column("missed_at")
        batch.drop_column("paused_at")
        batch.drop_column("auto_paused_at")
        batch.drop_column("consecutive_failures")
