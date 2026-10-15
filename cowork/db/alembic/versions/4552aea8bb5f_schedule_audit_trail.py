"""schedule audit trail: spec versions, lifecycle events, due-scan index

Each schedule carries a spec_version that edits to what it runs bump, and each
run records the version it ran. schedule_events records created, edited,
paused, resumed and auto-paused. An index on (enabled, next_run_at) lets the
poll fetch only due schedules.

Revision ID: 4552aea8bb5f
Revises: 15812ea140b3
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "4552aea8bb5f"
down_revision: Union[str, Sequence[str], None] = "15812ea140b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Guarded per object: a database first built from the models (pre-Alembic
    desktops) already has them.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    schedule_columns = {c["name"] for c in inspector.get_columns("schedules")}
    if "spec_version" not in schedule_columns:
        with op.batch_alter_table("schedules", schema=None) as batch:
            batch.add_column(sa.Column("spec_version", sa.Integer(), nullable=False, server_default="1"))
    if "ix_schedules_enabled_next_run_at" not in {ix["name"] for ix in inspector.get_indexes("schedules")}:
        op.create_index("ix_schedules_enabled_next_run_at", "schedules", ["enabled", "next_run_at"])

    run_columns = {c["name"] for c in inspector.get_columns("schedule_runs")}
    if "spec_version" not in run_columns:
        with op.batch_alter_table("schedule_runs", schema=None) as batch:
            batch.add_column(sa.Column("spec_version", sa.Integer(), nullable=True))

    if "schedule_events" not in inspector.get_table_names():
        op.create_table(
            "schedule_events",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("schedule_id", sa.Uuid(), nullable=False),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("actor", sa.String(length=64), nullable=True),
            sa.Column("detail", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedules.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_schedule_events_schedule_id", "schedule_events", ["schedule_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_schedule_events_schedule_id", table_name="schedule_events")
    op.drop_table("schedule_events")
    with op.batch_alter_table("schedule_runs", schema=None) as batch:
        batch.drop_column("spec_version")
    op.drop_index("ix_schedules_enabled_next_run_at", table_name="schedules")
    with op.batch_alter_table("schedules", schema=None) as batch:
        batch.drop_column("spec_version")
