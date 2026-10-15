"""schedule audit trail: immutable versions, lifecycle events, due-scan index

schedule_versions holds what a schedule runs, written once per edit to its
prompt, model, project, cadence or time zone; each occurrence references the
version it was claimed against, so its attempts run those settings whatever
is edited meanwhile. Every existing schedule gets its current settings as
version 1. schedule_events records created, edited, paused, resumed and
auto-paused. An index on (enabled, next_run_at) lets the poll fetch only due
schedules. Nothing is pruned: run history is kept.

Revision ID: 4552aea8bb5f
Revises: 15812ea140b3
Create Date: 2026-10-14 00:00:00.000000

"""
import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "4552aea8bb5f"
down_revision: Union[str, Sequence[str], None] = "15812ea140b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OCCURRENCE_FK = "fk_schedule_occurrences_version_id_schedule_versions"


def upgrade() -> None:
    """Upgrade schema.

    Guarded per object: a database first built from the models (pre-Alembic
    desktops) already has them.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "schedule_versions" not in tables:
        op.create_table(
            "schedule_versions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("schedule_id", sa.Uuid(), nullable=False),
            sa.Column("number", sa.Integer(), nullable=False),
            sa.Column("prompt", sa.String(), nullable=False),
            sa.Column("model", sa.String(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("cadence", sa.String(), nullable=False),
            sa.Column("timezone", sa.String(), nullable=False),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedules.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("schedule_id", "number", name="uq_schedule_versions_number"),
        )
        op.create_index("ix_schedule_versions_schedule_id", "schedule_versions", ["schedule_id"])

    schedules = sa.table(
        "schedules",
        sa.column("id", sa.Uuid()),
        sa.column("prompt", sa.String()),
        sa.column("model", sa.String()),
        sa.column("project_id", sa.Uuid()),
        sa.column("cadence", sa.String()),
        sa.column("timezone", sa.String()),
    )
    versions = sa.table(
        "schedule_versions",
        sa.column("id", sa.Uuid()),
        sa.column("schedule_id", sa.Uuid()),
        sa.column("number", sa.Integer()),
        sa.column("prompt", sa.String()),
        sa.column("model", sa.String()),
        sa.column("project_id", sa.Uuid()),
        sa.column("cadence", sa.String()),
        sa.column("timezone", sa.String()),
    )
    versioned = sa.select(versions.c.schedule_id)
    rows = bind.execute(
        sa.select(schedules).where(schedules.c.id.not_in(versioned))
    ).mappings().all()
    if rows:
        op.bulk_insert(versions, [
            {
                "id": uuid.uuid4(), "schedule_id": row["id"], "number": 1, "prompt": row["prompt"],
                "model": row["model"], "project_id": row["project_id"], "cadence": row["cadence"],
                "timezone": row["timezone"] or "UTC",
            }
            for row in rows
        ])

    occurrence_columns = {c["name"] for c in inspector.get_columns("schedule_occurrences")}
    if "version_id" not in occurrence_columns:
        with op.batch_alter_table("schedule_occurrences", schema=None) as batch:
            batch.add_column(sa.Column("version_id", sa.Uuid(), nullable=True))
            batch.create_foreign_key(_OCCURRENCE_FK, "schedule_versions", ["version_id"], ["id"])
            batch.create_index("ix_schedule_occurrences_version_id", ["version_id"])

    if "ix_schedules_enabled_next_run_at" not in {ix["name"] for ix in inspector.get_indexes("schedules")}:
        op.create_index("ix_schedules_enabled_next_run_at", "schedules", ["enabled", "next_run_at"])

    if "schedule_events" not in tables:
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
    op.drop_index("ix_schedules_enabled_next_run_at", table_name="schedules")
    with op.batch_alter_table("schedule_occurrences", schema=None) as batch:
        batch.drop_index("ix_schedule_occurrences_version_id")
        batch.drop_constraint(_OCCURRENCE_FK, type_="foreignkey")
        batch.drop_column("version_id")
    op.drop_index("ix_schedule_versions_schedule_id", table_name="schedule_versions")
    op.drop_table("schedule_versions")
