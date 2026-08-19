"""schedule_runs: one in-flight run per schedule (single-flight)

The scheduler selects due schedules and then dispatches them, and the manual
run-now endpoint dispatches independently. Neither claim was atomic, so
overlapping cron ticks — or a manual click racing a cron tick — could create
two concurrent 'running' rows for the same schedule and double-run it
(ENG-1733 #3/#4). Add a partial unique index enforcing at most one 'running'
row per schedule; the INSERT then becomes the atomic claim
(ScheduleRunService.try_claim_run turns the resulting IntegrityError into a
"lost the race, skip" signal). Finished runs (success/failed/cancelled) are
unconstrained and accumulate freely.

Pre-existing duplicate 'running' rows would block index creation, so first
resolve them: keep the newest running run per schedule and mark the rest
failed (boot-time reap_orphaned_runs would clear them anyway).

Revision ID: a7e4c2f1b9d3
Revises: f1a3c9d7e2b5
Create Date: 2026-08-19 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7e4c2f1b9d3"
down_revision: Union[str, Sequence[str], None] = "f1a3c9d7e2b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "uq_schedule_runs_one_active"
_RUNNING_WHERE = sa.text("status = 'running'")

# Keep the newest running run per schedule; fail the older duplicates so the
# unique index can be created. Correlated ORDER BY ... LIMIT 1 works on both
# PostgreSQL and SQLite.
_DEDUPE_SQL = sa.text(
    """
    UPDATE schedule_runs
    SET status = 'failed',
        error = 'Superseded by single-flight migration (duplicate in-flight run).',
        finished_at = CURRENT_TIMESTAMP
    WHERE status = 'running'
      AND id <> (
        SELECT r2.id FROM schedule_runs r2
        WHERE r2.schedule_id = schedule_runs.schedule_id
          AND r2.status = 'running'
        ORDER BY r2.started_at DESC
        LIMIT 1
      )
    """
)


def upgrade() -> None:
    """Upgrade schema."""
    op.get_bind().execute(_DEDUPE_SQL)
    op.create_index(
        INDEX_NAME,
        "schedule_runs",
        ["schedule_id"],
        unique=True,
        postgresql_where=_RUNNING_WHERE,
        sqlite_where=_RUNNING_WHERE,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX_NAME, table_name="schedule_runs")
