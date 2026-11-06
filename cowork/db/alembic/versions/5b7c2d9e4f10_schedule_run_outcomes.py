"""Record how each run's execution ended, and allow one running run per schedule.

`outcome` says what is known about the run's execution: whether its agent
acted before it failed, was stopped, or left no record of how it ended.

A partial unique index allows one `running` run per schedule, so a manual run
and a cron run, or two manual runs, can't go at once. Duplicate `running`
rows left by older builds are failed first, keeping each schedule's newest.

Revision ID: 5b7c2d9e4f10
Revises: c69488788ac7
"""
from alembic import op
import sqlalchemy as sa

revision = "5b7c2d9e4f10"
down_revision = "c69488788ac7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded: a database first built from the models already has them.
    inspector = sa.inspect(op.get_bind())
    if "outcome" not in {c["name"] for c in inspector.get_columns("schedule_runs")}:
        with op.batch_alter_table("schedule_runs") as batch:
            batch.add_column(sa.Column("outcome", sa.String(), nullable=True))
    if "execution_claimed_at" not in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedule_runs")}:
        with op.batch_alter_table("schedule_runs") as batch:
            batch.add_column(sa.Column("execution_claimed_at", sa.DateTime(timezone=True), nullable=True))
        # Older rows cannot prove execution never started: their chat may
        # already have been deleted. Preserve their slots conservatively.
        op.execute(sa.text("UPDATE schedule_runs SET execution_claimed_at = started_at"))
    indexes = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("schedule_runs")}
    if "uq_schedule_runs_one_running" not in indexes:
        _fail_extra_running_runs()
        op.create_index(
            "uq_schedule_runs_one_running",
            "schedule_runs",
            ["schedule_id"],
            unique=True,
            sqlite_where=sa.text("status = 'running'"),
            postgresql_where=sa.text("status = 'running'"),
        )


def _fail_extra_running_runs() -> None:
    """Fail every `running` run but the newest of its schedule, so the
    one-running index can be built. Such duplicates could only come from the
    race the index now prevents."""
    bind = op.get_bind()
    rows = bind.execute(sa.text(
        "SELECT id, schedule_id FROM schedule_runs WHERE status = 'running' "
        "ORDER BY schedule_id, started_at DESC"
    )).all()
    seen: set = set()
    for run_id, schedule_id in rows:
        if schedule_id in seen:
            bind.execute(
                sa.text(
                    "UPDATE schedule_runs SET status = 'failed', slot_at = NULL, lease_expires_at = NULL, "
                    "finished_at = CURRENT_TIMESTAMP, "
                    "error = 'Run ended by an upgrade: another run of this task was already going.' "
                    "WHERE id = :id"
                ),
                {"id": run_id},
            )
        seen.add(schedule_id)


def downgrade() -> None:
    op.drop_index("uq_schedule_runs_one_running", table_name="schedule_runs")
    with op.batch_alter_table("schedule_runs") as batch:
        batch.drop_column("outcome")
        batch.drop_column("execution_claimed_at")
