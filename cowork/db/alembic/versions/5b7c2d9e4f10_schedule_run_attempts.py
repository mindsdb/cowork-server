"""Record each run's attempt at its slot and how its execution ended.

A cron run that failed before its agent acted is tried again, up to three
attempts, each its own run row. `for_slot_at` keeps the slot a run was for
after the run gives the slot back, `attempt` numbers the attempts that count,
and `retry_of` points a retry at the slot's first attempt. `outcome` says what
is known about the run's execution, and `execution_started_at` when its turn
was handed to the agent.

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
    op.drop_index("ix_schedule_runs_for_slot", table_name="schedule_runs")
    with op.batch_alter_table("schedule_runs") as batch:
        batch.drop_constraint("fk_schedule_runs_retry_of", type_="foreignkey")
        batch.drop_column("retry_of")
        for name, _ in reversed(_COLUMNS):
            batch.drop_column(name)
