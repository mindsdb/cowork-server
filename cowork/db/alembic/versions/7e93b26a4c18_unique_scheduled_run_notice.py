"""Keep one notice per scheduled run, including concurrent reapers.

Revision ID: 7e93b26a4c18
Revises: 4b8e2c6d1f37
"""
import sqlalchemy as sa
from alembic import op

revision = "7e93b26a4c18"
down_revision = "4b8e2c6d1f37"
branch_labels = None
depends_on = None

_PREDICATE = "kind IN ('scheduled_run.failed', 'scheduled_run.stopped', 'scheduled_run.succeeded')"
_INDEX = "uq_notices_scheduled_run_source"


def upgrade() -> None:
    if _INDEX in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("notices")}:
        return
    # Keep the earliest copy of each run. A source identifies a run across
    # all three ending kinds, not just within one kind.
    op.execute(sa.text(f"""
        DELETE FROM notices WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY source_id ORDER BY created_at, id
                ) AS duplicate_rank
                FROM notices WHERE source_id IS NOT NULL AND {_PREDICATE}
            ) AS ranked WHERE duplicate_rank > 1
        )
    """))
    op.create_index(
        _INDEX, "notices", ["source_id"], unique=True,
        sqlite_where=sa.text(_PREDICATE), postgresql_where=sa.text(_PREDICATE),
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="notices")
