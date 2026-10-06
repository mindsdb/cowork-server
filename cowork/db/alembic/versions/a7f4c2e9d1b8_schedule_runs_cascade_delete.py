"""schedule_runs.schedule_id ON DELETE CASCADE

Deleting a schedule failed with HTTP 500 on Postgres: the service removed the
child schedule_runs by hand, but with no relationship the unit of work emitted
the parent DELETE first and tripped this foreign key. Give the key ON DELETE
CASCADE so the database removes the runs with their schedule (and cleans up any
run the scheduler inserts concurrently). Desktop SQLite runs without FK
enforcement, which is why the bug never surfaced there.

Revision ID: a7f4c2e9d1b8
Revises: e2b7d4f9a1c3
Create Date: 2026-09-04 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a7f4c2e9d1b8"
down_revision: Union[str, Sequence[str], None] = "e2b7d4f9a1c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The init migration created this FK unnamed. Postgres auto-named it
# `<table>_<column>_fkey`; SQLite's batch rebuild needs a naming convention to
# address the reflected constraint at all.
_PG_FK = "schedule_runs_schedule_id_fkey"
_NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
_SQLITE_FK = "fk_schedule_runs_schedule_id_schedules"


def _recreate_fk(ondelete: str | None) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "schedule_runs", schema=None, naming_convention=_NAMING
        ) as batch:
            batch.drop_constraint(_SQLITE_FK, type_="foreignkey")
            batch.create_foreign_key(
                _SQLITE_FK, "schedules", ["schedule_id"], ["id"], ondelete=ondelete
            )
    else:
        op.drop_constraint(_PG_FK, "schedule_runs", type_="foreignkey")
        op.create_foreign_key(
            _PG_FK, "schedule_runs", "schedules", ["schedule_id"], ["id"], ondelete=ondelete
        )


def upgrade() -> None:
    """Upgrade schema."""
    _recreate_fk("CASCADE")


def downgrade() -> None:
    """Downgrade schema."""
    _recreate_fk(None)
