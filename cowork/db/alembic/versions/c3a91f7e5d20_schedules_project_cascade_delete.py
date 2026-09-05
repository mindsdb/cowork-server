"""schedules.project_id ON DELETE CASCADE

Deleting a project failed with HTTP 500 on Postgres whenever the project had a
scheduled task (ENG-2357). `delete_project` cascades to the project's
conversations and nothing else, so the final DELETE tripped this foreign key.

Sibling of a7f4c2e9d1b8, which did the same one level down for
schedule_runs.schedule_id. Together the two keys let the whole subtree --
project -> schedules -> runs -- go in one correctly ordered delete. Desktop
SQLite runs without FK enforcement, which is why neither surfaced there.

Revision ID: c3a91f7e5d20
Revises: a7f4c2e9d1b8
Create Date: 2026-09-04 21:15:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c3a91f7e5d20"
down_revision: Union[str, Sequence[str], None] = "a7f4c2e9d1b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Same shape as the sibling migration: the init migration created this FK
# unnamed, so Postgres auto-named it `<table>_<column>_fkey` while SQLite's
# batch rebuild needs a naming convention to address the reflected constraint.
_PG_FK = "schedules_project_id_fkey"
_NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
_SQLITE_FK = "fk_schedules_project_id_projects"


def _recreate_fk(ondelete: str | None) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "schedules", schema=None, naming_convention=_NAMING
        ) as batch:
            batch.drop_constraint(_SQLITE_FK, type_="foreignkey")
            batch.create_foreign_key(
                _SQLITE_FK, "projects", ["project_id"], ["id"], ondelete=ondelete
            )
    else:
        op.drop_constraint(_PG_FK, "schedules", type_="foreignkey")
        op.create_foreign_key(
            _PG_FK, "schedules", "projects", ["project_id"], ["id"], ondelete=ondelete
        )


def upgrade() -> None:
    """Upgrade schema."""
    _recreate_fk("CASCADE")


def downgrade() -> None:
    """Downgrade schema."""
    _recreate_fk(None)
