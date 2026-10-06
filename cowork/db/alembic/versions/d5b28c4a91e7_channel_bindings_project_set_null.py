"""channel_bindings.anton_project_id ON DELETE SET NULL

The third foreign key into `projects.id` that could 500 a project delete
(ENG-2357). `conversations` and `task_objects` are cleaned up explicitly by the
delete, and `schedules` cascades as of c3a91f7e5d20 -- but nothing released the
channel binding, so deleting a project that a Slack/Telegram route pointed at
failed the same way.

SET NULL rather than CASCADE, deliberately: deleting a project must not delete
someone's channel route. The runtime already reads the column as optional --
`binding.anton_project_id or self._resolve_default_project_id(scoped)` at
runtime.py:357 and :417 -- so a released binding keeps serving on the default
project.

Revision ID: d5b28c4a91e7
Revises: c3a91f7e5d20
Create Date: 2026-09-04 22:05:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d5b28c4a91e7"
down_revision: Union[str, Sequence[str], None] = "c3a91f7e5d20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# b7c1d2e3f4a5_channels.py creates this FK unnamed, so Postgres auto-names it
# `<table>_<column>_fkey` and SQLite's batch rebuild needs a naming convention
# to address the reflected constraint. Same shape as a7f4c2e9d1b8 / c3a91f7e5d20.
_PG_FK = "channel_bindings_anton_project_id_fkey"
_NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
_SQLITE_FK = "fk_channel_bindings_anton_project_id_projects"


def _recreate_fk(ondelete: str | None) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "channel_bindings", schema=None, naming_convention=_NAMING
        ) as batch:
            batch.drop_constraint(_SQLITE_FK, type_="foreignkey")
            batch.create_foreign_key(
                _SQLITE_FK, "projects", ["anton_project_id"], ["id"], ondelete=ondelete
            )
    else:
        op.drop_constraint(_PG_FK, "channel_bindings", type_="foreignkey")
        op.create_foreign_key(
            _PG_FK,
            "channel_bindings",
            "projects",
            ["anton_project_id"],
            ["id"],
            ondelete=ondelete,
        )


def upgrade() -> None:
    """Upgrade schema."""
    _recreate_fk("SET NULL")


def downgrade() -> None:
    """Downgrade schema."""
    _recreate_fk(None)
