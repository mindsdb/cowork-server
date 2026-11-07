"""org_appearance.background_light and background_dark (ENG-4020)

The org's look keeps a background for the light theme and one for the dark
theme, as the app's custom theme does (bgLight, bgDark). Each column is
added only where it is missing, and a row's existing single ``background``
is copied into whichever of the two is still empty. ``background`` stays:
the server mirrors it on every write (light, else dark) for clients and
servers from before the split.

Never edit a revision a deployed environment may have run; add a new one.

Revision ID: 71332e7a1839
Revises: c3d548ba2bdb
Create Date: 2026-11-06 09:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "71332e7a1839"
down_revision: Union[str, Sequence[str], None] = "c3d548ba2bdb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_COLUMNS = ("background_light", "background_dark")


def _columns() -> set[str] | None:
    inspector = sa.inspect(op.get_bind())
    if "org_appearance" not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns("org_appearance")}


def upgrade() -> None:
    columns = _columns()
    if columns is None:
        return
    for name in _NEW_COLUMNS:
        if name not in columns:
            op.add_column("org_appearance", sa.Column(name, sa.String(length=7), nullable=True))
        op.execute(
            sa.text(
                f"UPDATE org_appearance SET {name} = background "
                f"WHERE {name} IS NULL AND background IS NOT NULL"
            )
        )


def downgrade() -> None:
    columns = _columns()
    present = [name for name in _NEW_COLUMNS if columns is not None and name in columns]
    if present:
        with op.batch_alter_table("org_appearance") as batch_op:
            for name in present:
                batch_op.drop_column(name)
