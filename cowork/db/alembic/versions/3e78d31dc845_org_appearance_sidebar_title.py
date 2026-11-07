"""org_appearance.sidebar_title: the org's sidebar title (ENG-4020)

d4b7e2a9c1f3 was edited after a deployed environment had already run it, so
a database there is at d4b7e2a9c1f3 with an org_appearance table that has no
sidebar_title. This revision adds the column only where it is missing; on a
database whose d4b7e2a9c1f3 already created it, it does nothing.

Rule: never edit a revision a deployed environment (a PR environment
included) may have run. Change the schema with a new revision instead.

Downgrade is a no-op on purpose: it can't tell whether this revision or
d4b7e2a9c1f3 created the column, and d4b7e2a9c1f3's own downgrade drops the
whole table.

Revision ID: 3e78d31dc845
Revises: d4b7e2a9c1f3
Create Date: 2026-11-05 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "3e78d31dc845"
down_revision: Union[str, Sequence[str], None] = "d4b7e2a9c1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _needs_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    if "org_appearance" not in inspector.get_table_names():
        return False
    return "sidebar_title" not in {column["name"] for column in inspector.get_columns("org_appearance")}


def upgrade() -> None:
    if _needs_column():
        op.add_column("org_appearance", sa.Column("sidebar_title", sa.String(length=40), nullable=True))


def downgrade() -> None:
    pass
