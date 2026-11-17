"""org_appearance.sidebar_title_mode: company, custom or no title

Which sidebar title the app shows: 'company' (the org's name, the
default), 'custom' (sidebar_title) or 'none' (no title, only the logo).
Added only where the column is missing. A row that already has a
sidebar_title was a custom title before modes existed, so it becomes
'custom'.

Never edit a revision a deployed environment may have run; add a new one.

Revision ID: c3d548ba2bdb
Revises: 5b9e1c7d3a20
Create Date: 2026-11-05 20:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c3d548ba2bdb"
down_revision: Union[str, Sequence[str], None] = "5b9e1c7d3a20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns() -> set[str] | None:
    inspector = sa.inspect(op.get_bind())
    if "org_appearance" not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns("org_appearance")}


def upgrade() -> None:
    columns = _columns()
    if columns is None or "sidebar_title_mode" in columns:
        return
    op.add_column(
        "org_appearance",
        sa.Column("sidebar_title_mode", sa.String(length=16), nullable=False, server_default="company"),
    )
    op.execute(
        sa.text(
            "UPDATE org_appearance SET sidebar_title_mode = 'custom' "
            "WHERE sidebar_title IS NOT NULL AND sidebar_title <> ''"
        )
    )


def downgrade() -> None:
    columns = _columns()
    if columns is not None and "sidebar_title_mode" in columns:
        with op.batch_alter_table("org_appearance") as batch_op:
            batch_op.drop_column("sidebar_title_mode")
