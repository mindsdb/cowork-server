"""notices: each person's in-app notification inbox

One row per notice: what happened (kind), the subject's name, the details the
app needs to open it, and when it was read. Keyed by the owner's org and user,
both NULL on the desktop.

Revision ID: 7a4e9c2d1b83
Revises: 1d5fe417cc0a
Create Date: 2026-11-13 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "7a4e9c2d1b83"
down_revision: Union[str, Sequence[str], None] = "1d5fe417cc0a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    # Guarded: a database first built from the models already has it.
    if _has_table("notices"):
        return
    op.create_table(
        "notices",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_notices_org_id_user_id_created_at", "notices", ["org_id", "user_id", "created_at"],
    )


def downgrade() -> None:
    if _has_table("notices"):
        op.drop_index("ix_notices_org_id_user_id_created_at", table_name="notices")
        op.drop_table("notices")
