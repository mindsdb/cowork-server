"""notices: each person's notification inbox

One row per notice a feature posted for a person: what happened, what it is
about, what opening it shows, and when it was read. Scheduled runs are the
first source.

Revision ID: 4b8e2c6d1f37
Revises: f6a2c4e8b1d3
Create Date: 2026-11-16 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "4b8e2c6d1f37"
down_revision: Union[str, Sequence[str], None] = "f6a2c4e8b1d3"
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
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("target", sa.JSON(), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_notices_org_id_user_id_created_at", "notices", ["org_id", "user_id", "created_at"],
    )
    op.create_index("ix_notices_kind_source_id", "notices", ["kind", "source_id"])


def downgrade() -> None:
    if _has_table("notices"):
        op.drop_index("ix_notices_kind_source_id", table_name="notices")
        op.drop_index("ix_notices_org_id_user_id_created_at", table_name="notices")
        op.drop_table("notices")
