"""org_appearance: an organization's look in the Cowork app (ENG-4020)

One row per org, keyed by the org's UUID like skill_access. Every look field
is nullable (NULL is the app's stock value); members_can_choose defaults to
true. Only hosted org mode writes it; the desktop's SQLite gets the empty
table and never reads it.

Revision ID: d4b7e2a9c1f3
Revises: c69488788ac7
Create Date: 2026-11-04 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d4b7e2a9c1f3"
down_revision: Union[str, Sequence[str], None] = "c69488788ac7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("org_appearance"):
        return
    op.create_table(
        "org_appearance",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True
        ),
        sa.Column(
            "modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True
        ),
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.Column("accent", sa.String(length=7), nullable=True),
        sa.Column("background", sa.String(length=7), nullable=True),
        sa.Column("font", sa.String(length=32), nullable=True),
        sa.Column("radius", sa.Integer(), nullable=True),
        sa.Column("skin", sa.String(length=32), nullable=True),
        sa.Column("default_theme", sa.String(length=8), nullable=True),
        sa.Column("greeting", sa.String(length=200), nullable=True),
        sa.Column("sidebar_title", sa.String(length=40), nullable=True),
        sa.Column("members_can_choose", sa.Boolean(), nullable=False),
        sa.Column("guidelines", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", name="uq_org_appearance_org"),
    )
    op.create_index("ix_org_appearance_org_id", "org_appearance", ["org_id"])


def downgrade() -> None:
    if _has_table("org_appearance"):
        op.drop_index("ix_org_appearance_org_id", table_name="org_appearance")
        op.drop_table("org_appearance")
