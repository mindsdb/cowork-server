"""drop org_appearance: auth now owns the org look (ENG-4020)

The org's look moved to auth (``GET /v1/organizations/current/branding/``
and ``PATCH .../branding/appearance/``), so cowork-server stops storing it.
The earlier revisions that built the table (d4b7e2a9c1f3, 3e78d31dc845,
c3d548ba2bdb, 71332e7a1839) stay as they are: a deployed environment ran
them. Never edit a revision a deployed environment may have run; add a new
one.

Downgrade recreates the table as 71332e7a1839 left it, empty: the look
itself lives in auth.

Revision ID: d16c08651013
Revises: 71332e7a1839
Create Date: 2026-11-06 15:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d16c08651013"
down_revision: Union[str, Sequence[str], None] = "71332e7a1839"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table() -> bool:
    return "org_appearance" in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table():
        return
    index_names = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("org_appearance")}
    if "ix_org_appearance_org_id" in index_names:
        op.drop_index("ix_org_appearance_org_id", table_name="org_appearance")
    op.drop_table("org_appearance")


def downgrade() -> None:
    if _has_table():
        return
    op.create_table(
        "org_appearance",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
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
        sa.Column("sidebar_title_mode", sa.String(length=16), nullable=False, server_default="company"),
        sa.Column("background_light", sa.String(length=7), nullable=True),
        sa.Column("background_dark", sa.String(length=7), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", name="uq_org_appearance_org"),
    )
    op.create_index("ix_org_appearance_org_id", "org_appearance", ["org_id"])
