"""skill_access: who a hosted org skill is shared with (ENG-3787)

One row per skill per org, keyed by the skill's slug. Names mirror auth's
ArtifactAccess (org_allowed, allowed_team_ids, allowed_user_ids); always_on is
skills' own. Only hosted org mode writes it; the desktop's SQLite gets the
empty table and never reads it.

Revision ID: b7d1e4a9c2f6
Revises: a3734c9e5210
Create Date: 2026-10-29 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b7d1e4a9c2f6"
down_revision: Union[str, Sequence[str], None] = "a3734c9e5210"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("skill_access"):
        return
    op.create_table(
        "skill_access",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column(
            "modified_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("org_id", sa.String(length=36), nullable=False),
        sa.Column("skill_slug", sa.String(length=255), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=True),
        sa.Column("org_allowed", sa.Boolean(), nullable=False),
        sa.Column("allowed_team_ids", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("allowed_user_ids", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("always_on", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", "skill_slug", name="uq_skill_access_org_skill"),
    )
    op.create_index("ix_skill_access_org_id", "skill_access", ["org_id"])


def downgrade() -> None:
    if not _has_table("skill_access"):
        return
    op.drop_index("ix_skill_access_org_id", table_name="skill_access")
    op.drop_table("skill_access")
