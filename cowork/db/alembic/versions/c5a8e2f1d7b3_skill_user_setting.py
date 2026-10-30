"""skill_user_setting: one person's own on/off per skill (ENG-3787)

One row per (org, person, skill slug), only for a skill the person turned off
for themselves; a missing row means on. Keyed by slug so a packaged built-in
takes one too. The desktop's SQLite uses it with no org.

Revision ID: c5a8e2f1d7b3
Revises: b7d1e4a9c2f6
Create Date: 2026-10-29 18:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c5a8e2f1d7b3"
down_revision: Union[str, Sequence[str], None] = "b7d1e4a9c2f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("skill_user_setting"):
        return
    op.create_table(
        "skill_user_setting",
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
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("skill_slug", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "org_id", "user_id", "skill_slug", name="uq_skill_user_setting_org_user_skill"
        ),
    )
    op.create_index("ix_skill_user_setting_org_id", "skill_user_setting", ["org_id"])


def downgrade() -> None:
    if not _has_table("skill_user_setting"):
        return
    op.drop_index("ix_skill_user_setting_org_id", table_name="skill_user_setting")
    op.drop_table("skill_user_setting")
