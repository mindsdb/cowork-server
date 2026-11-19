"""skill_access and skill_user_setting: who a skill is shared with, and one person's off

skill_access: who a skill in MindsHub mode is shared with, one row per skill
per org, keyed by the skill's slug. Names mirror MindsHub's artifact rule
(org_allowed, allowed_team_ids, allowed_user_ids); always_on is skills' own.
Only MindsHub mode writes it; the desktop's SQLite gets the empty table and
never reads it.

skill_user_setting: one person's own on/off per skill, one row per (org,
person, skill slug), only for a skill the person turned off for themselves; a
missing row means on. Keyed by slug so a packaged built-in takes one too. The
desktop's SQLite uses it with no org and the desktop's one person
(LOCAL_PERSON_ID, the nil UUID), so a partial unique index keeps those rows
unique too (NULLs are distinct in the org-keyed constraint).

Every org and person id is a UUID column. The audience lists stay JSON lists
of lowercase UUID strings, as MindsHub stores an artifact rule's.

One revision for both tables. It keeps the id of the second of the two
revisions it replaces (b7d1e4a9c2f6 skill_access, then c5a8e2f1d7b3
skill_user_setting), so a database that ran both is already at this head.

Revision ID: c5a8e2f1d7b3
Revises: a3734c9e5210
Create Date: 2026-10-29 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c5a8e2f1d7b3"
down_revision: Union[str, Sequence[str], None] = "a3734c9e5210"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NO_ORG = sa.text("org_id IS NULL")


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _timestamps() -> list[sa.Column]:
    return [
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
    ]


def upgrade() -> None:
    if not _has_table("skill_access"):
        op.create_table(
            "skill_access",
            *_timestamps(),
            sa.Column("org_id", sa.Uuid(), nullable=False),
            sa.Column("skill_slug", sa.String(length=255), nullable=False),
            sa.Column("owner_user_id", sa.Uuid(), nullable=True),
            sa.Column("org_allowed", sa.Boolean(), nullable=False),
            sa.Column("allowed_team_ids", sa.JSON(), server_default="[]", nullable=False),
            sa.Column("allowed_user_ids", sa.JSON(), server_default="[]", nullable=False),
            sa.Column("always_on", sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("org_id", "skill_slug", name="uq_skill_access_org_skill"),
        )
        op.create_index("ix_skill_access_org_id", "skill_access", ["org_id"])
    if not _has_table("skill_user_setting"):
        op.create_table(
            "skill_user_setting",
            *_timestamps(),
            sa.Column("org_id", sa.Uuid(), nullable=True),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("skill_slug", sa.String(length=255), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "org_id", "user_id", "skill_slug", name="uq_skill_user_setting_org_user_skill"
            ),
        )
        op.create_index("ix_skill_user_setting_org_id", "skill_user_setting", ["org_id"])
        op.create_index(
            "uq_skill_user_setting_user_skill_no_org",
            "skill_user_setting",
            ["user_id", "skill_slug"],
            unique=True,
            postgresql_where=_NO_ORG,
            sqlite_where=_NO_ORG,
        )


def downgrade() -> None:
    # The settings rows that mark an org's skills from before sharing as
    # given their skill_access rows (cowork.services.skill_access's
    # BACKFILL_SETTING_KEY, kept as a literal so this revision stands alone).
    # They go with the rows they vouch for: left behind, a later upgrade would
    # find the marker over an empty table and never backfill again, and every
    # skill from before sharing would read as its creator's alone.
    if _has_table("settings"):
        op.execute(sa.text("DELETE FROM settings WHERE key = '_skill_access_backfill_v1'"))
    if _has_table("skill_user_setting"):
        op.drop_table("skill_user_setting")
    if _has_table("skill_access"):
        op.drop_index("ix_skill_access_org_id", table_name="skill_access")
        op.drop_table("skill_access")
