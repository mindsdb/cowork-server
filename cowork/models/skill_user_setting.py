"""One person's own on/off for one skill.

A skill's ``enabled`` (its SKILL.md metadata, cowork.models.skill.META_ENABLED)
turns it off for everyone who reads it. This row turns it off for one person
only: Anton stops using the skill in their turns and nobody else's. A turn
uses a skill when it is enabled AND its reader has no row turning it off
(cowork.services.skills.select_turn_skills).

Keyed by the skill's slug, its only stable identity, so a packaged built-in or
a managed skill takes a row too without any skill_access row. A rename moves
a skill's rows and a delete removes them, as skill_access's. A missing row
means on; only ``enabled=False`` rows are kept.

Scoped like skill_access: by organization in MindsHub mode (and in
single-tenant mode, where skills are kept per person), with no org on the
desktop and wherever skills are left shared. ``user_id`` is the person's user
id, or LOCAL_PERSON_ID for the desktop's one person.
"""

from __future__ import annotations

from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel

# The desktop's one person, and the person of a deployment's operator view,
# which name nobody in their scope: the nil UUID, so the column stays a
# non-null UUID and both unique indexes below keep their rows unique.
LOCAL_PERSON_ID = UUID(int=0)


class SkillUserSetting(BaseSQLModel, table=True):
    __tablename__ = "skill_user_setting"
    __table_args__ = (
        sa.UniqueConstraint(
            "org_id", "user_id", "skill_slug", name="uq_skill_user_setting_org_user_skill"
        ),
        # NULLs are distinct in the constraint above, so rows with no org
        # (the desktop, and wherever skills are left shared) take their own
        # partial unique index.
        sa.Index(
            "uq_skill_user_setting_user_skill_no_org",
            "user_id",
            "skill_slug",
            unique=True,
            postgresql_where=sa.text("org_id IS NULL"),
            sqlite_where=sa.text("org_id IS NULL"),
        ),
    )

    org_id: UUID | None = Field(
        default=None, index=True, description="Owning organization or partition"
    )
    user_id: UUID = Field(description="The person this setting is theirs")
    skill_slug: str = Field(max_length=255, description="The skill's folder name")
    enabled: bool = Field(default=True, description="False: off in this person's turns")
