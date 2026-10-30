"""One person's own on/off for one skill (ENG-3787).

A skill's ``enabled`` (its SKILL.md metadata, cowork.models.skill.META_ENABLED)
turns it off for everyone who reads it. This row turns it off for one person
only: Anton stops using the skill in their turns and nobody else's. A turn
uses a skill when it is enabled AND its reader has no row turning it off
(cowork.services.skills.select_turn_skills).

Keyed by the skill's slug, its only stable identity, so a packaged built-in or
a managed skill takes a row too without any skill_access row. A rename moves
a skill's rows and a delete removes them, as skill_access's. A missing row
means on; only ``enabled=False`` rows are kept.

Scoped like skill_access: by organization in hosted org mode (and in a
personal scope that keeps skills per person), with no org on the desktop and
wherever skills are left shared. ``user_id`` is the person's Keycloak id or
sub, or cowork.services.skill_user_settings.LOCAL_USER for the desktop's one
person.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class SkillUserSetting(BaseSQLModel, table=True):
    __tablename__ = "skill_user_setting"
    __table_args__ = (
        sa.UniqueConstraint(
            "org_id", "user_id", "skill_slug", name="uq_skill_user_setting_org_user_skill"
        ),
    )

    org_id: str | None = Field(
        default=None, index=True, max_length=36, description="Owning organization or partition"
    )
    user_id: str = Field(max_length=255, description="The person this setting is theirs")
    skill_slug: str = Field(max_length=255, description="The skill's folder name")
    enabled: bool = Field(default=True, description="False: off in this person's turns")
