"""Who one skill in an org's store is shared with, in MindsHub mode.

The skill itself is a folder on disk (cowork.models.skill.Skill); its audience
lives here, server-owned, so an agent that can write the skill's folder can't
widen who sees a skill by editing its SKILL.md. Field names mirror MindsHub's
artifact rule (cowork.services.sharing_contract.AudienceField):
``org_allowed``, ``allowed_team_ids`` and ``allowed_user_ids`` (user ids of
people in the org). ``always_on`` is skills' own. One row per skill per org,
keyed by the skill's slug, which is its only stable identity: a rename moves
the row and a delete removes it.

The org and owner are UUID columns; the audience lists are JSON lists of
lowercase UUID strings, as MindsHub stores an artifact rule's. A skill from
before sharing gets its row once
(cowork.services.skill_access.backfill_legacy_access); one with no row is read
by cowork.services.skill_sharing.access_rule_for.
"""

from __future__ import annotations

from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class SkillAccess(BaseSQLModel, table=True):
    __tablename__ = "skill_access"
    __table_args__ = (
        sa.UniqueConstraint("org_id", "skill_slug", name="uq_skill_access_org_skill"),
    )

    org_id: UUID = Field(index=True, description="Owning organization")
    skill_slug: str = Field(max_length=255, description="The skill's folder name in the org store")
    owner_user_id: UUID | None = Field(
        default=None, description="User id of the person who owns the skill"
    )
    org_allowed: bool = Field(default=False, description="Everyone in the organization")
    allowed_team_ids: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(sa.JSON, nullable=False, server_default="[]"),
        description="Teams of the organization",
    )
    allowed_user_ids: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(sa.JSON, nullable=False, server_default="[]"),
        description="User ids of people in the organization",
    )
    always_on: bool = Field(
        default=False,
        description="In every turn's prompt of the org or the team the share reaches",
    )
