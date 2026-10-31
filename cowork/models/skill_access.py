"""Who one skill in a hosted org's store is shared with (ENG-3787).

The skill itself is a folder on disk (cowork.models.skill.Skill); its access
lives here, server-owned, so an agent that can write the skills volume can't
widen who sees a skill by editing its SKILL.md. Field names mirror auth's
``ArtifactAccess``: ``org_allowed``, ``allowed_team_ids``,
``allowed_user_ids`` (Keycloak user ids of people in the org). ``always_on``
is skills' own. One row per skill per org, keyed by the skill's slug, which is
its only stable identity: a rename moves the row and a delete removes it.

Hosted org mode only. A skill with no row is read by
cowork.services.skill_sharing.access_rule_for.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class SkillAccess(BaseSQLModel, table=True):
    __tablename__ = "skill_access"
    __table_args__ = (
        sa.UniqueConstraint("org_id", "skill_slug", name="uq_skill_access_org_skill"),
    )

    org_id: str = Field(index=True, max_length=36, description="Owning organization")
    skill_slug: str = Field(max_length=255, description="The skill's folder name in the org store")
    owner_user_id: str | None = Field(
        default=None, max_length=36, description="Keycloak id of the person who owns the skill"
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
        description="Keycloak ids of people in the organization",
    )
    always_on: bool = Field(
        default=False,
        description="In every turn's prompt of the people an org-wide or team share reaches",
    )
