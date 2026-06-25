from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from anton.core.tools.skill_format import AgentSkill
from sqlmodel import Field

from .base import BaseSQLModel


META_DISPLAY_NAME = "display_name"
META_CREATED_AT = "created_at"
META_ENABLED = "enabled"
META_PROJECTS = "projects"


class Skill(AgentSkill):
    """Cowork's alias for the agentskills.io in-memory model.
    """

    @property
    def id(self) -> str:
        """The slug; alias kept for call sites that expect ``label``."""
        return self.name

    @property
    def label(self) -> str:
        """The slug; alias kept for call sites that expect ``label``."""
        return self.name

    @property
    def display_name(self) -> str:
        return self.metadata.get(META_DISPLAY_NAME) or self.name

    @property
    def enabled(self) -> bool:
        # Default-on: only an explicit "false" disables the skill.
        return self.metadata.get(META_ENABLED, "true").strip().lower() != "false"

    @property
    def projects(self) -> list[str]:
        """Project ids the skill is distributed to (empty = none)."""
        raw = self.metadata.get(META_PROJECTS, "")
        return [p for p in (s.strip() for s in raw.split(",")) if p]

    @property
    def created_at(self) -> datetime | None:
        raw = self.metadata.get(META_CREATED_AT)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None


class SkillLegacy(BaseSQLModel, table=True):
    """Archived pre-file skills table.

    Read-only; kept only as a backup and as the source for the one-time file migration.
    """
    __tablename__ = "skills"

    label: str = Field(max_length=80, unique=True)
    name: str = Field(max_length=255, unique=True)
    description: str | None = Field(default=None, sa_type=sa.Text())
    when_to_use: str | None = Field(default=None, sa_type=sa.Text())
    instructions: str = Field(sa_type=sa.Text())
