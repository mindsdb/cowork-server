from __future__ import annotations

import sqlalchemy as sa
from sqlmodel import Field

from .base import BaseSQLModel


class Skill(BaseSQLModel, table=True):
    __tablename__ = "skills"

    label: str = Field(max_length=80, unique=True)
    name: str = Field(max_length=255, unique=True)
    description: str | None = Field(default=None, sa_type=sa.Text())
    when_to_use: str | None = Field(default=None, sa_type=sa.Text())
    instructions: str = Field(sa_type=sa.Text())
    # Usage stats. `used` is bumped each time Anton's `recall_skill` tool
    # pulls this skill into working context (see SkillService.record_use,
    # driven by the `response.skill_recalled` stream event). `confidence`
    # is reserved for the recall classifier signal Anton already tracks
    # per skill (StageStats.confidence) but cowork doesn't yet ingest.
    used: int = Field(default=0, sa_column_kwargs={"server_default": sa.text("0")})
    confidence: float = Field(default=0.0, sa_column_kwargs={"server_default": sa.text("0")})
