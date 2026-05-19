from __future__ import annotations

import sqlalchemy as sa
from sqlmodel import Field

from .base import BaseSQLModel


class Skill(BaseSQLModel, table=True):
    __tablename__ = "skills"

    name: str = Field(max_length=255, unique=True)
    description: str | None = Field(default=None, sa_type=sa.Text())
    when_to_use: str | None = Field(default=None, sa_type=sa.Text())
    instructions: str = Field(sa_type=sa.Text())
