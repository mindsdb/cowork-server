from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from cowork.schemas.base import CamelRequest, CamelResponse


class SkillCreateRequest(CamelRequest):
    label: str
    name: str
    description: str | None = None
    when_to_use: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")


class SkillUpdateRequest(CamelRequest):
    label: str | None = None
    name: str | None = None
    description: str | None = None
    when_to_use: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")


class SkillResponse(CamelResponse):
    id: UUID
    label: str
    name: str
    description: str | None
    when_to_use: str | None
    instructions: str = Field(serialization_alias="declarative")
    used: int = 0
    confidence: float = 0.0
    created_at: datetime | None
    modified_at: datetime | None
