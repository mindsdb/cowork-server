from datetime import datetime

from pydantic import Field

from cowork.schemas.base import CamelRequest, CamelResponse


class SkillCreateRequest(CamelRequest):
    label: str
    name: str
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")


class SkillUpdateRequest(CamelRequest):
    label: str | None = None
    name: str | None = None
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")


class SkillResponse(CamelResponse):
    id: str  # the slug
    label: str
    name: str
    description: str | None
    instructions: str = Field(serialization_alias="declarative")
    created_at: datetime | None

