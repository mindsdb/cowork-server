from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SkillCreateRequest(BaseModel):
    name: str
    description: str | None = None
    when_to_use: str | None = None
    instructions: str


class SkillUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    when_to_use: str | None = None
    instructions: str | None = None


class SkillResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    when_to_use: str | None
    instructions: str
    created_at: datetime | None
    modified_at: datetime | None

    model_config = {"from_attributes": True}
