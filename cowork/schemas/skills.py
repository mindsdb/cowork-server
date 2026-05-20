from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SkillCreateRequest(BaseModel):
    label: str
    name: str
    description: str | None = None
    when_to_use: str | None = None
    instructions: str


class SkillUpdateRequest(BaseModel):
    label: str | None = None
    name: str | None = None
    description: str | None = None
    when_to_use: str | None = None
    instructions: str | None = None


class SkillResponse(BaseModel):
    id: UUID
    label: str
    name: str
    description: str | None
    when_to_use: str | None
    instructions: str
    created_at: datetime | None
    modified_at: datetime | None

    model_config = {"from_attributes": True}
