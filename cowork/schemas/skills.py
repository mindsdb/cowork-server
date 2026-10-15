from datetime import datetime

from pydantic import AliasChoices, BaseModel, Field

from cowork.schemas.base import CamelRequest, CamelResponse
from cowork.schemas.shared_resources import ResourceAttribution, SkillCapabilities


class SkillCreateRequest(CamelRequest):
    label: str
    name: str | None = None
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")
    enabled: bool | None = None
    projects: list[str] | None = None


class SkillUpdateRequest(CamelRequest):
    label: str | None = None
    name: str | None = None
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")
    enabled: bool | None = None
    projects: list[str] | None = None


class SkillResponse(CamelResponse):
    id: str  # the slug
    label: str
    # get "name" (is the human-readable display name) from skill.display_name
    name: str = Field(validation_alias=AliasChoices("display_name", "name"))
    description: str | None
    instructions: str = Field(
        validation_alias=AliasChoices("instructions", "declarative"),
        serialization_alias="declarative",
    )
    created_at: datetime | None
    updated_at: datetime | None
    enabled: bool
    projects: list[str]
    attribution: ResourceAttribution
    is_builtin: bool
    # In the shared-skill registry: one someone else shares with the caller,
    # read-only for them, or one of the caller's own they share. With skills
    # left shared, also one someone else created, read-only for the caller.
    shared: bool = False
    capabilities: SkillCapabilities


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
