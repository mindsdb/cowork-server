from datetime import datetime

from pydantic import AliasChoices, Field

from cowork.schemas.base import CamelRequest, CamelResponse


class SkillCreateRequest(CamelRequest):
    label: str
    name: str | None = None
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")
    enabled: bool | None = None
    projects: list[str] | None = None
    # Insert-or-update: overwrite an existing skill (same slug) instead of a 409.
    # Used by the in-chat draft Save so re-saving a refined skill replaces the
    # stored version (scope included). The manual "Add" form leaves this false.
    upsert: bool = False


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
    instructions: str = Field(serialization_alias="declarative")
    created_at: datetime | None
    updated_at: datetime | None
    enabled: bool
    projects: list[str]

