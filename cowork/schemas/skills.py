from datetime import datetime

from pydantic import AliasChoices, Field

from cowork.schemas.base import CamelRequest, CamelResponse
from cowork.schemas.shared_resources import ResourceAttribution, SkillCapabilities


class SkillCreateRequest(CamelRequest):
    label: str
    name: str | None = None
    description: str | None = None
    instructions: str | None = Field(default=None, alias="declarative")
    enabled: bool | None = None
    projects: list[str] | None = None
    # Share the new skill with everyone, read-only. Only where the skills
    # list's ``canShare`` is true; false keeps it the creator's alone.
    shared: bool = False


class SkillSharingRequest(CamelRequest):
    """Share one of the caller's own skills with everyone, read-only, or
    stop sharing it."""

    shared: bool


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
    # In the shared-skill list: one someone else shares with the caller,
    # read-only for them, or one of the caller's own they share. With skills
    # left shared, also one someone else created, read-only for the caller.
    shared: bool = False
    capabilities: SkillCapabilities


class SkillListResponse(CamelResponse):
    skills: list[SkillResponse]
    # Whether the caller may share their own skills here, one at a time:
    # what asks "who can use it" when a skill is created, before any exists.
    can_share: bool = False
