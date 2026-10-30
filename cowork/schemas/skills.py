from datetime import datetime
from typing import Literal

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
    """Who one skill is shared with.

    Hosted org mode (ENG-3787) reads the skill's access, named as auth's
    ArtifactAccess names it: ``org_allowed`` (everyone in the org),
    ``allowed_team_ids``, ``allowed_user_ids`` (Keycloak ids of people in the
    org) and ``always_on`` (put an org-wide or team skill in every turn's
    prompt instead of recalling it on demand). Each is accepted in camelCase
    too; a field left out keeps what the skill has. Elsewhere, where a person
    shares their own skills one at a time, ``shared`` shares one with
    everyone, read-only, or stops sharing it.
    """

    shared: bool | None = None
    org_allowed: bool | None = None
    allowed_team_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None
    always_on: bool | None = None


class SkillAccessView(CamelResponse):
    """Who a skill is shared with in hosted org mode (its skill_access row).
    All empty elsewhere. ``allowed_user_ids`` is listed only to whoever may
    change it (owner or admin). ``always_on``: an org-wide or team skill in
    every turn's prompt; false for one recalled on demand, a built-in, a
    managed skill, and outside hosted org mode."""

    org_allowed: bool = False
    allowed_team_ids: list[str] = Field(default_factory=list)
    allowed_user_ids: list[str] = Field(default_factory=list)
    always_on: bool = False


class SkillOwner(CamelResponse):
    """The skill's owner, by Keycloak id. ``email`` only when the owner is
    the caller; anyone else is named through GET /hub/people/?ids=."""

    id: str | None = None
    email: str | None = None


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
    # Hosted org sharing (ENG-3787). Outside it: section "mine" (managed
    # skills "org"), can_edit as capabilities.can_edit, access all empty.
    access: SkillAccessView = Field(default_factory=SkillAccessView)
    # The skills page's bucket for the caller, by audience (the owner's
    # too): "org" (org-wide), else "team" (shared with a team), else "mine"
    # (their own, shared with nobody or only people), else "shared_with_me".
    section: Literal["mine", "shared_with_me", "team", "org"] = "mine"
    can_edit: bool = False
    owner: SkillOwner = Field(default_factory=SkillOwner)


class SkillListResponse(CamelResponse):
    skills: list[SkillResponse]
    # Whether the caller may share their own skills here, one at a time:
    # what asks "who can use it" when a skill is created, before any exists.
    can_share: bool = False


class SkillSharingCapabilities(CamelResponse):
    """What the caller may do with skill sharing. ``enabled`` is hosted org
    mode; an admin holds both capabilities."""

    enabled: bool = False
    can_share_org: bool = False
    can_write_org: bool = False
    org_name: str | None = None
