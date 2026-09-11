from __future__ import annotations

from datetime import datetime

from cowork.schemas.base import CamelResponse


class AttributionActor(CamelResponse):
    user_id: str
    email: str


class ResourceAttribution(CamelResponse):
    created_by: AttributionActor | None
    last_modified_by: AttributionActor | None
    last_modified_at: datetime | None


class ProjectCapabilities(CamelResponse):
    can_rename: bool
    can_delete: bool
    can_edit_instructions: bool
    # The project's directory is a folder the user chose, not one Cowork
    # allocated. Renaming it is refused and deleting the project leaves it in
    # place, so the client has to say both things differently.
    directory_is_external: bool = False


class SkillCapabilities(CamelResponse):
    can_edit: bool
    can_delete: bool
    can_disable: bool


class MutableResourceCapabilities(CamelResponse):
    can_edit: bool
    can_delete: bool
