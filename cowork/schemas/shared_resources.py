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


class SkillCapabilities(CamelResponse):
    can_edit: bool
    can_delete: bool
    can_disable: bool


class MutableResourceCapabilities(CamelResponse):
    can_edit: bool
    can_delete: bool
