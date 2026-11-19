from enum import Enum
from uuid import UUID

from pydantic import BaseModel, model_validator

from cowork.harnesses.memory.registry import MemorySlot
from cowork.schemas.base import CamelResponse
from cowork.schemas.shared_resources import (
    MutableResourceCapabilities,
    ResourceAttribution,
)


class MemoryScope(str, Enum):
    project = "project"
    global_ = "global"
    # The operator's managed folder (COWORK_MANAGED_DIR): rules every turn
    # reads ahead of the learned tiers, which nothing in Cowork writes.
    managed = "managed"


def validate_project_id(values):
    scope = values.get("scope")
    project_id = values.get("project_id")
    if scope == MemoryScope.project and project_id is None:
        raise ValueError("project_id is required for project-scoped memory.")
    return values


class MemoryUpdateRequest(BaseModel):
    scope: MemoryScope
    category: MemorySlot
    content: str
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)


class MemoryDeleteRequest(BaseModel):
    scope: MemoryScope
    category: MemorySlot
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)


class MemoryResponse(BaseModel):
    scope: MemoryScope
    category: MemorySlot
    content: str
    project_id: UUID | None = None
    attribution: ResourceAttribution
    capabilities: MutableResourceCapabilities

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)


class MemoryCapabilities(CamelResponse):
    """What the caller may do to their memory as a whole."""

    # Whether DELETE /api/v1/memory/all removes the caller's memory
    # (cowork.services.memory.person_may_reset_memory).
    can_reset: bool


class MemoryReset(CamelResponse):
    """What DELETE /api/v1/memory/all removed."""

    # False when the caller had no memory yet, so nothing was there to remove.
    removed: bool
