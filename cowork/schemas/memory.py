from enum import Enum
from uuid import UUID

from pydantic import BaseModel, model_validator


class MemoryScope(str, Enum):
    project = "project"
    global_ = "global"


def validate_project_id(values):
    scope = values.get("scope")
    project_id = values.get("project_id")
    if scope == MemoryScope.project and project_id is None:
        raise ValueError("project_id is required for project-scoped memory.")
    return values


# An endpoint for creating memory does not exist,
# because users will only update the content of the existing memory (files).
# The relevant file will be created, however, if it does not exist.
# TODO: Do we want to allow users to create these files? Or should only edits be allowed?
class MemoryUpdateRequest(BaseModel):
    scope: MemoryScope
    category: str
    content: str  # Only the content can be updated, the scope and category are used to identify what to update.
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)


class MemoryDeleteRequest(BaseModel):
    scope: MemoryScope
    category: str
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)


class MemoryResponse(BaseModel):
    scope: MemoryScope
    category: str
    content: str
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)
