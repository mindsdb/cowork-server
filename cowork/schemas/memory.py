from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class MemoryScope(str, Enum):
    project = "project"
    global_ = "global"


# An endpoint for creating memory does not exist,
# because users will only update the content of the existig memory (files).
class MemoryUpdateRequest(BaseModel):
    scope: MemoryScope
    category: str
    content: str  # Only the content can be updated, the scope and category are used to identify what to update.
    project_id: UUID | None = None


class MemoryDeleteRequest(BaseModel):
    scope: MemoryScope
    category: str
    project_id: UUID | None = None


class MemoryResponse(BaseModel):
    scope: MemoryScope
    category: str
    content: str
    project_id: UUID | None = None
