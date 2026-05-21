from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class MemoryScope(str, Enum):
    project = "project"
    global_ = "global"


# It is important to note here that although the request and response models look similar,
# there is an important distinction.
# Creating a request is adding an item to memory, i.e., appending to the relevant memory file.
# The other requests and the response, on the other hand, relate to all of the memory (entire file) 
# for that scope and category.
class MemoryCreateRequest(BaseModel):
    scope: MemoryScope
    category: str  # The categories supported by each harness will vary.
    content: str
    project_id: UUID | None = None  # When the scope is for a project.


class MemoryUpdateRequest(BaseModel):
    scope: MemoryScope
    category: str
    content: str  # Only the content can be updated, the scope and category are used to identify the memory to be updated.
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
