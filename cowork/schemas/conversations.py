from datetime import datetime
from uuid import UUID

from cowork.schemas.base import CamelRequest, CamelResponse


class ConversationCreateRequest(CamelRequest):
    topic: str | None = None
    title: str | None = None
    project: str | None = None
    project_id: UUID | None = None


class ConversationUpdateRequest(CamelRequest):
    topic: str | None = None
    title: str | None = None
    project: str | None = None
    project_id: UUID | None = None
    disabled_connections: list[dict] | None = None


class ConversationMoveRequest(CamelRequest):
    """Move a task to another project. `move_objects` (default true) also
    relocates the artifacts the task created and re-tags its files."""
    project: str | None = None
    project_id: UUID | None = None
    move_objects: bool = True


class ConversationListItem(CamelResponse):
    id: UUID
    title: str
    preview: str
    updated_at: datetime | None
    created_at: datetime | None
    project: str | None = None
    project_path: str | None = None
    project_id: UUID | None
