from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from cowork.schemas.base import CamelRequest, CamelResponse


class ConversationCreateRequest(CamelRequest):
    topic: str | None = None
    title: str | None = None
    project: str | None = None
    project_id: UUID | None = None
    harness: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


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
    harness: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


class ConversationItemsPage(CamelResponse):
    """Cursor-paginated envelope for GET /conversations/{id}/items. `items`
    is a plain list[dict] passthrough (NOT nested CamelResponse models) —
    only the envelope's own keys (`items`/`hasMore`/`nextBefore`) go through
    the camelCase alias; per-item fields keep their existing names exactly
    (e.g. `created_at`, not `createdAt`) so this doesn't silently change
    what an item dict looks like."""

    items: list[dict]
    has_more: bool
    next_before: str | None = None
