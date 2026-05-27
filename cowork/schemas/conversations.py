from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from cowork.schemas.base import CamelRequest, CamelResponse


class ConversationCreateRequest(CamelRequest):
    topic: str
    project_id: UUID | None = None


class ConversationUpdateRequest(CamelRequest):
    topic: str | None = None
    project_id: UUID | None = None


class ConversationListItem(CamelResponse):
    id: UUID
    title: str
    preview: str
    updated_at: datetime | None
    created_at: datetime | None
    project: str | None = None
    project_id: UUID | None
