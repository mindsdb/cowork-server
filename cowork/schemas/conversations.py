from uuid import UUID

from pydantic import BaseModel


class ConversationCreateRequest(BaseModel):
    topic: str
    project_id: UUID | None = None


class ConversationUpdateRequest(BaseModel):
    topic: str | None = None
    project_id: UUID | None = None
