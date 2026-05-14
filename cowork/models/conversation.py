from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from cowork.models.base import BaseSQLModel

if TYPE_CHECKING:
    from cowork.models.message import Message
    from cowork.models.project import Project


class Conversation(BaseSQLModel, table=True):
    __tablename__ = "conversations"

    topic: str = Field(description="Topic of the conversation", max_length=255)
    project_id: UUID = Field(foreign_key="projects.id", description="Project this conversation belongs to")

    project: "Project" = Relationship()
    messages: list["Message"] = Relationship()

