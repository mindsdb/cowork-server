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
    # No FK on cutoff_id — a stale/missing id should fall back to full
    # history, not block message deletion.
    history_summary: str | None = Field(default=None, description="Anton's compacted summary of earlier turns")
    history_summary_cutoff_id: UUID | None = Field(
        default=None, description="Last message id covered by history_summary"
    )

    project: "Project" = Relationship()
    messages: list["Message"] = Relationship()

