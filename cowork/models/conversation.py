from typing import TYPE_CHECKING
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from cowork.models.base import BaseSQLModel

if TYPE_CHECKING:
    from cowork.models.message import Message
    from cowork.models.project import Project


class Conversation(BaseSQLModel, table=True):
    __tablename__ = "conversations"
    __table_args__ = (
        # A person's own conversations in an organization
        # (ConversationService.owned_ids), which every org-mode artifact
        # request reads. Declared here as well as in migration dce94d4bafc0.
        sa.Index("ix_conversations_org_id_created_by", "org_id", "created_by"),
    )

    topic: str = Field(description="Topic of the conversation", max_length=255)
    project_id: UUID = Field(foreign_key="projects.id", description="Project this conversation belongs to")
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")
    # No FK on cutoff_id — a stale/missing id should fall back to full
    # history, not block message deletion.
    history_summary: str | None = Field(default=None, description="Anton's compacted summary of earlier turns")
    history_summary_cutoff_id: UUID | None = Field(
        default=None, description="Last message id covered by history_summary"
    )
    harness: str | None = Field(
        default=None,
        description=(
            "Harness this task launched with (e.g. 'anton', 'claude-code'). "
            "Distinct from UserSettings.harness (the global default) and "
            "Message.harness (per-turn) — this is the task's own choice, "
            "recorded even when the actual work happens outside the app "
            "(e.g. an external CLI in a terminal)."
        ),
    )
    model: str | None = Field(
        default=None, description="Model alias the task's harness was launched with"
    )
    reasoning_effort: str | None = Field(
        default=None, description="Reasoning effort the task's harness was launched with"
    )
    channel_type: str | None = Field(
        default=None,
        max_length=64,
        description="Channel the runtime created this conversation for (discord, slack, ...); NULL otherwise",
    )

    project: "Project" = Relationship()
    messages: list["Message"] = Relationship()

