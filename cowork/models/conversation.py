from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship

from cowork.models.base import BaseSQLModel

if TYPE_CHECKING:
    from cowork.models.message import Message
    from cowork.models.project import Project


class Conversation(BaseSQLModel, table=True):
    __tablename__ = "conversations"

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
    # Anton's completion-verifier latch, carried across the per-message session
    # rebuild. Opaque here: the shape is anton's contract and this row only
    # relays it. Model alias plus counters, so nothing secret.
    verifier_latch: dict[str, object] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Anton's verifier latch state for this conversation",
    )

    project: "Project" = Relationship()
    messages: list["Message"] = Relationship()

