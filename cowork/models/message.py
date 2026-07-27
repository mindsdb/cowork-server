from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import JSON, Index
from sqlmodel import Column, Field, Relationship

from cowork.models.base import BaseSQLModel
from cowork.schemas.responses import Role, Message as OpenAIMessage


if TYPE_CHECKING:
    from cowork.models.message_event import MessageEvent


class Message(BaseSQLModel, table=True):
    __tablename__ = "messages"
    # Composite index for the per-conversation MAX(created_at) that derives a
    # conversation's last-activity time (ENG-961): with conversation_id as the
    # leading column the DB can seek to the max instead of scanning the group.
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    conversation_id: UUID = Field(
        ...,
        foreign_key="conversations.id",
        description="ID of the conversation that this message belongs to",
        index=True,
    )
    role: Role = Field(description="Role of the message")
    content: dict[str, Any] | BaseModel | str | list[Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Content of the message as JSON",
    )
    harness: str | None = Field(
        default=None,
        description="Harness/agent that generated this message (e.g. 'anton', 'hermes')",
    )
    # Authorship only — tenancy is scoped via the conversation (roots-only rule).
    created_by: str | None = Field(default=None, max_length=36, description="User who authored the message; NULL on local/desktop rows")
    seq: int = Field(
        default=0,
        sa_column_kwargs={"server_default": "0"},
        description=(
            "Per-conversation monotonic ordinal. Several block-messages "
            "(assistant tool_use, user tool_result, ...) can share one "
            "created_at (second precision); seq orders every message in the "
            "conversation deterministically, without depending on that "
            "resolution. Assigned as max(seq)+1 on insert; 0 for legacy rows."
        ),
    )

    message_events: list["MessageEvent"] = Relationship(
        sa_relationship_kwargs={"order_by": "MessageEvent.sequence_number"}
    )

    def to_openai_message(self) -> OpenAIMessage:
        """Convert to the OpenAI-compatible message format used in the API."""
        content = self.content
        if isinstance(content, list):
            # Tool block-rows (tool_use / tool_result) must reach the model
            # verbatim so the next turn sees prior tool calls and results.
            if any(
                isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result")
                for block in content
            ):
                return OpenAIMessage(role=self.role.value, content=content)
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "input_text"
            ]
            content = "\n\n".join(text_parts) if text_parts else ""
        return OpenAIMessage(
            role=self.role.value,
            content=content,
        )