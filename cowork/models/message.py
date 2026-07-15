from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship

from cowork.models.base import BaseSQLModel
from cowork.schemas.responses import Role, Message as OpenAIMessage


if TYPE_CHECKING:
    from cowork.models.message_event import MessageEvent


class Message(BaseSQLModel, table=True):
    __tablename__ = "messages"

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
    seq: int = Field(
        default=0,
        sa_column_kwargs={"server_default": "0"},
        description=(
            "Intra-turn ordinal. A turn persists several block-messages "
            "(assistant tool_use, user tool_result, ...) that share one "
            "created_at; seq keeps them in emission order, since the role "
            "tiebreak in message ordering would otherwise sort tool_result "
            "(user) ahead of tool_use (assistant). 0 for single-row turns."
        ),
    )

    message_events: list["MessageEvent"] = Relationship(
        sa_relationship_kwargs={"order_by": "MessageEvent.sequence_number"}
    )

    def to_openai_message(self) -> OpenAIMessage:
        """Convert to the OpenAI-compatible message format used in the API."""
        content = self.content
        if isinstance(content, list):
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