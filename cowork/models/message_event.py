from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import JSON
from sqlmodel import Column, Field

from cowork.models.base import BaseSQLModel


class MessageEvent(BaseSQLModel, table=True):
    __tablename__ = "message_events"

    message_id: UUID = Field(..., foreign_key="messages.id")
    sequence_number: int = Field(..., description="Sequence number of the event")
    event_data: dict[str, Any] | BaseModel | str | list[Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Data of the event",
    )
