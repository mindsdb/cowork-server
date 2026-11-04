"""A conversation's Chimera sidekick, kept between turns.

The harness builds a new anton session every message, so the sidekick's own
message list, summary and status would be lost at the end of each turn. One
row per conversation holds anton's ``SidekickState`` (``ChatSession.
sidekick_state``) as the turn left it, and the next turn hands it back
(cowork.services.sidekick_contexts).

No org column: the row is reached only through its conversation, so the
conversation's visibility is the tenancy check, as for task_objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field, SQLModel

#: The status a sidekick is left in when its last report handed the work back
#: to the main agent (``needs_main`` or ``budget_exhausted``). The route gate
#: sends every turn of such a conversation to the agent.
SIDEKICK_OPEN = "open"
SIDEKICK_IDLE = "idle"


class SidekickContext(SQLModel, table=True):
    __tablename__ = "sidekick_contexts"

    conversation_id: UUID = Field(
        primary_key=True,
        foreign_key="conversations.id",
        description="The conversation this sidekick works for",
    )
    sidekick_id: str | None = Field(default=None, max_length=128, description="anton's id for the sidekick")
    model: str | None = Field(default=None, max_length=256, description="The model the sidekick last ran on")
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=sa.Column(sa.JSON, nullable=False),
        description="The sidekick's own message list",
    )
    summary: str | None = Field(
        default=None,
        sa_column=sa.Column(sa.Text, nullable=True),
        description="What the sidekick restarts from once its messages are too old to reuse",
    )
    status: str = Field(default=SIDEKICK_IDLE, max_length=16, description="'idle' or 'open'")
    last_call_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore[arg-type]
        description="When the main agent last delegated to the sidekick",
    )
    # Not in the contract's column list, but part of anton's SidekickState:
    # dropping it would reset the model-upgrade evidence every turn.
    signals: Any = Field(
        default=None,
        sa_column=sa.Column(sa.JSON, nullable=True),
        description="anton's stuck/take-back signal count",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=sa.DateTime(timezone=True),  # type: ignore[arg-type]
        sa_column_kwargs={"server_default": sa.func.now(), "onupdate": sa.func.now()},
        description="When this row was last written",
    )
