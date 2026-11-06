from __future__ import annotations

from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ConversationFolder(BaseSQLModel, table=True):
    """A local folder the user added to a desktop chat as a working folder.

    The chat keeps its project; these are extra folders the agent is told it may
    work in. Desktop only: org deployments never write a row.
    """

    __tablename__ = "conversation_folders"

    conversation_id: UUID = Field(
        foreign_key="conversations.id",
        index=True,
        description="The chat the folder is attached to.",
    )
    # 1024 is macOS PATH_MAX, and keeps the unique index below Postgres's
    # btree row-size limit.
    path: str = Field(max_length=1024, description="Resolved absolute folder path.")
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    __table_args__ = (
        UniqueConstraint("conversation_id", "path", name="uq_conversation_folders_conversation_path"),
    )
