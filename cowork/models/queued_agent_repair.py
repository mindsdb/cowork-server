"""Which artifact folders hold an agent repair queued to a conversation."""
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class QueuedAgentRepair(BaseSQLModel, table=True):
    """One row per conversation and artifact folder an agent repair was queued
    to. The repair itself lives in the folder (cowork/services/
    artifact_revisions.py); this row tells the end of that conversation's turn
    which folder to check, so it stats that folder instead of every artifact
    folder in the project. A row is a hint: the turn end reads the folder's
    own repair records, then drops the row.

    No foreign key on the conversation: a repair can name a conversation the
    composer allocated before its first turn creates the row.
    """

    __tablename__ = "queued_agent_repairs"
    __table_args__ = (
        UniqueConstraint("conversation_id", "slug", name="uq_queued_agent_repairs_conversation_slug"),
    )

    conversation_id: UUID = Field(index=True)
    slug: str = Field(max_length=255)
