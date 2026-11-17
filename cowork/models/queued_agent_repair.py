"""Which artifact folders hold an agent repair queued to a conversation."""
from uuid import UUID

import sqlalchemy as sa
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
    composer allocated before its first turn creates the row. Rows are keyed
    by org like the conversations they name, so one organization's rows are
    never read, replaced or dropped through another's.
    """

    __tablename__ = "queued_agent_repairs"
    # One row per conversation and folder in an org, or among the rows with
    # no org (desktop and local). Mirrors custom_connectors' partial-index
    # split, since a NULL org_id never collides in a plain unique key.
    __table_args__ = (
        sa.Index(
            "uq_queued_agent_repairs_conversation_slug_global",
            "conversation_id",
            "slug",
            unique=True,
            sqlite_where=sa.text("org_id IS NULL"),
            postgresql_where=sa.text("org_id IS NULL"),
        ),
        sa.Index(
            "uq_queued_agent_repairs_conversation_slug_org",
            "org_id",
            "conversation_id",
            "slug",
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL"),
        ),
    )

    org_id: str | None = Field(
        default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows"
    )
    conversation_id: UUID = Field(index=True)
    slug: str = Field(max_length=255)
