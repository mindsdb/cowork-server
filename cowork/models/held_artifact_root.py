"""The artifact roots the server has seen hold an artifact folder, so the
artifacts list opens only those (cowork/services/held_artifact_roots.py)."""
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class HeldArtifactRoot(BaseSQLModel, table=True):
    """One row per artifact root that has held an artifact folder, in an
    organization (or a person's partition).

    The root is `<project>/.anton/artifacts` when `conversation_id` is NULL,
    and the legacy `<project>/conversations/<conversation_id>/.anton/
    artifacts` otherwise. The server writes every row from folders it found
    itself, and never removes one: a root emptied later only costs the list
    one open. A row is a hint, not an authorization record. The list opens a
    row's root only for a project the caller's scope returns, and a legacy
    root only for a conversation the caller owns. The desktop writes no rows.
    """

    __tablename__ = "held_artifact_roots"
    __table_args__ = (
        # One row per root. A NULL conversation never equals another, so the
        # shared root has its own index.
        sa.Index(
            "uq_held_artifact_roots_shared",
            "org_id",
            "project_id",
            unique=True,
            sqlite_where=sa.text("conversation_id IS NULL"),
            postgresql_where=sa.text("conversation_id IS NULL"),
        ),
        sa.Index(
            "uq_held_artifact_roots_legacy",
            "org_id",
            "project_id",
            "conversation_id",
            unique=True,
            sqlite_where=sa.text("conversation_id IS NOT NULL"),
            postgresql_where=sa.text("conversation_id IS NOT NULL"),
        ),
    )

    org_id: str = Field(max_length=36)
    project_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    conversation_id: UUID | None = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
