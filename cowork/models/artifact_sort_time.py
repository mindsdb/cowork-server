"""The time the artifacts list sorts each artifact folder by, so it opens only
the folders it shows (cowork/services/artifact_sort_times.py)."""
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ArtifactSortTime(BaseSQLModel, table=True):
    """One row per artifact folder in an organization (or a person's
    partition): the modification time of its `metadata.json` as the server
    last read it.

    The folder is `<project>/.anton/artifacts/<slug>` when `conversation_id`
    is NULL, and the legacy `<project>/conversations/<conversation_id>/.anton/
    artifacts/<slug>` otherwise. A turn end that created or changed the
    folder writes the time, so does the self-heal's listing for a folder no
    row names, and the backfill for every folder that existed before the
    table. A rewrite of `metadata.json` the server makes on its own, such as
    an id widening, leaves the row as it was, so an old artifact does not
    move up the list. A row is a hint, not an authorization record: the list
    opens a row's folder only inside a root the caller may already read. The
    desktop writes no rows.
    """

    __tablename__ = "artifact_sort_times"
    __table_args__ = (
        # One row per folder. A NULL conversation never equals another, so a
        # shared root's folders have their own index.
        sa.Index(
            "uq_artifact_sort_times_shared",
            "org_id",
            "project_id",
            "slug",
            unique=True,
            sqlite_where=sa.text("conversation_id IS NULL"),
            postgresql_where=sa.text("conversation_id IS NULL"),
        ),
        sa.Index(
            "uq_artifact_sort_times_legacy",
            "org_id",
            "project_id",
            "conversation_id",
            "slug",
            unique=True,
            sqlite_where=sa.text("conversation_id IS NOT NULL"),
            postgresql_where=sa.text("conversation_id IS NOT NULL"),
        ),
        # The list reads a project's newest rows.
        sa.Index("ix_artifact_sort_times_project_time", "project_id", "sort_time"),
    )

    org_id: str = Field(max_length=36)
    project_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
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
    slug: str = Field(max_length=255)
    # Seconds since the epoch, as os.stat_result.st_mtime gives them.
    sort_time: float = Field(sa_column=sa.Column(sa.Float(), nullable=False))
