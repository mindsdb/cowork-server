"""The time the artifacts list sorts each artifact folder by, so it opens only
the folders it shows (cowork/services/artifact_sort_times.py)."""
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ArtifactSortTime(BaseSQLModel, table=True):
    """One row per artifact folder in an organization: the newest
    modification time of its `metadata.json` the server has read. Only org tenancy
    writes rows; a person's partition and the desktop write none.

    The folder is `<project>/.anton/artifacts/<slug>` when `conversation_id`
    is NULL, and the legacy `<project>/conversations/<conversation_id>/.anton/
    artifacts/<slug>` otherwise. A turn end that created, changed or named
    the folder writes the time, so does the self-heal's full listing of a
    shared root for every folder it read, the project files PUT for the
    folder it wrote, a comparison side's carry for the folders it brought,
    and the backfill for every folder that existed before the table. A
    writer that read a time no newer than the row's leaves it as it is;
    only the self-heal's listing moves a row back, to the older time a
    folder replaced by an older copy now holds. A rewrite of
    `metadata.json` the server makes on its own, such as an id widening,
    puts the file's time back, so an old artifact does not move up the list.
    A row is a hint, not an authorization record: the list opens a row's
    folder only inside a root the caller may already read.
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
        # The list reads a project's newest rows, and an organization's
        # newest rows across its projects (newest_folders), each newest
        # first, stopping at the limit.
        sa.Index("ix_artifact_sort_times_project_time", "project_id", "sort_time"),
        sa.Index("ix_artifact_sort_times_org_time", "org_id", "sort_time"),
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
    # Compared by code point, as Python's sorted() and SQLite compare text:
    # PostgreSQL would otherwise use the database's collation, and a locale
    # such as en_US.utf8 puts "salesforce" before "sales-q1". newest_folders
    # cuts a run of tied times by slug, and the cut has to keep the folders
    # a walk of the root puts first.
    slug: str = Field(
        max_length=255,
        sa_type=sa.String(length=255).with_variant(sa.String(length=255, collation="C"), "postgresql"),
    )
    # Seconds since the epoch, as os.stat_result.st_mtime gives them.
    sort_time: float = Field(sa_column=sa.Column(sa.Float(), nullable=False))
