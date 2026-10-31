"""Where each artifact's folder lives, so a lookup by id opens one folder."""
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ArtifactLocation(BaseSQLModel, table=True):
    """One row per artifact id in an organization (or a person's partition).

    The folder is `<project>/.anton/artifacts/<slug>` when `conversation_id`
    is NULL, and the legacy `<project>/conversations/<conversation_id>/.anton/
    artifacts/<slug>` otherwise. The server writes every row from folders it
    found itself after reading every root of the project, and moves it with
    a task's folders; a row is a hint, not an authorization record. A lookup
    opens the folder only inside a root the caller may already read, and
    serves it only when the folder's `metadata.json` names the requested id.
    The desktop writes no rows: its folders are on a local disk, where the
    scan is cheap.

    `conflicted` marks an id the server has seen held by two folders of the
    row's project. A lookup never serves such a row: it lists the project's
    roots and answers 409 when two of the id's folders are in the roots that
    decide the lookup (the caller's own, or every member's when the caller's
    own hold none), as before rows existed.
    """

    __tablename__ = "artifact_locations"
    __table_args__ = (
        UniqueConstraint("org_id", "artifact_id", name="uq_artifact_locations_org_artifact"),
    )

    org_id: str = Field(max_length=36)
    artifact_id: str = Field(max_length=32)
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
    slug: str = Field(max_length=255)
    conflicted: bool = Field(default=False, sa_column_kwargs={"server_default": sa.false()})
