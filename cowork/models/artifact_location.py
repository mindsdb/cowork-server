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
    artifacts/<slug>` otherwise. The server writes every row from a folder it
    found or made itself; a row is a hint, not an authorization record. A
    lookup opens the folder only inside a root the caller may already read,
    and serves it only when the folder's `metadata.json` names the requested
    id. The desktop writes no rows: its folders are on a local disk, where the
    scan is cheap.
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
        ),
    )
    slug: str = Field(max_length=255)
