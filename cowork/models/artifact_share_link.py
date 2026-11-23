"""Lookup hints for mobile notifications that carry only a published URL."""
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ArtifactShareLink(BaseSQLModel, table=True):
    __tablename__ = "artifact_share_links"
    __table_args__ = (
        UniqueConstraint("org_id", "url_digest", "project_id", "artifact_id", name="uq_artifact_share_target"),
    )

    org_id: str = Field(max_length=36)
    url_digest: str = Field(max_length=64)
    project_id: UUID = Field(foreign_key="projects.id", ondelete="CASCADE", index=True)
    artifact_id: str = Field(max_length=32)
