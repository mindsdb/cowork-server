"""The connections whose credentials a person approved for an artifact's publish.

A full-stack publish uploads the credentials of every connection the
artifact's metadata.json names, and the agent writes that file. A publish
sends credentials only of the connections recorded here
(cowork.services.datasource_approval). Only the server writes rows: after the
person approves the connections in the chat card or the publish dialog.
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.artifact_owner_choice import ARTIFACT_KEY_MAX_CHARS
from cowork.models.base import BaseSQLModel


class ArtifactDatasourceApproval(BaseSQLModel, table=True):
    """One row per artifact folder, among the rows with no org (desktop and
    local) or within one org. The folder is part of the key, so an id copied
    from another artifact's metadata.json never inherits its approval."""

    __tablename__ = "artifact_datasource_approvals"
    # A NULL org_id never collides in a plain unique key, so each side of the
    # split has its own partial index.
    __table_args__ = (
        sa.Index(
            "uq_artifact_datasource_approvals_global",
            "artifact_key",
            "artifact_dir",
            unique=True,
            sqlite_where=sa.text("org_id IS NULL"),
            postgresql_where=sa.text("org_id IS NULL"),
        ),
        sa.Index(
            "uq_artifact_datasource_approvals_org",
            "org_id",
            "artifact_key",
            "artifact_dir",
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL"),
        ),
    )

    org_id: str | None = Field(
        default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows"
    )
    artifact_key: str = Field(max_length=ARTIFACT_KEY_MAX_CHARS, description="The artifact's authorization key")
    artifact_dir: str = Field(sa_type=sa.Text, description="The artifact folder the approval holds for")
    # Sorted [{"engine", "name"}], spelled exactly as the vault keys them (DatasourceKey).
    datasources: list[dict[str, str]] = Field(
        default_factory=list, sa_column=sa.Column(sa.JSON, nullable=False)
    )
    approved_by: str | None = Field(default=None, max_length=36, description="Who approved last; NULL on desktop")
