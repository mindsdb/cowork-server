from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class CustomConnector(BaseSQLModel, table=True):
    """A connector definition built in Cowork, served beside the static registry.

    Holds the connection form and display metadata only; credentials stay in
    the data vault per connection, as for built-in connectors.
    """

    __tablename__ = "custom_connectors"
    # One definition per connector id per org (or one local definition when
    # org_id is NULL). Mirrors channel_installations' partial-index split.
    __table_args__ = (
        sa.Index(
            "uq_custom_connectors_connector_id_global",
            "connector_id",
            unique=True,
            sqlite_where=sa.text("org_id IS NULL"),
            postgresql_where=sa.text("org_id IS NULL"),
        ),
        sa.Index(
            "uq_custom_connectors_connector_id_org",
            "connector_id",
            "org_id",
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL"),
        ),
    )

    connector_id: str = Field(max_length=64, description="Engine id the connections are saved under")
    label: str = Field(max_length=255, description="Display name")
    description: str = Field(default="", description="One-line description shown on the tile")
    category: str | None = Field(default=None, max_length=64, description="Picker category")
    logo_color: str | None = Field(default=None, max_length=32, description="Tile tint, a CSS color")
    spec: dict[str, Any] = Field(
        sa_column=sa.Column(sa.JSON, nullable=False),
        description="Validated connection form plus how_to and help_url",
    )
    usage_notes: str | None = Field(default=None, description="Agent-facing notes for using the connector")
    featured: bool = Field(default=True, description="Shown in the picker's Featured section")
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")
