from __future__ import annotations

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field

from .base import BaseSQLModel


class Pin(BaseSQLModel, table=True):
    __tablename__ = "pins"

    item_type: str = Field(max_length=64)
    item_id: str = Field(max_length=64)
    title: str | None = Field(default=None, max_length=255)
    # Personal bookmarks of org-owned items: user owns the pin, org bounds it.
    user_id: str | None = Field(default=None, max_length=36, description="Owning user; NULL on local/desktop rows")
    org_id: str | None = Field(default=None, max_length=36, description="Org of the pinned item; NULL on local/desktop rows")

    __table_args__ = (
        UniqueConstraint("item_type", "item_id"),
        Index("ix_pins_user_id_org_id", "user_id", "org_id"),
    )
