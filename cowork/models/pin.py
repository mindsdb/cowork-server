from __future__ import annotations

from sqlalchemy import Index, text
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
        # One pin per (item, user, org); COALESCE keeps NULL desktop rows on
        # the old one-pin-per-item rule. Full rationale: migration d2e8f1a4c7b9.
        Index(
            "uq_pins_item_user",
            "item_type",
            "item_id",
            text("coalesce(user_id, '')"),
            text("coalesce(org_id, '')"),
            unique=True,
        ),
        Index("ix_pins_user_id_org_id", "user_id", "org_id"),
    )
