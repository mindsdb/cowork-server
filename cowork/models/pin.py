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
        # One pin per (item, user, org) so two org members can pin the same
        # item. org_id belongs in the boundary because neither identifier is
        # globally unique: a user belongs to many orgs (Keycloak user id +
        # active-org header), and item_id is client-supplied (project pins
        # use the name, unique only per org). COALESCE turns the NULLs of
        # local/desktop rows into real values, keeping the pre-tenancy
        # "one pin per item" dedupe there.
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
