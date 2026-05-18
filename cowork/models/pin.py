from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from .base import BaseSQLModel


class Pin(BaseSQLModel, table=True):
    __tablename__ = "pins"

    item_type: str = Field(max_length=64)
    item_id: str = Field(max_length=64)
    title: str | None = Field(default=None, max_length=255)

    __table_args__ = (UniqueConstraint("item_type", "item_id"),)
