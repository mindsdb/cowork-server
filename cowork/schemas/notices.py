from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from cowork.schemas.base import CamelResponse


class NoticeResponse(CamelResponse):
    id: UUID
    kind: str
    title: str
    data: dict[str, Any]
    created_at: datetime | None
    read_at: datetime | None


class NoticeListResponse(CamelResponse):
    notices: list[NoticeResponse]
    unread_count: int


class UnreadCountResponse(CamelResponse):
    unread_count: int
