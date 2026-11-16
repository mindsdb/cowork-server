from datetime import datetime
from typing import Any
from uuid import UUID

from cowork.schemas.base import CamelResponse


class NoticeResponse(CamelResponse):
    """One notice in the caller's inbox. The client words it from ``kind``
    and ``subject``."""

    id: UUID
    kind: str
    subject: str
    target: dict[str, Any]
    created_at: datetime | None = None
    read_at: datetime | None = None


class NoticeListResponse(CamelResponse):
    notices: list[NoticeResponse]
    unread_count: int


class UnreadCountResponse(CamelResponse):
    unread_count: int
