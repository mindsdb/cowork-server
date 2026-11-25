from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from cowork.schemas.base import CamelResponse


class NoticeDestination(CamelResponse):
    kind: Literal["schedule", "conversation"]
    id: UUID


class NoticeFallback(CamelResponse):
    """A summary and an in-app destination understood independently of the notice kind."""

    summary: str
    destination: NoticeDestination


class NoticeResponse(CamelResponse):
    """One notice in the caller's inbox, with a fallback for unfamiliar kinds."""

    id: UUID
    kind: str
    subject: str
    target: dict[str, Any]
    fallback: NoticeFallback | None = None
    created_at: datetime | None = None
    read_at: datetime | None = None


class NoticeListResponse(CamelResponse):
    notices: list[NoticeResponse]
    unread_count: int


class UnreadCountResponse(CamelResponse):
    unread_count: int
