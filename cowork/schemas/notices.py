from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import ValidationError, ValidatorFunctionWrapHandler, field_validator

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

    @field_validator("fallback", mode="wrap")
    @classmethod
    def _unreadable_fallback_is_none(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> NoticeFallback | None:
        """A fallback a later build wrote, with a destination this one does
        not know, is dropped: one unreadable notice must not fail the inbox."""
        try:
            return handler(value)
        except ValidationError:
            return None


class NoticeListResponse(CamelResponse):
    notices: list[NoticeResponse]
    unread_count: int


class UnreadCountResponse(CamelResponse):
    unread_count: int
