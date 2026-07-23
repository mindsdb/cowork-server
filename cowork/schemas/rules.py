from __future__ import annotations

from datetime import datetime
from uuid import UUID

from cowork.schemas.base import CamelResponse


class StandingRuleResponse(CamelResponse):
    id: UUID
    origin: str
    action_kind: str
    source_approval_id: UUID
    hit_count: int
    last_fired_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime

    @classmethod
    def serialize(cls, obj):
        return cls.model_validate(obj, from_attributes=True).model_dump(by_alias=True, mode="json")
