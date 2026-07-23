from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class StandingRule(BaseSQLModel, table=True):
    """A standing permission: this agent action on this origin needs no
    further proposals ("Always").

    Enforcement is a deterministic exact-match lookup in the gate — NEVER a
    memory retrieval. scope = origin + action_kind (tool:normalized-label,
    e.g. 'browser_click:send' on mail.google.com). Granted from real
    resolutions (evidence-gated: 3+ identical unmodified approvals), revoked
    here with one click; revocation is checked at act time."""

    __tablename__ = "standing_rules"

    origin: str = Field(
        index=True,
        description="Host scope, e.g. mail.google.com (scheme-less, lowercase)",
    )
    action_kind: str = Field(
        index=True,
        description="Action scope: tool:normalized-label, e.g. browser_click:send",
    )
    source_approval_id: UUID = Field(
        foreign_key="approvals.id",
        description="The resolution this rule was granted from (audit trail)",
    )
    hit_count: int = Field(
        default=0,
        description="Times the rule bypassed a proposal — shown in the Memories shelf",
    )
    last_fired_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant of the most recent bypass",
    )
    revoked_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant revoked (null = active). Checked at act time.",
    )
