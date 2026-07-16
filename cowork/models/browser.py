"""Browser Control (Milestone 1, read-only) — content-free data model.

Three tables back the read-only Browser Control feature:

- ``BrowserSession`` — one per (conversation, project) browser attachment.
  Holds the live control/bridge state and the approved active domain.
- ``BrowserTabGrant`` — the per-domain, per-action-class permission grant
  for a session (one approved tab / one active domain in M1).
- ``BrowserAction`` — the ordered history of brokered read-only actions.
  Its ``observed_result`` is a content-free DIGEST ONLY (allowlisted keys:
  ``http_status``, ``final_domain``, ``link_count``, ``settled``) — never
  page text, full URLs, paths/queries, titles, hrefs, cookies, or form
  values. The store rejects any disallowed key (AC8).

Everything here is content-free by construction: host-only ``domain``,
action type/class, timing, and typed codes only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import JSON
from sqlmodel import Column, Field

from cowork.models.base import BaseSQLModel


class BrowserSession(BaseSQLModel, table=True):
    """A browser attachment scoped to one conversation/project.

    The control gate (``control_state``) and the bridge lifecycle
    (``bridge_state``) live here so a Stop survives reconnect and a
    stopped session stays stopped.
    """

    __tablename__ = "browser_sessions"
    __table_args__ = (
        sa.UniqueConstraint("conversation_id", name="uq_browser_sessions_conversation"),
    )

    conversation_id: UUID = Field(
        foreign_key="conversations.id",
        ondelete="CASCADE",
        index=True,
        description="Conversation (task) this browser session belongs to.",
    )
    project_id: UUID = Field(
        foreign_key="projects.id",
        index=True,
        description="Project the session lives in.",
    )
    control_state: str = Field(
        default="active",
        max_length=16,
        description="Control gate: active | stopped | taken_over. Never auto-cleared from stopped.",
    )
    bridge_state: str = Field(
        default="disconnected",
        max_length=24,
        description="Mirrored Electron-main bridge state: disconnected | awaiting_approval | connected | lost.",
    )
    active_domain: str | None = Field(
        default=None,
        max_length=255,
        description="Host-only registrable domain of the single approved tab (no path/query).",
    )
    available: bool = Field(
        default=False,
        description="Whether a bridge is currently connected and can execute commands.",
    )
    requires_reapproval: bool = Field(
        default=False,
        description="Set when Chrome restarted / target ids changed — a fresh approval is required.",
    )


class BrowserTabGrant(BaseSQLModel, table=True):
    """A per-domain, per-action-class permission grant for a session.

    M1: one approved tab / one active domain grant per session. The
    ``decision`` follows the M1 vocabulary (granted | denied | expired |
    revoked). Uniqueness on (session_id, domain, action_class) mirrors the
    channel models' dedupe constraints.
    """

    __tablename__ = "browser_tab_grants"
    __table_args__ = (
        sa.UniqueConstraint(
            "session_id",
            "domain",
            "action_class",
            name="uq_browser_tab_grants_scope",
        ),
    )

    session_id: UUID = Field(
        foreign_key="browser_sessions.id",
        ondelete="CASCADE",
        index=True,
        description="Session this grant belongs to.",
    )
    domain: str = Field(
        max_length=255,
        index=True,
        description="Host-only registrable domain the grant covers (no path/query).",
    )
    action_class: str = Field(
        max_length=16,
        description="Capability class: read | navigate.",
    )
    decision: str = Field(
        default="granted",
        max_length=16,
        description="granted | denied | expired | revoked.",
    )
    granted_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the grant was approved.",
    )
    expires_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="Optional expiry; None = no expiry in M1.",
    )


class BrowserAction(BaseSQLModel, table=True):
    """One brokered read-only browser action, in order.

    ``observed_result`` is a content-free DIGEST ONLY — the store rejects
    any key outside the allowlist. It is NEVER the transient visible
    extraction returned to the model (that is never persisted).
    """

    __tablename__ = "browser_actions"
    __table_args__ = (
        sa.UniqueConstraint(
            "session_id",
            "sequence",
            name="uq_browser_actions_sequence",
        ),
        sa.UniqueConstraint(
            "command_id",
            name="uq_browser_actions_command_id",
        ),
    )

    session_id: UUID = Field(
        foreign_key="browser_sessions.id",
        ondelete="CASCADE",
        index=True,
        description="Session this action belongs to.",
    )
    sequence: int = Field(
        description="Monotonic per-session action ordinal (1-based).",
    )
    command_id: str = Field(
        max_length=64,
        index=True,
        description="Broker command id correlating enqueue → poller result.",
    )
    idempotency_key: str = Field(
        max_length=128,
        description="Stable key so a retried enqueue reuses the pending row.",
    )
    action_type: str = Field(
        max_length=16,
        description="Stored action: inspect | navigate | scroll | wait.",
    )
    action_class: str = Field(
        max_length=16,
        description="Capability class checked against the grant: read | navigate.",
    )
    domain: str | None = Field(
        default=None,
        max_length=255,
        description="Host-only domain the action targeted (no path/query).",
    )
    status: str = Field(
        default="pending",
        max_length=16,
        description="Lifecycle: pending | in_flight | observed | failed.",
    )
    result_code: str | None = Field(
        default=None,
        max_length=24,
        description="WS4-internal result code once resolved: ok | timeout | target_lost | unapproved_tab | permission_denied | error.",
    )
    observed_result: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSON),
        description=(
            "Content-free digest ONLY (allowlisted keys: http_status, "
            "final_domain, link_count, settled). Never page text, full URL, "
            "path/query, title, href, cookies, or form values."
        ),
    )
    duration_ms: int | None = Field(
        default=None,
        description="Wall-clock duration of the brokered command, if resolved.",
    )
