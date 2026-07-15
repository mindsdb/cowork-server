"""`BrowserActionStore` — DAO for the ordered browser-action history.

Every persisted `observed_result` is a content-free DIGEST built via
`build_observed_digest` and validated with `assert_content_free_digest`, so
a disallowed key (text, full url, path/query, title, href, cookie, value,
selector, …) is REJECTED before it can touch the database (AC8).
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.browser import BrowserAction
from cowork.schemas.browser import (
    ACTION_TYPE_TO_CLASS,
    ActionStatus,
    BrowserActionType,
    ResultCode,
    assert_content_free_digest,
    build_observed_digest,
    coerce_enum,
)


class BrowserActionStore:
    """DAO over `browser_actions` for one server session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _next_sequence(self, session_id: UUID) -> int:
        rows = self._session.exec(
            select(BrowserAction.sequence).where(
                BrowserAction.session_id == session_id
            )
        ).all()
        return (max(rows) + 1) if rows else 1

    def get_or_reuse_pending(
        self, session_id: UUID, idempotency_key: str
    ) -> BrowserAction | None:
        """Return an existing pending row for this idempotency key, if any.

        Lets a retried enqueue reuse the pending row instead of appending a
        duplicate.
        """
        return self._session.exec(
            select(BrowserAction).where(
                BrowserAction.session_id == session_id,
                BrowserAction.idempotency_key == idempotency_key,
                BrowserAction.status == ActionStatus.pending.value,
            )
        ).first()

    def append_pending(
        self,
        *,
        session_id: UUID,
        command_id: str,
        idempotency_key: str,
        action_type: BrowserActionType | str,
        domain: str | None = None,
    ) -> BrowserAction:
        """Append a new pending action, assigning sequence + idempotency key.

        Reuses an existing pending row for the same idempotency key.
        """
        existing = self.get_or_reuse_pending(session_id, idempotency_key)
        if existing is not None:
            return existing

        at = coerce_enum(BrowserActionType, action_type)
        action = BrowserAction(
            session_id=session_id,
            sequence=self._next_sequence(session_id),
            command_id=command_id,
            idempotency_key=idempotency_key,
            action_type=at.value,
            action_class=ACTION_TYPE_TO_CLASS[at].value,
            domain=domain,
            status=ActionStatus.pending.value,
        )
        return self._save(action)

    def mark_in_flight(self, command_id: str) -> BrowserAction | None:
        action = self._by_command(command_id)
        if action is None:
            return None
        action.status = ActionStatus.in_flight.value
        return self._save(action)

    def record_observed(
        self,
        command_id: str,
        *,
        result_code: ResultCode | str,
        transient: dict[str, Any] | None = None,
        digest: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> BrowserAction | None:
        """Record a terminal result for a command.

        `transient` is the raw (possibly content-bearing) observed blob from
        the bridge; only its allowlisted digest is persisted. Passing
        `digest` directly is also validated. A non-`ok` result never persists
        an observed digest and marks the row `failed`; an `ok` result marks
        it `observed`.
        """
        action = self._by_command(command_id)
        if action is None:
            return None

        rc = coerce_enum(ResultCode, result_code)

        stored_digest: dict[str, Any] | None = None
        if rc == ResultCode.ok:
            if digest is not None:
                assert_content_free_digest(digest)
                stored_digest = digest
            else:
                stored_digest = build_observed_digest(transient)
            action.status = ActionStatus.observed.value
        else:
            # A failed/timed-out/lost action NEVER records observed=ok.
            action.status = ActionStatus.failed.value
            stored_digest = None

        action.result_code = rc.value
        action.observed_result = stored_digest
        if duration_ms is not None:
            action.duration_ms = int(duration_ms)
        # Keep the digest's final_domain in sync with the row's domain when
        # present (still host-only).
        if stored_digest and stored_digest.get("final_domain"):
            action.domain = stored_digest["final_domain"]
        return self._save(action)

    def mark_failed(
        self,
        command_id: str,
        *,
        result_code: ResultCode | str = ResultCode.error,
        duration_ms: int | None = None,
    ) -> BrowserAction | None:
        """Force a row terminal-failed without any observed digest.

        Used when a tab/Chrome death kills an in-flight command
        (`target_lost`) — the row is `failed`, never `observed=ok`.
        """
        return self.record_observed(
            command_id,
            result_code=result_code,
            transient=None,
            digest=None,
            duration_ms=duration_ms,
        )

    def last_observed(self, session_id: UUID) -> BrowserAction | None:
        """The most recent successfully-observed action for a session."""
        return self._session.exec(
            select(BrowserAction)
            .where(
                BrowserAction.session_id == session_id,
                BrowserAction.status == ActionStatus.observed.value,
            )
            .order_by(BrowserAction.sequence.desc())
        ).first()

    def action_count(self, session_id: UUID) -> int:
        rows = self._session.exec(
            select(BrowserAction.id).where(BrowserAction.session_id == session_id)
        ).all()
        return len(rows)

    def _by_command(self, command_id: str) -> BrowserAction | None:
        return self._session.exec(
            select(BrowserAction).where(BrowserAction.command_id == command_id)
        ).first()

    def _save(self, action: BrowserAction) -> BrowserAction:
        self._session.add(action)
        self._session.commit()
        self._session.refresh(action)
        return action
