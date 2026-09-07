from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from cowork.coding.contracts import ApprovalDecision, PendingApproval
from cowork.coding.redaction import redact_text


@dataclass
class _Waiter:
    session_id: str
    pending: PendingApproval
    event: threading.Event
    decision: ApprovalDecision | None = None
    closed: bool = False


class ApprovalBroker:
    """Blocks the Codex reader thread until the user resolves an approval."""

    def __init__(
        self,
        on_open: Callable[[str, PendingApproval], None],
        on_close: Callable[[str, PendingApproval, ApprovalDecision], None],
    ) -> None:
        self._on_open = on_open
        self._on_close = on_close
        self._lock = threading.RLock()
        self._waiters: dict[str, _Waiter] = {}

    def request(self, session_id: str, method: str, params: dict | None) -> dict[str, str]:
        pending = self._describe(method, params or {})
        waiter = _Waiter(session_id=session_id, pending=pending, event=threading.Event())
        with self._lock:
            if any(item.session_id == session_id and not item.closed for item in self._waiters.values()):
                # CodingSession intentionally exposes one approval card. Fail a
                # concurrent request closed instead of replacing the visible
                # approval and leaving the first operation impossible to resume.
                return {"decision": "decline"}
            self._waiters[pending.id] = waiter
        try:
            self._on_open(session_id, pending)
        except Exception:
            # Do not retain an unreachable approval if persisting its open
            # state fails. The engine request fails closed with the turn.
            with self._lock:
                self._waiters.pop(pending.id, None)
            raise
        # A forgotten approval must not leave a hidden command process alive
        # indefinitely. One hour is long enough for a deliberate pause while
        # still failing closed if the UI disappears.
        waiter.event.wait(timeout=60 * 60)
        with self._lock:
            self._waiters.pop(pending.id, None)
            decision = waiter.decision or ApprovalDecision.deny
            close_now = not waiter.closed
            waiter.closed = True
        if close_now:
            self._on_close(session_id, pending, decision)
        if decision == ApprovalDecision.approve_session and pending.allow_session:
            return {"decision": "acceptForSession"}
        if decision in {ApprovalDecision.approve_once, ApprovalDecision.approve_session}:
            return {"decision": "accept"}
        return {"decision": "decline"}

    def resolve(self, session_id: str, approval_id: str, decision: ApprovalDecision) -> PendingApproval:
        with self._lock:
            waiter = self._waiters.get(approval_id)
            if waiter is None or waiter.session_id != session_id or waiter.closed:
                raise KeyError("approval is no longer pending")
            waiter.decision = decision
            waiter.closed = True
        try:
            # Persist the decision before waking Codex so the approval response
            # cannot still describe the task as awaiting the same decision.
            self._on_close(session_id, waiter.pending, decision)
        except Exception:
            # An approval that the local store could not record must never
            # authorize the underlying action. Wake Codex with a denial while
            # preserving the persistence error for the HTTP caller.
            with self._lock:
                waiter.decision = ApprovalDecision.deny
            waiter.event.set()
            raise
        waiter.event.set()
        return waiter.pending

    def cancel_session(self, session_id: str) -> None:
        cancelled: list[_Waiter] = []
        with self._lock:
            for waiter in self._waiters.values():
                if waiter.session_id == session_id and not waiter.closed:
                    waiter.decision = ApprovalDecision.deny
                    waiter.closed = True
                    cancelled.append(waiter)
        first_error: Exception | None = None
        for waiter in cancelled:
            try:
                self._on_close(session_id, waiter.pending, ApprovalDecision.deny)
            except Exception as exc:  # noqa: BLE001 - every waiter still needs to be released.
                first_error = first_error or exc
            finally:
                waiter.event.set()
        if first_error is not None:
            raise first_error

    @staticmethod
    def _describe(method: str, params: dict) -> PendingApproval:
        kind = "elevated_action"
        title = "Approve elevated action"
        risk = "This action requires access beyond the task's normal workspace policy."
        detail = "Codex requested elevated access."
        cwd = ApprovalBroker._string(params.get("cwd") or params.get("workingDirectory"))
        allow_session = False

        if "commandExecution" in method:
            kind = "command"
            title = "Run command"
            command = params.get("command") or params.get("commands") or params.get("reason")
            detail = ApprovalBroker._string(command) or "Codex requested permission to run a command."
            risk = "The command may modify files, start processes, or access resources outside the task sandbox."
            allow_session = bool(params.get("proposedExecpolicyAmendment") or params.get("proposedExecPolicyAmendment"))
        elif "fileChange" in method:
            kind = "file_change"
            title = "Modify protected files"
            detail = ApprovalBroker._string(params.get("reason")) or "Codex requested a file change outside its normal write policy."
            risk = "This change may write outside the managed task worktree or alter protected files."
        elif "network" in method.lower() or "permission" in method.lower():
            kind = "network_or_permission"
            title = "Grant additional access"
            detail = ApprovalBroker._string(params.get("reason") or params.get("permissions")) or detail
            risk = "This may access the network or expand the agent's permissions."

        return PendingApproval(
            id=str(uuid.uuid4()),
            method=method[:256],
            kind=kind,
            title=title,
            detail=redact_text(detail)[:8_192],
            cwd=redact_text(cwd)[:32_768] if cwd else None,
            risk=risk,
            scope="This task only",
            allow_session=allow_session,
        )

    @staticmethod
    def _string(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value[:20])
        if isinstance(value, dict):
            return ", ".join(f"{key}: {val}" for key, val in list(value.items())[:20])
        return ""
