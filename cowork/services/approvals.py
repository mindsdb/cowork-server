from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.approval import Approval
from cowork.models.approval_token import ApprovalToken
from cowork.schemas.approvals import (
    ActionDescriptorV1,
    ApprovalResponse,
    AuthDescriptorV1,
    parse_descriptor,
)
from cowork.services.conversations import ConversationService

DEFAULT_TTL_SECONDS = 72 * 3600
# Execution tokens outlive the resolve click by minutes, not days — a token
# nobody spends was almost certainly interrupted.
TOKEN_TTL_SECONDS = 10 * 60

EVENT_REQUESTED = "response.approval_requested"
EVENT_RESOLVED = "response.approval_resolved"

# Executors run a resolved descriptor verbatim and return the receipt payload.
# Signature: (session, args, raw_token) -> dict. The browser gate (P5)
# registers the browser_* set; tests register stubs.
ExecutorFn = Callable[[Session, dict[str, Any], str], dict[str, Any]]
_EXECUTORS: dict[str, ExecutorFn] = {}


def register_executor(tool: str, fn: ExecutorFn) -> None:
    _EXECUTORS[tool] = fn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone=True columns — treat
    # naive as UTC rather than crashing the comparison.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ApprovalService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Create / list / get
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        conversation_id: UUID,
        descriptor: ActionDescriptorV1 | AuthDescriptorV1,
        draft: str = "",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> Approval:
        approval = Approval(
            conversation_id=conversation_id,
            kind=descriptor.kind,
            action_descriptor=descriptor.model_dump(),
            draft=draft,
            ttl_seconds=ttl_seconds,
            expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
        )
        self.session.add(approval)
        self.session.commit()
        self.session.refresh(approval)
        summary = _summary_of(approval)
        ConversationService(self.session).save_assistant_turn(
            conversation_id,
            f"Proposal parked: {summary}",
            [{"type": EVENT_REQUESTED, "approval": ApprovalResponse.serialize(approval)}],
        )
        return approval

    def list(
        self,
        *,
        status: str | None = None,
        conversation_id: UUID | None = None,
    ) -> list[Approval]:
        stmt = select(Approval).order_by(Approval.created_at.desc())  # type: ignore[attr-defined]
        if status is not None:
            stmt = stmt.where(Approval.status == status)
        if conversation_id is not None:
            stmt = stmt.where(Approval.conversation_id == conversation_id)
        return list(self.session.exec(stmt).all())

    def get(self, approval_id: UUID) -> Approval:
        approval = self.session.get(Approval, approval_id)
        if approval is None:
            raise ValueError("Approval not found")
        return approval

    # ------------------------------------------------------------------
    # Resolve (+ deterministic execution)
    # ------------------------------------------------------------------

    def resolve(
        self,
        approval_id: UUID,
        *,
        resolution: str,
        edited_draft: str | None = None,
    ) -> tuple[Approval, bool]:
        """Resolve a pending approval. Returns (approval, executed_now).

        Idempotent: anything already resolved comes back as-is with
        executed_now=False and NO re-execution — a double-click never
        double-sends.
        """
        approval = self.get(approval_id)
        if approval.status != "pending":
            return approval, False
        if _as_utc(approval.expires_at) <= _utcnow():
            self._settle(approval, "expired", receipt=None)
            return approval, False
        if resolution == "edited" and not edited_draft:
            raise ValueError("edited resolution requires edited_draft")

        descriptor = parse_descriptor(approval.action_descriptor)
        receipt: dict[str, Any] | None = None

        if resolution == "skipped":
            self._settle(approval, "skipped", receipt=None)
            return approval, True

        # approved / edited — execute deterministically, exactly as approved.
        if isinstance(descriptor, AuthDescriptorV1):
            # Auth cards park no execution: approving hands the tab to the
            # human; the agent re-attempts on its next turn.
            receipt = {"executed": False, "handed_to_user": descriptor.app_name}
        else:
            args = dict(descriptor.args)
            if resolution == "edited":
                # v1 convention: the descriptor's draft-bearing arg is `text`.
                args["text"] = edited_draft
            raw_token = self._issue_token(approval, tool=descriptor.tool, args=args)
            executor = _EXECUTORS.get(descriptor.tool)
            if executor is None:
                receipt = {
                    "executed": False,
                    "error": f"no executor registered for tool '{descriptor.tool}'",
                }
            else:
                result = executor(self.session, args, raw_token)
                result.setdefault("executed", True)
                receipt = result

        receipt = receipt or {}
        receipt["resolved_at"] = _utcnow().isoformat()
        self._settle(approval, resolution, receipt=receipt)
        return approval, True

    # ------------------------------------------------------------------
    # One-shot tokens
    # ------------------------------------------------------------------

    def _issue_token(self, approval: Approval, *, tool: str, args: dict[str, Any]) -> str:
        raw = secrets.token_urlsafe(24)
        token = ApprovalToken(
            approval_id=approval.id,
            token_hash=_hash(raw),
            payload={"tool": tool, "args": args, "snapshot_v": args.get("snapshot_v")},
            expires_at=_utcnow() + timedelta(seconds=TOKEN_TTL_SECONDS),
        )
        self.session.add(token)
        self.session.commit()
        return raw

    def consume_token(self, raw: str, *, tool: str, args: dict[str, Any]) -> ApprovalToken:
        """Spend a token: it must exist, be unconsumed, unexpired, and carry
        EXACTLY the payload the caller is trying to execute."""
        token = self.session.exec(
            select(ApprovalToken).where(ApprovalToken.token_hash == _hash(raw))
        ).first()
        if token is None:
            raise ValueError("invalid approval token")
        if token.consumed_at is not None:
            raise ValueError("approval token already consumed")
        if _as_utc(token.expires_at) <= _utcnow():
            raise ValueError("approval token expired")
        payload = token.payload if isinstance(token.payload, dict) else {}
        if payload.get("tool") != tool or payload.get("args") != args:
            raise ValueError("approval token payload mismatch")
        token.consumed_at = _utcnow()
        self.session.add(token)
        self.session.commit()
        return token

    # ------------------------------------------------------------------
    # Expiry sweep (scheduler poll + boot)
    # ------------------------------------------------------------------

    def sweep_expired(self, now: datetime | None = None) -> int:
        now = now or _utcnow()
        overdue = self.session.exec(
            select(Approval).where(
                Approval.status == "pending",
                Approval.expires_at <= now,
            )
        ).all()
        for approval in overdue:
            self._settle(approval, "expired", receipt=None)
        return len(overdue)

    # ------------------------------------------------------------------

    def _settle(self, approval: Approval, status: str, receipt: dict[str, Any] | None) -> None:
        approval.status = status
        approval.resolved_at = _utcnow()
        approval.receipt = receipt
        self.session.add(approval)
        self.session.commit()
        self.session.refresh(approval)
        summary = _summary_of(approval)
        verb = {
            "approved": "Approved",
            "edited": "Edited & approved",
            "skipped": "Skipped",
            "expired": "Expired",
        }.get(status, status.title())
        ConversationService(self.session).save_assistant_turn(
            approval.conversation_id,
            f"{verb}: {summary}",
            [{"type": EVENT_RESOLVED, "approval": ApprovalResponse.serialize(approval)}],
        )


def _summary_of(approval: Approval) -> str:
    try:
        descriptor = parse_descriptor(approval.action_descriptor)
    except Exception:
        return approval.kind
    if isinstance(descriptor, ActionDescriptorV1):
        return descriptor.summary or f"{descriptor.tool} action"
    return f"Sign in to {descriptor.app_name}"
