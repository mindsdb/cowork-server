"""Standing rules (R1): durable "Always" permissions, enforced deterministically.

Grant path is evidence-gated: "Always" only becomes an option after
ALWAYS_EVIDENCE_COUNT identical, UNMODIFIED approvals (status 'approved',
never 'edited') of the same scope (origin + action_kind). The gate does an
exact-match lookup at act time — never a memory retrieval — and revocation
is one click on the Memories shelf (R2 renders it).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.approval import Approval
from cowork.models.standing_rule import StandingRule
from cowork.schemas.approvals import parse_descriptor

ALWAYS_EVIDENCE_COUNT = 3


def normalize_label(label: str) -> str:
    """The action-kind's label half: lowercase, [!] stripped, collapsed
    whitespace — 'Send', '[!] Send', '  send ' all scope to 'send'."""
    text = label.strip()
    while text.startswith("[!]"):
        text = text[3:].strip()
    return " ".join(text.lower().split())


def scope_of(*, origin: str, tool: str, label: str) -> tuple[str, str]:
    return origin.strip().lower(), f"{tool}:{normalize_label(label)}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RuleService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Grant / revoke / lookup
    # ------------------------------------------------------------------

    def grant(self, *, origin: str, action_kind: str, source_approval_id: UUID) -> StandingRule:
        existing = self._active(origin, action_kind)
        if existing is not None:
            return existing
        rule = StandingRule(
            origin=origin,
            action_kind=action_kind,
            source_approval_id=source_approval_id,
        )
        self.session.add(rule)
        self.session.commit()
        self.session.refresh(rule)
        return rule

    def revoke(self, rule_id: UUID) -> StandingRule:
        rule = self.session.get(StandingRule, rule_id)
        if rule is None:
            raise ValueError("Standing rule not found")
        if rule.revoked_at is None:
            rule.revoked_at = _utcnow()
            self.session.add(rule)
            self.session.commit()
            self.session.refresh(rule)
        return rule

    def matching(self, *, origin: str, action_kind: str) -> StandingRule | None:
        """The active rule for this exact scope, or None. This IS the
        revocation check — revoked rules never match."""
        return self._active(origin, action_kind)

    def record_hit(self, rule: StandingRule) -> None:
        rule.hit_count += 1
        rule.last_fired_at = _utcnow()
        self.session.add(rule)
        self.session.commit()

    def list(self, *, include_revoked: bool = False) -> list[StandingRule]:
        stmt = select(StandingRule).order_by(StandingRule.created_at.desc())  # type: ignore[attr-defined]
        if not include_revoked:
            stmt = stmt.where(StandingRule.revoked_at.is_(None))
        return list(self.session.exec(stmt).all())

    # ------------------------------------------------------------------
    # Evidence gate
    # ------------------------------------------------------------------

    def eligible_for_always(self, *, origin: str, action_kind: str) -> bool:
        """True once the user has approved the SAME scope unchanged at least
        ALWAYS_EVIDENCE_COUNT times (status 'approved' exactly — an edited
        approval proves nothing about wanting it always-that-way)."""
        rows = self.session.exec(
            select(Approval).where(Approval.status == "approved")
        ).all()
        count = 0
        for approval in rows:
            try:
                descriptor = parse_descriptor(approval.action_descriptor)
            except Exception:
                continue
            desc_origin = (descriptor.args.get("origin") or "").lower() if hasattr(descriptor, "args") else ""
            desc_label = getattr(descriptor, "summary", "")
            desc_tool = getattr(descriptor, "tool", "")
            if desc_origin != origin.lower():
                continue
            if f"{desc_tool}:{normalize_label(desc_label)}" == action_kind:
                count += 1
        return count >= ALWAYS_EVIDENCE_COUNT

    # ------------------------------------------------------------------

    def _active(self, origin: str, action_kind: str) -> StandingRule | None:
        return self.session.exec(
            select(StandingRule).where(
                StandingRule.origin == origin,
                StandingRule.action_kind == action_kind,
                StandingRule.revoked_at.is_(None),
            )
        ).first()
