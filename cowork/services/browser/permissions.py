"""Browser permission enforcement (Milestone 1).

`BrowserPermissionService.check` resolves whether a session may perform an
action of a given class against a given host-only domain, following the M1
cross-domain policy: one approved tab / one active domain grant, same-host
only. `PolicyHook` is a no-op seam for the post-M1 Governance milestone
(org role/domain/action/retention policy) — it lets a future policy layer
veto or annotate a decision without changing call sites now.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.browser import BrowserTabGrant
from cowork.schemas.browser import (
    BrowserActionClass,
    PermissionDecision,
    coerce_enum,
    coerce_uuid,
    host_only,
)


@dataclass(frozen=True)
class PermissionResult:
    """Outcome of a permission check."""

    decision: PermissionDecision
    domain: str
    action_class: BrowserActionClass
    reason: str | None = None

    @property
    def granted(self) -> bool:
        return self.decision == PermissionDecision.granted


class PolicyHook:
    """No-op governance seam (post-M1).

    A future org-policy layer subclasses this and overrides `evaluate` to
    veto or downgrade a decision. In M1 it always passes the decision
    through unchanged so the call site is already policy-aware.
    """

    def evaluate(self, result: PermissionResult) -> PermissionResult:
        return result


class BrowserPermissionService:
    """Checks per-domain, per-action-class grants for a browser session."""

    def __init__(self, session: Session, policy_hook: PolicyHook | None = None) -> None:
        self._session = session
        self._policy = policy_hook or PolicyHook()

    def check(
        self,
        session_id: UUID | str,
        domain: str,
        action_class: BrowserActionClass | str,
    ) -> PermissionResult:
        """Resolve granted/denied/expired/revoked for a session+domain+class.

        The `domain` is reduced to its bare host before matching — a grant
        covers a registrable host, never a full URL. A navigate that would
        leave the approved host is `denied` (cross-domain policy). A `read`
        action is allowed under either a `read` or a `navigate` grant on the
        same host (navigate implies the tab is inspectable).
        """
        sid = coerce_uuid(session_id)
        host = host_only(domain)
        ac = coerce_enum(BrowserActionClass, action_class)

        if not host:
            return self._policy.evaluate(
                PermissionResult(
                    decision=PermissionDecision.denied,
                    domain=host,
                    action_class=ac,
                    reason="no target domain",
                )
            )

        grants = self._session.exec(
            select(BrowserTabGrant).where(
                BrowserTabGrant.session_id == sid,
                BrowserTabGrant.domain == host,
            )
        ).all()

        if not grants:
            return self._policy.evaluate(
                PermissionResult(
                    decision=PermissionDecision.denied,
                    domain=host,
                    action_class=ac,
                    reason="no grant for domain",
                )
            )

        now = datetime.now(timezone.utc)
        # A read action is satisfied by any active grant on the host; a
        # navigate needs a navigate-class grant.
        acceptable_classes = (
            {BrowserActionClass.read.value, BrowserActionClass.navigate.value}
            if ac == BrowserActionClass.read
            else {BrowserActionClass.navigate.value}
        )

        best_revoked = False
        best_expired = False
        for grant in grants:
            if grant.action_class not in acceptable_classes:
                continue
            if grant.decision == PermissionDecision.revoked.value:
                best_revoked = True
                continue
            if grant.decision == PermissionDecision.denied.value:
                continue
            if grant.decision == PermissionDecision.expired.value:
                best_expired = True
                continue
            # granted — check expiry.
            if grant.expires_at is not None:
                exp = grant.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp <= now:
                    best_expired = True
                    continue
            return self._policy.evaluate(
                PermissionResult(
                    decision=PermissionDecision.granted,
                    domain=host,
                    action_class=ac,
                )
            )

        if best_revoked:
            decision = PermissionDecision.revoked
            reason = "grant revoked"
        elif best_expired:
            decision = PermissionDecision.expired
            reason = "grant expired"
        else:
            decision = PermissionDecision.denied
            reason = "no grant for action class"
        return self._policy.evaluate(
            PermissionResult(
                decision=decision,
                domain=host,
                action_class=ac,
                reason=reason,
            )
        )
