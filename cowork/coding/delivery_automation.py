from __future__ import annotations

from cowork.coding.contracts import (
    CodingSession,
    DeliveryAutomationClaim,
    DeliveryAutomationClaimRequest,
    DeliveryAutomationPolicy,
)
from cowork.coding.store import CodingStore


class DeliveryAutomationService:
    """Persist opt-in delivery policy and atomically bound automatic fix attempts."""

    def __init__(self, store: CodingStore) -> None:
        self._store = store

    def update_policy(self, session_id: str, policy: DeliveryAutomationPolicy) -> CodingSession:
        return self._store.update_session(
            session_id,
            lambda current: setattr(current, "delivery_policy", policy),
        )

    def claim_fix(self, session_id: str, request: DeliveryAutomationClaimRequest) -> DeliveryAutomationClaim:
        claimed = False
        attempts = 0
        limit = 0

        def claim(current: CodingSession) -> None:
            nonlocal claimed, attempts, limit
            limit = current.delivery_policy.max_fix_attempts
            attempts = current.delivery_automation.fix_attempts.get(request.fingerprint, 0)
            if not current.delivery_policy.fix_failing_checks or attempts >= limit:
                return
            if request.fingerprint not in current.delivery_automation.fix_attempts:
                self._prune_oldest_attempt(current)
            attempts += 1
            current.delivery_automation.fix_attempts[request.fingerprint] = attempts
            claimed = True

        self._store.update_session(session_id, claim)
        return DeliveryAutomationClaim(claimed=claimed, attempts=attempts, limit=limit)

    @staticmethod
    def _prune_oldest_attempt(session: CodingSession) -> None:
        attempts = session.delivery_automation.fix_attempts
        if len(attempts) >= 128:
            attempts.pop(next(iter(attempts)))
