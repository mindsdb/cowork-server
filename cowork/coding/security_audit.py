from __future__ import annotations

import uuid
from typing import Literal

from cowork.coding.control_models import SecurityAuditEvent
from cowork.coding.control_store import ControlPlaneStore


def record_security_event(
    store: ControlPlaneStore,
    action: str,
    outcome: Literal["allowed", "denied", "completed"],
    actor_type: Literal["user", "runtime", "agent", "system"],
    target_id: str,
    *,
    run_id: str | None = None,
    computer_id: str | None = None,
    detail: str = "",
) -> None:
    """Append a secret-free record of a sensitive control-plane decision."""

    store.save_audit_event(SecurityAuditEvent(
        id=f"audit-{uuid.uuid4().hex}",
        action=action,
        outcome=outcome,
        actor_type=actor_type,
        target_id=target_id,
        run_id=run_id,
        computer_id=computer_id,
        detail=detail,
    ))
