from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import timedelta

from cowork.coding.contracts import utc_now
from cowork.coding.control_errors import RuntimeAuthenticationError
from cowork.coding.control_models import (
    TERMINAL_RUN_STATUSES,
    ConnectorGrant,
    RunStatus,
    TaskRun,
)
from cowork.coding.control_store import ControlPlaneStore
from cowork.coding.security_audit import record_security_event


class ConnectorDelegationService:
    """Issue, authorize, and revoke bounded connector capabilities."""

    def __init__(self, store: ControlPlaneStore) -> None:
        self.store = store

    def issue(
        self,
        run_id: str,
        provider: str,
        connection_name: str,
        actions: list[str],
        resource_constraints: dict[str, str] | None = None,
        ttl: timedelta = timedelta(minutes=15),
    ) -> tuple[ConnectorGrant, str]:
        run = self.store.get_run(run_id)
        if run.status in {RunStatus.completed, RunStatus.cancelled, RunStatus.failed}:
            raise ValueError("Connector access is unavailable after a Task Run ends")
        token = secrets.token_urlsafe(40)
        grant = ConnectorGrant(
            id=f"grant-{uuid.uuid4().hex}",
            run_id=run.id,
            computer_id=run.computer_id,
            epoch=run.epoch,
            provider=provider,
            connection_name=connection_name,
            actions=actions,
            resource_constraints=resource_constraints or {},
            token_hash=self._digest(token),
            expires_at=utc_now() + ttl,
        )
        saved = self.store.save_grant(grant)
        record_security_event(
            self.store,
            "connector.grant",
            "allowed",
            "system",
            grant.id,
            run_id=run.id,
            computer_id=run.computer_id,
            detail=f"{provider}:{','.join(actions)}",
        )
        return saved, token

    def authorize(
        self,
        grant_id: str,
        token: str,
        action: str,
        constraints: dict[str, str] | None = None,
        computer_id: str | None = None,
    ) -> ConnectorGrant:
        grant_snapshot: ConnectorGrant | None = None
        run_snapshot: TaskRun | None = None
        try:
            grant_snapshot = self.store.get_grant(grant_id)

            def authorize_grant(grant: ConnectorGrant, run: TaskRun) -> None:
                nonlocal run_snapshot
                run_snapshot = run
                self._validate(grant, run, token, action, constraints, computer_id)
                grant.use_count += 1
                grant.last_used_at = utc_now()

            grant = self.store.update_grant(grant_id, authorize_grant)
            if run_snapshot is None:
                raise RuntimeAuthenticationError("Connector capability could not be validated")
            record_security_event(
                self.store,
                "connector.invoke",
                "allowed",
                "agent",
                grant.id,
                run_id=run_snapshot.id,
                computer_id=computer_id or run_snapshot.computer_id,
                detail=f"{grant.provider}:{action}",
            )
            return grant
        except RuntimeAuthenticationError as exc:
            record_security_event(
                self.store,
                "connector.invoke",
                "denied",
                "agent",
                grant_id,
                run_id=grant_snapshot.run_id if grant_snapshot else None,
                computer_id=computer_id,
                detail=str(exc),
            )
            raise

    def revoke_for_run(self, run_id: str) -> None:
        now = utc_now()
        for grant in self.store.list_grants(run_id):
            if grant.revoked_at is not None:
                continue
            grant.revoked_at = now
            self.store.save_grant(grant)
            record_security_event(
                self.store,
                "connector.revoke",
                "completed",
                "system",
                grant.id,
                run_id=run_id,
                computer_id=grant.computer_id,
            )

    @classmethod
    def _validate(
        cls,
        grant: ConnectorGrant,
        run: TaskRun,
        token: str,
        action: str,
        constraints: dict[str, str] | None,
        computer_id: str | None,
    ) -> None:
        if grant.revoked_at or grant.expires_at <= utc_now():
            raise RuntimeAuthenticationError("Connector capability expired")
        if grant.epoch != run.epoch or run.status in TERMINAL_RUN_STATUSES:
            raise RuntimeAuthenticationError("Connector capability belongs to a stale Task Run")
        if grant.computer_id != run.computer_id or (
            computer_id is not None and grant.computer_id != computer_id
        ):
            raise RuntimeAuthenticationError("Connector capability belongs to another computer")
        if not hmac.compare_digest(grant.token_hash, cls._digest(token)):
            raise RuntimeAuthenticationError("Connector capability authentication failed")
        if action not in grant.actions:
            raise RuntimeAuthenticationError("Connector capability does not allow this action")
        cls._validate_constraints(grant, action, constraints or {})
        if grant.use_count >= grant.max_uses:
            raise RuntimeAuthenticationError("Connector capability use limit reached")

    @staticmethod
    def _validate_constraints(
        grant: ConnectorGrant,
        action: str,
        requested: dict[str, str],
    ) -> None:
        action_constraint_names = {
            "read_source": ("url",),
            "pull_request_status": ("target_url",),
            "search_work": ("repository",),
        }.get(action, tuple(grant.resource_constraints))
        required_names = tuple(
            name for name in action_constraint_names if name in grant.resource_constraints
        ) or tuple(grant.resource_constraints)
        for name in required_names:
            allowed = grant.resource_constraints.get(name)
            if allowed is None:
                continue
            supplied = requested.get(name)
            if supplied is None or not hmac.compare_digest(allowed, supplied):
                raise RuntimeAuthenticationError("Connector capability is outside its resource scope")

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
