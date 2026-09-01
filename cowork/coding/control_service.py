from __future__ import annotations

import hashlib
import hmac
import os
import platform
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cowork.coding.connector_delegation import ConnectorDelegationService
from cowork.coding.contracts import CodingSession, SessionStatus, TaskCapability, utc_now
from cowork.coding.control_errors import RuntimeAuthenticationError, StaleRuntimeEvent, StateConflict
from cowork.coding.control_models import (
    RUNTIME_PROTOCOL_VERSION,
    TERMINAL_RUN_STATUSES,
    CodeTask,
    Computer,
    ComputerCapabilities,
    ComputerPage,
    ComputerStatus,
    ConnectorGrant,
    ExecutionWorkspace,
    RecoveryPlan,
    ResourceAvailability,
    ResourceAvailabilityPage,
    RunStatus,
    RuntimeCommand,
    RuntimeCredential,
    RuntimeEvent,
    RuntimeRegistrationCredential,
    TaskControlSnapshot,
    TaskResourceScope,
    TaskRun,
    TaskRunCredential,
    WorkspaceStatus,
)
from cowork.coding.control_store import ControlPlaneStore, LocalControlPlaneStore
from cowork.coding.project_models import (
    CodeProject,
    LocalFolderResource,
    ProjectResource,
    RepositoryResource,
)
from cowork.coding.redaction import redact_text, sanitize
from cowork.coding.run_recovery import (
    NoEligibleComputer,
    build_recovery_plan,
)
from cowork.coding.run_recovery import (
    recover_run as apply_recovery,
)
from cowork.coding.run_state import transition_run
from cowork.coding.security_audit import record_security_event

if TYPE_CHECKING:
    from cowork.coding.contracts import TaskWorkspace


_OFFLINE_AFTER = timedelta(seconds=35)
_LEASE_DURATION = timedelta(seconds=30)
_COMMAND_CLAIM_DURATION = timedelta(seconds=20)
_SESSION_STATUS: dict[SessionStatus, RunStatus] = {
    SessionStatus.ready: RunStatus.ready,
    SessionStatus.running: RunStatus.running,
    SessionStatus.awaiting_approval: RunStatus.awaiting_approval,
    SessionStatus.completed: RunStatus.completed,
    SessionStatus.cancelled: RunStatus.cancelled,
    SessionStatus.interrupted: RunStatus.interrupted,
    SessionStatus.failed: RunStatus.failed,
}


class ControlPlaneService:
    """Durable Task/Run/Computer orchestration above local execution details."""

    def __init__(
        self,
        root: Path,
        capabilities: ComputerCapabilities,
        store: ControlPlaneStore | None = None,
    ) -> None:
        self.root = root
        self.store = store or LocalControlPlaneStore(root)
        self.connector_delegation = ConnectorDelegationService(self.store)
        self._lock = threading.RLock()
        # Protocol records are useful for retries and audit, but are not user
        # task history. Bound them at service start so a long-lived desktop or
        # tenant cannot accumulate an ever-growing polling surface.
        now = utc_now()
        self.store.prune(now - timedelta(days=30), now - timedelta(days=90))
        self.local_computer = self._register_local(capabilities)

    @staticmethod
    def default_capabilities(agent_engines: list[str], shells: list[str]) -> ComputerCapabilities:
        system = platform.system().lower()
        normalized = "darwin" if system == "darwin" else "windows" if system == "windows" else "linux"
        return ComputerCapabilities(
            platform=normalized,
            architecture=platform.machine() or "unknown",
            runtime_version="cowork-desktop-1",
            agent_engines=agent_engines,
            shells=shells,
            has_git=True,
            has_terminal=True,
            supports_local_folders=True,
            task_capabilities=list(TaskCapability),
        )

    def list_computers(self) -> ComputerPage:
        self.heartbeat(self.local_computer.id, active_run_count=self._active_count(self.local_computer.id))
        self.expire_stale_computers()
        return ComputerPage(items=[item for item in self.store.list_computers() if item.revoked_at is None])

    def heartbeat(self, computer_id: str, active_run_count: int = 0) -> Computer:
        computer = self.store.get_computer(computer_id)
        if computer.revoked_at is not None:
            raise RuntimeAuthenticationError("This computer has been revoked")
        computer.last_seen_at = utc_now()
        computer.updated_at = computer.last_seen_at
        computer.active_run_count = active_run_count
        if computer.status != ComputerStatus.draining:
            computer.status = ComputerStatus.online
        return self.store.save_computer(computer)

    def rename_computer(self, computer_id: str, name: str) -> Computer:
        computer = self.store.get_computer(computer_id)
        if computer.revoked_at is not None:
            raise KeyError("Computer not found")
        computer.name = self._computer_name(name)
        if computer.is_local:
            self._write_local_computer_name(computer.name)
        return self.store.save_computer(computer)

    def revoke_computer(self, computer_id: str) -> None:
        computer = self.store.get_computer(computer_id)
        if computer.revoked_at is not None:
            raise KeyError("Computer not found")
        if computer.is_local:
            raise ValueError("This computer cannot be revoked from itself")
        if any(
            run.computer_id == computer_id and run.status not in TERMINAL_RUN_STATUSES
            for run in self.store.list_runs()
        ):
            raise StateConflict("Finish or move this computer's active tasks before revoking it")
        computer.registration_epoch += 1
        computer.status = ComputerStatus.offline
        computer.active_run_count = 0
        computer.revoked_at = utc_now()
        self.store.save_computer(computer)

    def issue_registration_token(self) -> str:
        token = secrets.token_urlsafe(36)
        digest = self._digest(token)
        self.store.save_registration_credential(RuntimeRegistrationCredential(
            id=digest,
            token_hash=digest,
            expires_at=utc_now() + timedelta(minutes=10),
        ))
        return token

    def register_runtime(
        self,
        registration_token: str,
        name: str,
        capabilities: ComputerCapabilities,
    ) -> tuple[Computer, str]:
        self._require_protocol(capabilities.protocol_versions)
        token_hash = self._digest(registration_token)
        if not self.store.consume_registration_credential(token_hash, utc_now()):
            raise RuntimeAuthenticationError("Runtime registration expired or was already used")
        identifier = f"computer-{uuid.uuid4().hex}"
        computer = Computer(
            id=identifier,
            name=self._computer_name(name),
            is_local=False,
            capabilities=capabilities,
        )
        runtime_token = secrets.token_urlsafe(40)
        self.store.save_runtime_credential(RuntimeCredential(
            id=identifier,
            computer_id=identifier,
            token_hash=self._digest(runtime_token),
            registration_epoch=computer.registration_epoch,
        ))
        record_security_event(self.store,
            "runtime.register",
            "completed",
            "runtime",
            computer.id,
            computer_id=computer.id,
            detail=f"protocol={RUNTIME_PROTOCOL_VERSION}",
        )
        return self.store.save_computer(computer), runtime_token

    def authenticate_runtime(self, computer_id: str, runtime_token: str) -> Computer:
        try:
            credential = self.store.get_runtime_credential(computer_id)
        except KeyError as exc:
            raise RuntimeAuthenticationError("Runtime authentication failed") from exc
        computer = self.store.get_computer(computer_id)
        if computer.revoked_at is not None:
            raise RuntimeAuthenticationError("This computer has been revoked")
        if (
            credential.registration_epoch != computer.registration_epoch
            or not hmac.compare_digest(credential.token_hash, self._digest(runtime_token))
        ):
            raise RuntimeAuthenticationError("Runtime authentication failed")
        return computer

    def issue_run_token(self, run_id: str) -> str:
        """Mint one epoch-fenced capability for the agent inside a Task Run."""

        run = self.store.get_run(run_id)
        if not run.lease_id or run.status not in {
            RunStatus.preparing,
            RunStatus.ready,
            RunStatus.running,
            RunStatus.awaiting_approval,
        }:
            raise RuntimeAuthenticationError("Task Run is not actively leased")
        token = secrets.token_urlsafe(40)
        self.store.save_run_credential(TaskRunCredential(
            id=run.id,
            run_id=run.id,
            computer_id=run.computer_id,
            epoch=run.epoch,
            token_hash=self._digest(token),
        ))
        return token

    def authenticate_run_token(self, run_id: str, computer_id: str, token: str) -> TaskRun:
        try:
            credential = self.store.get_run_credential(run_id)
            run = self.store.get_run(run_id)
        except KeyError as exc:
            raise RuntimeAuthenticationError("Task Run authentication failed") from exc
        if (
            credential.computer_id != computer_id
            or credential.epoch != run.epoch
            or run.computer_id != computer_id
            or not run.lease_id
            or run.lease_expires_at is None
            or run.lease_expires_at < utc_now()
            or run.status in TERMINAL_RUN_STATUSES
            or not hmac.compare_digest(credential.token_hash, self._digest(token))
        ):
            raise RuntimeAuthenticationError("Task Run authentication failed")
        return run

    def eligible_computers(
        self,
        project: CodeProject | None,
        scope: TaskResourceScope,
        engine_id: str | None = None,
    ) -> list[Computer]:
        resources = self.scoped_resources(project, scope)
        eligible = []
        for computer in self.list_computers().items:
            capabilities = computer.capabilities
            if computer.status != ComputerStatus.online:
                continue
            active_runs = max(computer.active_run_count, self._active_count(computer.id))
            if active_runs >= capabilities.max_concurrent_runs:
                continue
            if engine_id and engine_id not in capabilities.agent_engines:
                continue
            if all(self._resource_eligible(item, computer) for item in resources):
                eligible.append(computer)
        return eligible

    def resource_availability(self, project: CodeProject) -> ResourceAvailabilityPage:
        computers = self.list_computers().items
        items: list[ResourceAvailability] = []
        for resource in project.resources:
            eligible = [
                computer.id
                for computer in computers
                if computer.status == ComputerStatus.online and self._resource_eligible(resource, computer)
            ]
            required = resource.computer_id if isinstance(resource, LocalFolderResource) else None
            if eligible:
                status, detail = "available", ""
            elif required and any(computer.id == required for computer in computers):
                status, detail = "offline", "The computer with this folder is offline"
            else:
                status, detail = "unavailable", "No registered computer can access this resource"
            items.append(ResourceAvailability(
                resource_id=resource.id,
                status=status,
                eligible_computer_ids=eligible,
                required_computer_id=required,
                detail=detail,
            ))
        return ResourceAvailabilityPage(items=items)

    def create_task_run(
        self,
        task_id: str,
        title: str,
        prompt: str,
        project: CodeProject | None,
        requested_resource_ids: list[str] | None,
        computer_id: str | None,
        engine_id: str,
        standalone_computer_id: str | None = None,
    ) -> TaskControlSnapshot:
        with self._lock:
            return self._create_task_run(
                task_id=task_id,
                title=title,
                prompt=prompt,
                project=project,
                requested_resource_ids=requested_resource_ids,
                computer_id=computer_id,
                engine_id=engine_id,
                standalone_computer_id=standalone_computer_id,
            )

    def _create_task_run(
        self,
        task_id: str,
        title: str,
        prompt: str,
        project: CodeProject | None,
        requested_resource_ids: list[str] | None,
        computer_id: str | None,
        engine_id: str,
        standalone_computer_id: str | None,
    ) -> TaskControlSnapshot:
        scope = TaskResourceScope(
            all_project_resources=requested_resource_ids is None,
            resource_ids=requested_resource_ids or [],
        )
        resources = self.scoped_resources(project, scope)
        eligible = self.eligible_computers(project, scope, engine_id)
        if project is None and standalone_computer_id:
            eligible = [item for item in eligible if item.id == standalone_computer_id]
        selected = next((item for item in eligible if item.id == computer_id), None) if computer_id else None
        if computer_id and selected is None:
            raise NoEligibleComputer("That computer cannot access every resource selected for this task")
        if selected is None:
            if not eligible:
                raise NoEligibleComputer("No online computer can access every resource selected for this task")
            selected = eligible[0]
        task = CodeTask(
            id=task_id,
            title=title,
            prompt=prompt,
            engine_id=engine_id,
            project_id=project.id if project else None,
            resource_scope=scope,
            execution_project=self.execution_project_snapshot(project, resources),
        )
        run = TaskRun(
            id=f"run-{uuid.uuid4().hex}",
            task_id=task.id,
            computer_id=selected.id,
        )
        workspaces = [ExecutionWorkspace(
            id=f"workspace-{uuid.uuid4().hex}",
            run_id=run.id,
            resource_id=resource.id,
            computer_id=selected.id,
        ) for resource in resources]
        task, run, workspaces = self.store.create_task_run(task, run, workspaces)
        return TaskControlSnapshot(task=task, run=run, computer=selected, workspaces=workspaces)

    def runtime_project(self, project: CodeProject, scope: TaskResourceScope, computer_id: str) -> CodeProject:
        resources: list[ProjectResource] = []
        for resource in self.scoped_resources(project, scope):
            if (
                isinstance(resource, RepositoryResource)
                and resource.source_url
                and resource.computer_id != computer_id
            ):
                resources.append(resource.model_copy(update={"local_path": None, "computer_id": None}))
            else:
                resources.append(resource)
        return CodeProject.model_validate({
            **project.model_dump(mode="python"),
            "resources": resources,
        })

    def runtime_project_for_task(self, task: CodeTask, computer_id: str) -> CodeProject | None:
        """Resolve a lease from the immutable task snapshot, never live project state."""

        if task.execution_project is None:
            return None
        return self.runtime_project(
            task.execution_project,
            TaskResourceScope(all_project_resources=True),
            computer_id,
        )

    def set_run_status(self, run_id: str, status: RunStatus, *, error: str | None = None) -> TaskRun:
        return self.store.update_run(run_id, lambda run: transition_run(run, status, error=error))

    def continue_task(self, task_id: str, previous_run_id: str) -> TaskRun:
        previous = self.store.get_run(previous_run_id)
        if previous.task_id != task_id:
            raise ValueError("Task Run belongs to another task")
        computer = self.store.get_computer(previous.computer_id)
        if computer.status != ComputerStatus.online:
            raise NoEligibleComputer("The task's computer is offline")
        workspaces = self.store.list_workspaces(previous.id)
        restorable = bool(workspaces) and all(
            item.status == WorkspaceStatus.ready and item.path for item in workspaces
        )
        run = self.store.save_run(TaskRun(
            id=f"run-{uuid.uuid4().hex}",
            task_id=task_id,
            computer_id=computer.id,
            status=RunStatus.ready if computer.id == self.local_computer.id else RunStatus.queued,
            epoch=previous.epoch + 1,
            workspace_resume_mode="restore" if restorable else "prepare",
            recovery_count=previous.recovery_count,
        ))
        for workspace in workspaces:
            record = workspace.model_copy(update={
                "id": f"workspace-{uuid.uuid4().hex}",
                "run_id": run.id,
            }) if restorable else ExecutionWorkspace(
                id=f"workspace-{uuid.uuid4().hex}",
                run_id=run.id,
                resource_id=workspace.resource_id,
                computer_id=computer.id,
            )
            self.store.save_workspace(record)
        return run

    def attach_prepared_workspaces(self, run_id: str, prepared: list[TaskWorkspace]) -> list[ExecutionWorkspace]:
        run = self.store.get_run(run_id)
        existing = {item.resource_id: item for item in self.store.list_workspaces(run_id)}
        saved = []
        for workspace in prepared:
            record = existing.get(workspace.folder_id) or ExecutionWorkspace(
                id=f"workspace-{uuid.uuid4().hex}",
                run_id=run.id,
                resource_id=workspace.folder_id,
                computer_id=run.computer_id,
            )
            record.status = WorkspaceStatus.ready
            record.computer_id = run.computer_id
            record.path = workspace.workspace_path
            record.workspace_kind = workspace.workspace_kind
            record.base_revision = workspace.base_revision
            record.task_branch = workspace.task_branch
            record.detail = ""
            saved.append(self.store.save_workspace(record))
        return saved

    def release_workspaces(self, run_id: str) -> list[ExecutionWorkspace]:
        released = []
        for workspace in self.store.list_workspaces(run_id):
            workspace.status = WorkspaceStatus.released
            workspace.path = ""
            workspace.detail = "Workspace released"
            released.append(self.store.save_workspace(workspace))
        return released

    def migrate_session(self, session: CodingSession, project: CodeProject | None) -> TaskControlSnapshot:
        task_id = session.task_id or session.id
        run_id = session.run_id or f"run-{session.id}"
        computer_id = session.computer_id or self.local_computer.id
        resource_ids = session.resource_ids or [item.folder_id for item in session.workspaces]
        if project and not resource_ids:
            resource_ids = [item.id for item in project.resources]
        scope = TaskResourceScope(
            all_project_resources=session.scope_all_project_resources,
            resource_ids=[] if session.scope_all_project_resources else resource_ids,
        )
        try:
            task = self.store.get_task(task_id)
        except KeyError:
            task = self.store.save_task(CodeTask(
                id=task_id,
                title=session.title,
                engine_id=session.engine_id,
                project_id=session.project_id,
                resource_scope=scope,
                execution_project=self.execution_project_snapshot(
                    project,
                    self.scoped_resources(project, scope),
                ),
                source_contexts=session.source_contexts,
                deliveries=session.deliveries,
            ))
        try:
            run = self.store.get_run(run_id)
        except KeyError:
            run = self.store.save_run(TaskRun(
                id=run_id,
                task_id=task.id,
                computer_id=computer_id,
                status=_SESSION_STATUS[session.status],
                epoch=session.runtime_epoch,
            ))
        workspaces = self.store.list_workspaces(run.id)
        if not workspaces:
            source = session.workspaces or [self._fallback_workspace(session)]
            workspaces = [self.store.save_workspace(self._execution_workspace(run, item)) for item in source]
        return TaskControlSnapshot(
            task=task,
            run=run,
            computer=self.store.get_computer(computer_id),
            workspaces=workspaces,
        )

    def sync_session(self, session: CodingSession) -> TaskRun:
        if not session.run_id:
            raise StateConflict("Coding session has not been linked to a Task Run")

        def reconcile(run: TaskRun) -> None:
            if run.computer_id != self.local_computer.id:
                # A remote runtime owns the canonical Task Run lifecycle.  The
                # CodingSession is only a compatibility read model for the
                # renderer, so ordinary UI events must never project its stale
                # status back into a leased/fenced remote run.
                return
            transition_run(run, _SESSION_STATUS[session.status], error=session.last_error)

        return self.store.update_run(session.run_id, reconcile)

    def acquire_lease(self, computer_id: str) -> tuple[TaskRun, str] | None:
        with self._lock:
            if self.store.get_computer(computer_id).revoked_at is not None:
                raise RuntimeAuthenticationError("This computer has been revoked")
            self.expire_leases()
            lease_id = secrets.token_urlsafe(32)
            run = self.store.claim_run(
                computer_id,
                lease_id,
                utc_now() + _LEASE_DURATION,
            )
            if run is None:
                return None
            return run, lease_id

    def accept_event(
        self,
        event: RuntimeEvent,
        apply: Callable[[TaskRun], None] | None = None,
    ) -> TaskRun:
        """Apply one fenced runtime event and record its sequence in a single run write.

        ``apply`` runs inside the same store operation, so the sequence never
        advances unless every effect ran. In the SQL store those effects share
        the run's transaction and roll back with it; in the local store each
        write lands on its own, so a crash mid-way leaves them applied without
        the sequence and redelivery redoes them (at-least-once). Redelivering
        the last applied event returns the same acknowledgement without
        re-applying it; only an older sequence, or a different event reusing
        one, is stale.
        """

        if event.protocol_version != RUNTIME_PROTOCOL_VERSION:
            raise StaleRuntimeEvent("Runtime protocol version is not supported")

        def operation(run: TaskRun) -> None:
            if event.seq == run.last_event_seq and event.id == run.last_event_id:
                # A terminal event clears the lease as part of the same atomic
                # transition. If its acknowledgement is lost, the authenticated
                # runtime must still be able to redeliver that exact event and
                # receive the same acknowledgement. Keep the ownership fence,
                # but do not require a lease that the accepted event removed.
                self._require_event_owner(run, event.computer_id, event.epoch)
                return
            self._require_fence(run, event.computer_id, event.lease_id, event.epoch)
            run.lease_expires_at = utc_now() + _LEASE_DURATION
            if event.seq <= run.last_event_seq:
                raise StaleRuntimeEvent("Runtime event sequence is stale")
            self._apply_event(run, event)
            if apply is not None:
                apply(run)
            run.last_event_seq = event.seq
            run.last_event_id = event.id

        with self._lock:
            return self.store.update_run(event.run_id, operation)

    @staticmethod
    def _apply_event(run: TaskRun, event: RuntimeEvent) -> None:
        if event.kind == "checkpoint":
            checkpoint = sanitize(event.payload)
            run.checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        elif event.kind == "status" and isinstance(event.payload.get("status"), str):
            transition_run(run, RunStatus(event.payload["status"]))
        elif event.kind == "turn_completed":
            transition_run(run, RunStatus.ready)
        elif event.kind == "error":
            transition_run(
                run,
                RunStatus.failed,
                error=redact_text(str(event.payload.get("detail", "Runtime failed")))[:4_000],
            )

    def queue_command(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, object] | None = None,
        idempotency_key: str = "",
    ) -> RuntimeCommand:
        with self._lock:
            run = self.store.get_run(run_id)
            if idempotency_key:
                existing = next(
                    (
                        command
                        for command in self.store.list_commands(run_id)
                        if command.epoch == run.epoch and command.idempotency_key == idempotency_key
                    ),
                    None,
                )
                if existing is not None:
                    return existing
            return self.store.save_command(RuntimeCommand(
                id=f"command-{uuid.uuid4().hex}",
                run_id=run.id,
                epoch=run.epoch,
                kind=kind,
                payload=payload or {},
                idempotency_key=idempotency_key,
            ))

    def issue_connector_grant(
        self,
        run_id: str,
        provider: str,
        connection_name: str,
        actions: list[str],
        resource_constraints: dict[str, str] | None = None,
        ttl: timedelta = timedelta(minutes=15),
    ) -> tuple[ConnectorGrant, str]:
        return self.connector_delegation.issue(
            run_id,
            provider,
            connection_name,
            actions,
            resource_constraints,
            ttl,
        )

    def authorize_connector(
        self,
        grant_id: str,
        token: str,
        action: str,
        constraints: dict[str, str] | None = None,
        computer_id: str | None = None,
    ) -> ConnectorGrant:
        return self.connector_delegation.authorize(
            grant_id,
            token,
            action,
            constraints,
            computer_id,
        )

    def revoke_connector_grants(self, run_id: str) -> None:
        self.connector_delegation.revoke_for_run(run_id)

    def claim_commands(self, run_id: str, computer_id: str, lease_id: str, epoch: int) -> list[RuntimeCommand]:
        claimed: list[RuntimeCommand] = []

        def claim(run: TaskRun) -> None:
            self._require_fence(run, computer_id, lease_id, epoch)
            now = utc_now()
            for command in self.store.list_commands(run_id):
                if command.acked_at is not None or command.epoch != run.epoch:
                    continue
                if command.claim_expires_at is not None and command.claim_expires_at > now:
                    continue
                command.claimed_at = now
                command.claim_expires_at = now + _COMMAND_CLAIM_DURATION
                command.delivery_count += 1
                claimed.append(self.store.save_command(command))

        with self._lock:
            self.store.update_run(run_id, claim)
            return claimed

    def acknowledge_command(
        self,
        run_id: str,
        command_id: str,
        computer_id: str,
        lease_id: str,
        epoch: int,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> tuple[RuntimeCommand, bool]:
        """Record the acknowledgement; the flag is True only for the first ack of a command."""

        with self._lock:
            run = self.store.get_run(run_id)
            self._require_fence(run, computer_id, lease_id, epoch)
            try:
                command = self.store.get_command(command_id)
            except KeyError:
                raise KeyError("Runtime command not found")
            if command.run_id != run_id:
                raise KeyError("Runtime command not found")
            if command.epoch != run.epoch:
                raise StaleRuntimeEvent("Runtime command belongs to a stale epoch")
            first_ack = command.acked_at is None
            if first_ack:
                command.acked_at = utc_now()
                command.claim_expires_at = None
                command.result = result
                command.error = error
                self.store.save_command(command)
            return command, first_ack

    def wait_for_command(self, run_id: str, command_id: str, timeout: float = 20.0) -> RuntimeCommand:
        """Wait for one fenced runtime reply without hiding persistence behind memory."""

        deadline = time.monotonic() + timeout
        while True:
            run = self.store.get_run(run_id)
            try:
                command = self.store.get_command(command_id)
            except KeyError:
                raise KeyError("Runtime command not found")
            if command.run_id != run_id:
                raise KeyError("Runtime command not found")
            if command.epoch != run.epoch:
                raise StaleRuntimeEvent("Runtime command belongs to a superseded execution")
            if command.acked_at is not None:
                return command
            if time.monotonic() >= deadline:
                raise RuntimeError("The selected computer did not answer in time")
            time.sleep(0.05)

    def delete_task(self, run_id: str) -> None:
        """Remove canonical control records after compatibility cleanup succeeds."""

        run = self.store.get_run(run_id)
        self.store.delete_task(run.task_id)

    def recovery_plan(self, run_id: str) -> RecoveryPlan:
        run = self.store.get_run(run_id)
        task = self.store.get_task(run.task_id)
        project = task.execution_project
        eligible = (
            self.eligible_computers(
                project,
                TaskResourceScope(all_project_resources=True),
                task.engine_id,
            )
            if project is not None
            else []
        )
        return build_recovery_plan(self.store, run, eligible)

    def recover_run(
        self,
        run_id: str,
        computer_id: str | None = None,
        *,
        allow_recreate: bool = False,
    ) -> TaskRun:
        with self._lock:
            run = self.store.get_run(run_id)
            computer = self.store.get_computer(computer_id or run.computer_id)
            return apply_recovery(
                self.store,
                run.id,
                computer,
                allow_recreate=allow_recreate,
            )

    def reassign_queued_run(self, run_id: str, computer_id: str) -> TaskRun:
        """Move a run that no computer has leased yet; leased runs go through recovery."""

        with self._lock:
            run = self.store.get_run(run_id)
            if run.status != RunStatus.queued:
                raise StateConflict("Only queued Task Runs can be reassigned; recover a leased run instead")
            task = self.store.get_task(run.task_id)
            if task.execution_project is None:
                raise NoEligibleComputer("This task includes resources that only exist on its original computer")
            eligible = self.eligible_computers(
                task.execution_project,
                TaskResourceScope(all_project_resources=True),
                task.engine_id,
            )
            if not any(item.id == computer_id for item in eligible):
                raise NoEligibleComputer("That computer cannot access every resource selected for this task")
            previous_computer_id = run.computer_id

            def reassign(current: TaskRun) -> None:
                if current.status != RunStatus.queued:
                    raise StateConflict("Only queued Task Runs can be reassigned; recover a leased run instead")
                current.computer_id = computer_id

            saved = self.store.update_run(run.id, reassign)
            for workspace in self.store.list_workspaces(run.id):
                workspace.computer_id = computer_id
                self.store.save_workspace(workspace)
            record_security_event(self.store,
                "run.reassign",
                "completed",
                "user",
                run.id,
                run_id=run.id,
                computer_id=computer_id,
                detail=f"from={previous_computer_id}",
            )
            return saved

    def expire_stale_computers(self) -> None:
        threshold = utc_now() - _OFFLINE_AFTER
        for computer in self.store.list_computers():
            if computer.id == self.local_computer.id or computer.status == ComputerStatus.draining:
                continue
            if computer.last_seen_at < threshold and computer.status != ComputerStatus.offline:
                computer.status = ComputerStatus.offline
                self.store.save_computer(computer)

    def expire_leases(self) -> None:
        with self._lock:
            now = utc_now()
            for candidate in self.store.list_runs():
                if candidate.lease_expires_at is None or candidate.lease_expires_at >= now:
                    continue
                workspaces = self.store.list_workspaces(candidate.id)
                resume_mode = "restore" if workspaces and all(
                    item.status == WorkspaceStatus.ready and item.path for item in workspaces
                ) else "prepare"
                self.store.update_run(
                    candidate.id,
                    lambda run, mode=resume_mode: self._expire_lease(run, now, mode),
                )

    @staticmethod
    def _expire_lease(run: TaskRun, now: datetime, workspace_resume_mode: Literal["prepare", "restore"]) -> None:
        if run.lease_expires_at is None or run.lease_expires_at >= now:
            return
        if run.status not in {RunStatus.preparing, RunStatus.ready, RunStatus.running, RunStatus.awaiting_approval}:
            return
        run.lease_id = None
        run.lease_expires_at = None
        run.epoch += 1
        run.last_event_seq = 0
        run.last_event_id = None
        run.checkpoint = {}
        run.workspace_resume_mode = workspace_resume_mode
        run.recovery_count += 1
        transition_run(
            run,
            RunStatus.recovering,
            error="The computer stopped responding. The task can be resumed safely.",
        )

    def _register_local(self, capabilities: ComputerCapabilities) -> Computer:
        identifier = self._local_computer_id()
        name = self._read_local_computer_name() or self._computer_name(platform.node())
        try:
            existing = self.store.get_computer(identifier)
            existing.name = name
            existing.is_local = True
            existing.capabilities = capabilities
            existing.status = ComputerStatus.online
            existing.last_seen_at = utc_now()
            existing.revoked_at = None
            return self.store.save_computer(existing)
        except KeyError:
            return self.store.save_computer(Computer(
                id=identifier,
                name=name,
                is_local=True,
                capabilities=capabilities,
            ))

    def _read_local_computer_name(self) -> str | None:
        try:
            value = (self.root / "control" / "local-computer-name").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return self._computer_name(value) if value else None

    def _write_local_computer_name(self, name: str) -> None:
        path = self.root / "control" / "local-computer-name"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            temp.write_text(self._computer_name(name) + "\n", encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def _local_computer_id(self) -> str:
        path = self.root / "control" / "local-computer-id"
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except FileNotFoundError:
            pass
        value = f"computer-{uuid.uuid4().hex}"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(value + "\n", encoding="utf-8")
        os.replace(temp, path)
        return value

    @staticmethod
    def _computer_name(value: str) -> str:
        normalized = " ".join(value.replace(".local", "").replace("-", " ").split())
        return (normalized or "This computer")[:120]

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _require_protocol(protocol_versions: list[str]) -> None:
        if RUNTIME_PROTOCOL_VERSION not in protocol_versions:
            raise ValueError(f"Runtime protocol {RUNTIME_PROTOCOL_VERSION} is required")

    def _active_count(self, computer_id: str) -> int:
        capacity_statuses = {
            RunStatus.preparing,
            RunStatus.running,
            RunStatus.awaiting_approval,
            RunStatus.recovering,
        }
        return sum(
            run.computer_id == computer_id and run.status in capacity_statuses
            for run in self.store.list_runs()
        )

    @staticmethod
    def _resource_eligible(resource: ProjectResource, computer: Computer) -> bool:
        if isinstance(resource, LocalFolderResource):
            return computer.capabilities.supports_local_folders and resource.computer_id == computer.id
        if not computer.capabilities.has_git:
            return False
        return bool(resource.source_url) or resource.computer_id == computer.id

    @staticmethod
    def scoped_resources(project: CodeProject | None, scope: TaskResourceScope) -> list[ProjectResource]:
        if project is None:
            return []
        if scope.all_project_resources:
            return list(project.resources)
        by_id = {resource.id: resource for resource in project.resources}
        missing = [resource_id for resource_id in scope.resource_ids if resource_id not in by_id]
        if missing:
            raise ValueError(f"Unknown project resources: {', '.join(missing)}")
        return [by_id[resource_id] for resource_id in scope.resource_ids]

    @staticmethod
    def execution_project_snapshot(
        project: CodeProject | None,
        resources: list[ProjectResource],
    ) -> CodeProject | None:
        if project is None:
            return None
        return CodeProject.model_validate({
            **project.model_dump(mode="python"),
            "resources": [resource.model_copy(deep=True) for resource in resources],
            "folders": [],
        })

    @staticmethod
    def _require_fence(run: TaskRun, computer_id: str, lease_id: str, epoch: int) -> None:
        ControlPlaneService._require_event_owner(run, computer_id, epoch)
        if not run.lease_id:
            raise StaleRuntimeEvent("Task Run ownership changed")
        if not hmac.compare_digest(run.lease_id, lease_id):
            raise StaleRuntimeEvent("Task Run lease is stale")
        if run.lease_expires_at is None or run.lease_expires_at < utc_now():
            raise StaleRuntimeEvent("Task Run lease expired")

    @staticmethod
    def _require_event_owner(run: TaskRun, computer_id: str, epoch: int) -> None:
        if run.computer_id != computer_id or run.epoch != epoch:
            raise StaleRuntimeEvent("Task Run ownership changed")

    @staticmethod
    def _fallback_workspace(session: CodingSession) -> TaskWorkspace:
        from cowork.coding.contracts import TaskWorkspace

        return TaskWorkspace(
            folder_id="folder",
            folder_name=Path(session.source_path).name or "Folder",
            source_path=session.source_path,
            workspace_path=session.workspace_path,
            workspace_kind=session.workspace_kind,
            repository_root=session.repository_root,
            base_revision=session.base_revision,
            source_dirty=session.source_dirty,
        )

    @staticmethod
    def _execution_workspace(run: TaskRun, workspace: TaskWorkspace) -> ExecutionWorkspace:
        return ExecutionWorkspace(
            id=f"workspace-{uuid.uuid4().hex}",
            run_id=run.id,
            resource_id=workspace.folder_id,
            computer_id=run.computer_id,
            status=WorkspaceStatus.ready,
            path=workspace.workspace_path,
            workspace_kind=workspace.workspace_kind,
            base_revision=workspace.base_revision,
            task_branch=workspace.task_branch,
        )
