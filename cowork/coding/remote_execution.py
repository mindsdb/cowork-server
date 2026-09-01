from __future__ import annotations

import uuid
from collections.abc import Callable

from cowork.coding.commands import CommandIntent
from cowork.coding.connector_capabilities import ConnectorCapability
from cowork.coding.contracts import (
    CodingEvent,
    CodingSession,
    EventType,
    InputReference,
    PendingApproval,
    SessionStatus,
    TaskWorkspace,
)
from cowork.coding.control_models import (
    RunStatus,
    RuntimeEvent,
    TaskResourceScope,
    TaskRun,
)
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.project_service import CodeProjectService
from cowork.coding.redaction import redact_text
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease
from cowork.coding.store import CodingStore


class RemoteExecutionCoordinator:
    """Project remote-runtime state into the existing desktop task read model."""

    def __init__(
        self,
        control: ControlPlaneService,
        store: CodingStore,
        projects: CodeProjectService,
        get_session: Callable[[str], CodingSession],
    ) -> None:
        self.control = control
        self.store = store
        self.projects = projects
        self.get_session = get_session

    def is_remote(self, session: CodingSession) -> bool:
        return bool(session.computer_id and session.computer_id != self.control.local_computer.id)

    def accept_event(self, event: RuntimeEvent) -> TaskRun:
        run = self.control.renew_lease(event)
        # Checkpoints are protocol heartbeats, not user-visible activity.  A
        # remote worker sends them while an already-completed task waits for a
        # follow-up command; projecting each one into the task timeline both
        # floods the UI and makes the completed task appear to become "ready"
        # again every polling interval.  The control plane has already stored
        # the checkpoint and renewed the fenced lease above, so no read-model
        # mutation is required here.
        if event.kind == "checkpoint":
            return run
        if event.kind == "workspace":
            return self._accept_workspace(run, event)
        if event.kind == "turn_completed":
            return self._accept_turn_completed(run, event)
        session = self.get_session(run.task_id)
        pending_approval, coding_event = self._coding_event(run, event)

        def update(current: CodingSession) -> None:
            current.status = _SESSION_STATUS[run.status]
            current.runtime_epoch = run.epoch
            current.last_error = run.last_error
            current.pending_approval = pending_approval
            if run.status == RunStatus.awaiting_approval:
                current.status = SessionStatus.awaiting_approval
            current.workspace_warning = (
                "The computer stopped responding. Resume this task when it is available."
                if run.status == RunStatus.recovering
                else None
            )

        self.store.append_event(session.id, coding_event, update)
        if run.status in {RunStatus.completed, RunStatus.cancelled, RunStatus.failed}:
            self.control.release_workspaces(run.id)
            self.control.revoke_connector_grants(run.id)
        return run

    def queue_turn(
        self,
        session: CodingSession,
        prompt: str,
        attachments: list[InputReference] | tuple[InputReference, ...],
        intent: CommandIntent,
    ) -> CodingSession:
        if not session.run_id:
            raise RuntimeError("Remote task is missing its Task Run")
        if attachments:
            raise RuntimeError("File attachments are not portable to another computer yet")
        command = self.control.queue_command(
            session.run_id,
            "start",
            intent.runtime_payload(prompt),
            f"turn-{uuid.uuid4().hex}",
        )
        self.store.append_event(
            session.id,
            CodingEvent(
                type=EventType.user_message,
                title="You",
                text=prompt,
                phase="pending",
                data={"commandId": command.id, "computerId": session.computer_id},
            ),
            lambda current: setattr(current, "workspace_warning", "Waiting for the selected computer"),
        )
        return self.get_session(session.id)

    def start_next_queued(self, session: CodingSession) -> CodingSession:
        if not session.queued_instructions:
            return self.get_session(session.id)
        instruction = session.queued_instructions[0]
        if instruction.attachments:
            raise RuntimeError("Queued file attachments cannot move to another computer yet")
        if not session.run_id:
            raise RuntimeError("Remote task is missing its Task Run")
        run = self.control.store.get_run(session.run_id)
        if run.status != RunStatus.ready:
            raise RuntimeError("Wait for the active turn to finish before resuming queued work")
        command = self.control.queue_command(
            session.run_id,
            "start",
            CommandIntent.parse(instruction.prompt).runtime_payload(instruction.prompt),
            f"queued-turn-{instruction.id}",
        )

        def update(current: CodingSession) -> None:
            current.queued_instructions = [
                item for item in current.queued_instructions if item.id != instruction.id
            ]
            current.workspace_warning = "Starting queued work"

        self.store.append_event(
            session.id,
            CodingEvent(
                type=EventType.session,
                title="Starting queued instruction",
                text=instruction.prompt,
                phase="pending",
                data={"queueId": instruction.id, "commandId": command.id},
            ),
            update,
        )
        return self.get_session(session.id)

    def recovery_plan(self, session_id: str):
        session = self.get_session(session_id)
        if not session.run_id:
            raise RuntimeError("This task does not have a recoverable run")
        return self.control.recovery_plan(session.run_id)

    def recover(
        self,
        session_id: str,
        computer_id: str | None = None,
        *,
        allow_recreate: bool = False,
    ) -> CodingSession:
        session = self.get_session(session_id)
        if not session.run_id or not session.task_id:
            raise RuntimeError("This task does not have a recoverable run")
        run = self.control.store.get_run(session.run_id)
        if run.status not in {RunStatus.interrupted, RunStatus.failed, RunStatus.recovering}:
            raise RuntimeError("This task does not need to be restored")
        task = self.control.store.get_task(session.task_id)
        project = task.execution_project
        if project is None and task.project_id:
            project = self.projects.get(task.project_id)
        eligible = self.control.eligible_computers(
            project,
            TaskResourceScope(all_project_resources=True) if task.execution_project else task.resource_scope,
            session.engine_id,
        )
        target = computer_id or run.computer_id
        if not any(item.id == target for item in eligible):
            if computer_id is None and any(item.id == run.computer_id for item in eligible):
                target = run.computer_id
            elif computer_id is None and len(eligible) == 1:
                target = eligible[0].id
            else:
                raise RuntimeError("No online computer can restore this task with its selected resources")
        recovered = self.control.recover_run(run.id, target, allow_recreate=allow_recreate)
        session.computer_id = recovered.computer_id
        session.runtime_epoch = recovered.epoch
        session.status = SessionStatus.ready
        session.pending_approval = None
        session.last_error = None
        self.store.save_session(session)
        return self.get_session(session_id)

    def operation(
        self,
        session: CodingSession,
        operation: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float = 20.0,
    ) -> dict[str, object]:
        if not session.run_id:
            raise RuntimeError("Remote task is missing its Task Run")
        run = self.control.store.get_run(session.run_id)
        if run.status in {RunStatus.completed, RunStatus.cancelled, RunStatus.failed}:
            raise RuntimeError("The task execution workspace is no longer available")
        command = self.control.queue_command(
            session.run_id,
            "operation",
            {"operation": operation, **(payload or {})},
            f"operation-{operation}-{uuid.uuid4().hex}",
        )
        completed = self.control.wait_for_command(session.run_id, command.id, timeout)
        if completed.error:
            raise RuntimeError(redact_text(completed.error)[:4_000])
        return completed.result or {}

    def release_workspace(self, session: CodingSession) -> None:
        if not session.run_id:
            return
        run = self.control.store.get_run(session.run_id)
        if run.status in {RunStatus.completed, RunStatus.cancelled, RunStatus.failed}:
            return
        command = self.control.queue_command(
            run.id,
            "release",
            {},
            f"release-{run.epoch}",
        )
        try:
            self.control.wait_for_command(run.id, command.id, timeout=5)
        except RuntimeError as exc:
            self.control.set_run_status(run.id, RunStatus.interrupted, error="Workspace release was not acknowledged")
            raise RuntimeError(
                "The selected computer did not release this task workspace; retry when it is online"
            ) from exc

    @staticmethod
    def require_idle(session: CodingSession, action: str) -> None:
        if session.run_status in {"running", "awaiting_approval", "preparing"}:
            raise RuntimeError(f"Wait for the remote agent before {action}")

    def connector_capabilities(self, session: CodingSession) -> list[ConnectorCapability]:
        """Issue exact, short-lived connector authority for one leased run."""

        if not session.run_id or not session.task_id:
            return []
        task = self.control.store.get_task(session.task_id)
        project = task.execution_project
        if project is None:
            return []
        self.control.revoke_connector_grants(session.run_id)
        capabilities: list[ConnectorCapability] = []
        issued: set[tuple[str, str, str, tuple[str, ...]]] = set()

        def connection_name(provider: str, requested: str | None) -> str | None:
            if requested:
                return requested
            matches = [item.name for item in project.connections if item.provider == provider]
            return matches[0] if len(matches) == 1 else None

        def issue(
            provider: str,
            requested_connection: str | None,
            url: str,
            actions: list[str],
        ) -> None:
            selected = connection_name(provider, requested_connection)
            if provider not in {"github", "linear"} or not selected or not url:
                return
            normalized_actions = tuple(dict.fromkeys(actions))
            key = (provider, selected, url, normalized_actions)
            if key in issued:
                return
            issued.add(key)
            constraints = {"url": url}
            if "pull_request_status" in normalized_actions:
                constraints["target_url"] = url
            grant, token = self.control.issue_connector_grant(
                session.run_id,
                provider,
                selected,
                list(normalized_actions),
                constraints,
            )
            capabilities.append(ConnectorCapability(
                id=grant.id,
                provider=provider,
                token=token,
                actions=list(normalized_actions),
                resource_constraints=grant.resource_constraints,
                expires_at=grant.expires_at,
            ))

        for source in session.source_contexts:
            actions = ["read_source"]
            if source.provider == "github" and source.kind == "pull_request":
                actions.append("pull_request_status")
            issue(source.provider, source.connection_name, source.url, actions)
        for delivery in session.deliveries:
            if delivery.provider == "github" and delivery.external_url:
                issue("github", delivery.connection_name, delivery.external_url, ["pull_request_status"])
        return capabilities

    def acquire_lease(self, computer_id: str) -> RuntimeLease | None:
        """Lease one portable task and freeze its complete execution contract.

        Keeping this assembly below the HTTP layer makes the same lifecycle
        usable by the API, in-process verification, and future transports.
        """
        acquired = self.control.acquire_lease(computer_id)
        if acquired is None:
            return None
        run, lease_id = acquired
        task = self.control.store.get_task(run.task_id)
        runtime_project = self.control.runtime_project_for_task(task, computer_id)
        # One-time compatibility for tasks created before immutable execution
        # snapshots. Once leased, recovery no longer depends on live Project
        # metadata.
        if runtime_project is None and task.project_id:
            project = self.projects.get(task.project_id)
            scoped_resources = self.control.scoped_resources(project, task.resource_scope)
            runtime_project = self.control.runtime_project(project, task.resource_scope, computer_id)
            task.execution_project = self.control.execution_project_snapshot(project, scoped_resources)
            self.control.store.save_task(task)
        session = self.store.load_session(task.id)
        return RuntimeLease(
            task=task,
            run=run,
            lease_id=lease_id,
            agent_token=self.control.issue_run_token(run.id),
            project=runtime_project,
            execution=RuntimeExecutionConfig(
                engine_id=session.engine_id,
                model=session.model,
                permission_mode=session.permission_mode,
                reasoning_effort=session.reasoning_effort,
                service_tier=session.service_tier,
                personality=session.personality,
                network_access=session.network_access,
                web_search=session.web_search,
                developer_instructions=session.developer_instructions,
                environment=session.environment,
            ),
            connector_capabilities=self.connector_capabilities(session),
            workspaces=self.control.store.list_workspaces(run.id),
        )

    def _accept_workspace(self, run: TaskRun, event: RuntimeEvent) -> TaskRun:
        raw_items = event.payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("Runtime workspace publication is empty")
        workspaces = [TaskWorkspace.model_validate(item) for item in raw_items]
        expected = {item.resource_id for item in self.control.store.list_workspaces(run.id)}
        received = {item.folder_id for item in workspaces}
        if received != expected:
            raise ValueError("Runtime workspace publication does not match the task resource scope")
        raw_ports = event.payload.get("ports")
        ports = {
            str(name): int(value)
            for name, value in raw_ports.items()
            if isinstance(name, str) and isinstance(value, int)
        } if isinstance(raw_ports, dict) else {}
        self.control.attach_prepared_workspaces(run.id, workspaces)
        primary = workspaces[0]

        def update(current: CodingSession) -> None:
            current.source_path = primary.source_path
            current.workspace_path = primary.workspace_path
            current.workspace_kind = primary.workspace_kind
            current.workspaces = workspaces
            current.repository_root = primary.repository_root
            current.base_revision = primary.base_revision
            current.source_dirty = primary.source_dirty
            current.allocated_ports = ports
            current.workspace_warning = None

        self.store.append_event(
            run.task_id,
            CodingEvent(
                type=EventType.session,
                title="Workspace prepared",
                text=f"{len(workspaces)} resource{'s' if len(workspaces) != 1 else ''} ready",
                phase="completed",
                data={"computerId": run.computer_id, "runId": run.id},
            ),
            update,
        )
        return run

    def _accept_turn_completed(self, run: TaskRun, event: RuntimeEvent) -> TaskRun:
        outcome = str(event.payload.get("status") or "completed")
        if outcome not in {"completed", "cancelled"}:
            raise ValueError("Runtime turn completion status is invalid")
        run = self.control.set_run_status(run.id, RunStatus.ready)

        def update(current: CodingSession) -> None:
            current.status = SessionStatus.cancelled if outcome == "cancelled" else SessionStatus.completed
            current.active_turn_id = None
            current.pending_approval = None
            current.workspace_warning = None
            current.last_error = None

        self.store.append_event(
            run.task_id,
            CodingEvent(
                type=EventType.session,
                title="Stopped" if outcome == "cancelled" else "Completed",
                text="The task workspace remains available for review and follow-up.",
                phase="completed",
                data={"computerId": run.computer_id, "runId": run.id},
            ),
            update,
        )
        latest = self.store.load_session(run.task_id)
        if outcome == "completed" and latest.queued_instructions:
            self.start_next_queued(latest)
        return run

    @staticmethod
    def _coding_event(run: TaskRun, event: RuntimeEvent) -> tuple[PendingApproval | None, CodingEvent]:
        pending: PendingApproval | None = None
        if event.kind == "approval":
            parameters = event.payload.get("params")
            data = parameters if isinstance(parameters, dict) else {}
            pending = PendingApproval(
                id=str(event.payload.get("approvalId") or ""),
                method=str(event.payload.get("method") or "runtime"),
                kind=str(data.get("kind") or "command"),
                title=redact_text(str(data.get("title") or "Approve agent action"))[:512],
                detail=redact_text(str(
                    data.get("reason") or data.get("command") or "The agent needs your approval."
                ))[:8_192],
                cwd=redact_text(str(data.get("cwd")))[:32_768] if data.get("cwd") else None,
                risk=str(data.get("risk") or "review"),
                scope=str(data.get("scope") or "once"),
                allow_session=bool(data.get("allowSession")),
            )
            return pending, CodingEvent(
                type=EventType.approval,
                title=pending.title,
                text=pending.detail,
                phase="pending",
                data={
                    "approvalId": pending.id,
                    "kind": pending.kind,
                    "cwd": pending.cwd,
                    "risk": pending.risk,
                    "scope": pending.scope,
                    "allowSession": pending.allow_session,
                },
            )
        if event.kind == "event":
            raw = event.payload.get("event")
            return None, CodingEvent.model_validate(raw) if isinstance(raw, dict) else CodingEvent(
                type=EventType.session,
                title="Remote task update",
                text=str(event.payload.get("detail") or "The task reported progress."),
                phase="pending",
            )
        title = _STATUS_TITLES.get(run.status, "Task updated")
        return None, CodingEvent(
            type=EventType.error if run.status == RunStatus.failed else EventType.session,
            title=title,
            text=redact_text(str(event.payload.get("detail") or run.last_error or "")),
            phase=(
                "failed" if run.status in {RunStatus.failed, RunStatus.interrupted}
                else "completed" if run.status in {RunStatus.completed, RunStatus.cancelled}
                else "pending"
            ),
            data={"computerId": run.computer_id, "runId": run.id, "runStatus": run.status.value},
        )


_SESSION_STATUS = {
    RunStatus.queued: SessionStatus.ready,
    RunStatus.preparing: SessionStatus.ready,
    RunStatus.ready: SessionStatus.ready,
    RunStatus.running: SessionStatus.running,
    RunStatus.awaiting_approval: SessionStatus.awaiting_approval,
    RunStatus.completed: SessionStatus.completed,
    RunStatus.cancelled: SessionStatus.cancelled,
    RunStatus.interrupted: SessionStatus.interrupted,
    RunStatus.failed: SessionStatus.failed,
    RunStatus.recovering: SessionStatus.interrupted,
}

_STATUS_TITLES = {
    RunStatus.preparing: "Preparing task",
    RunStatus.ready: "Workspace ready",
    RunStatus.running: "Agent started",
    RunStatus.awaiting_approval: "Your approval is needed",
    RunStatus.completed: "Completed",
    RunStatus.cancelled: "Cancelled",
    RunStatus.interrupted: "Computer disconnected",
    RunStatus.failed: "Task failed",
    RunStatus.recovering: "Ready to resume",
}
