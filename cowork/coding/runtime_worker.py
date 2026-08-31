from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from cowork.coding.context import goal_status_text
from cowork.coding.contracts import CodingEvent, EventType
from cowork.coding.control_models import RuntimeCommand
from cowork.coding.engines.base import (
    EngineCredentials,
    EngineMcpServer,
    EngineSession,
    EngineSessionConfig,
)
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.remote_integration_mcp import (
    RemoteIntegrationConfig,
    write_remote_integration_config,
)
from cowork.coding.runtime_operations import RuntimeWorkspaceOperations
from cowork.coding.runtime_client import (
    RemoteRuntimeClient,
    RuntimeClientError,
    RuntimeIdentity,
    load_runtime_identity,
    run_runtime_forever,
    save_runtime_identity,
)
from cowork.coding.runtime_protocol import RuntimeLease
from cowork.coding.workspace import WorkspaceManager


class CodeOnlyRuntime:
    """Prepare isolated workspaces and run an agent without a desktop UI."""

    HEARTBEAT_INTERVAL_SECONDS = 10.0

    def __init__(
        self,
        root: Path,
        client: RemoteRuntimeClient,
        registry: CodingEngineRegistry = engine_registry,
        heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.root = root
        self.client = client
        self.registry = registry
        self.workspaces = ProjectWorkspaceManager(WorkspaceManager(root / "coding"))
        self._approval_lock = threading.Lock()
        self._approval_waiters: dict[str, tuple[threading.Event, dict[str, str]]] = {}
        self._pending_commands: list[RuntimeCommand] = []
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    def run_once(self, wait_seconds: float = 0) -> bool:
        self.client.heartbeat()
        lease = self.client.lease(wait_seconds)
        if lease is None:
            return False
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_while_active,
            args=(heartbeat_stop,),
            name=f"runtime-heartbeat-{lease.run.id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self._execute(lease)
        except BaseException as exc:
            try:
                self.client.event(lease, "error", {"detail": str(exc)[:4_000]})
            except RuntimeClientError:
                pass
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
        return True

    def _heartbeat_while_active(self, stop: threading.Event) -> None:
        """Keep the computer online while workspace or agent work blocks the lease loop."""

        while not stop.is_set():
            try:
                self.client.heartbeat(active_run_count=1)
            except httpx.TransportError:
                # The event/command paths surface control-plane outages to the
                # main run. A transient heartbeat failure must not kill useful
                # local work or produce an unhandled daemon-thread exception.
                pass
            except RuntimeClientError as exc:
                if exc.status_code not in {429, 500, 502, 503, 504}:
                    return
            if stop.wait(self._heartbeat_interval_seconds):
                return

    def _execute(self, lease: RuntimeLease) -> None:
        if lease.project is None:
            raise RuntimeClientError("A remote Task Run requires portable project resources")
        prepared = (
            self.workspaces.restore(lease.task.id, lease.project, lease.workspaces)
            if lease.run.workspace_resume_mode == "restore"
            else self.workspaces.prepare(lease.task.id, lease.project)
        )
        self.client.event(lease, "workspace", {
            "items": [item.model_dump(mode="json") for item in prepared.workspaces],
            "ports": prepared.ports,
        })
        self.client.event(lease, "status", {
            "status": "ready",
            "detail": f"Prepared {len(prepared.workspaces)} project resource(s)",
        })
        engine = self.registry.get(lease.execution.engine_id)
        connector_config_path = write_remote_integration_config(
            self.root / "run-secrets" / f"{lease.run.id}.json",
            RemoteIntegrationConfig(
                server_url=self.client.server_url,
                computer_id=self.client.identity.computer_id,
                run_id=lease.run.id,
                agent_token=lease.agent_token,
                project_context={
                    "project": lease.project.name,
                    "resources": [
                        {"id": item.id, "name": item.name, "kind": item.kind}
                        for item in lease.project.resources
                    ],
                    "linkedSources": [
                        {
                            "provider": item.provider,
                            "kind": item.kind,
                            "url": item.url,
                            "title": item.title,
                        }
                        for item in lease.task.source_contexts
                    ],
                    "connectedTools": [
                        {"provider": item.provider, "label": item.label or item.name}
                        for item in lease.project.connections
                    ],
                },
                capabilities=lease.connector_capabilities,
            ),
        )
        session = engine.open_session(
            cowork_root=str(self.root / "coding"),
            workspace=prepared.primary.workspace_path,
            config=EngineSessionConfig(
                model=lease.execution.model,
                permission_mode=lease.execution.permission_mode,
                reasoning_effort=lease.execution.reasoning_effort,
                service_tier=lease.execution.service_tier,
                personality=lease.execution.personality,
                network_access=lease.execution.network_access,
                web_search=lease.execution.web_search,
                additional_dirs=tuple(item.workspace_path for item in prepared.workspaces[1:]),
                developer_instructions=lease.execution.developer_instructions,
                session_id=lease.task.id,
                cowork_root=str(self.root / "coding"),
                environment=tuple({
                    **lease.execution.environment,
                    **{name: str(value) for name, value in prepared.ports.items()},
                }.items()),
                workspace_label=lease.project.name,
                inference_base_url=self.client.inference_endpoint(lease),
                inference_api_key=lease.agent_token,
                mcp_servers=(EngineMcpServer(
                    name="mindshub_code",
                    command=sys.executable,
                    args=("-m", "cowork.coding.remote_integration_mcp", str(connector_config_path)),
                ),),
            ),
            credentials=EngineCredentials(
                minds_url=self.client.inference_endpoint(lease),
                minds_api_key=lease.agent_token,
            ),
            existing_session_id=None,
            approval_handler=lambda method, params: self._approval(lease, method, params),
        )
        operations = RuntimeWorkspaceOperations(lease, self.workspaces, prepared, session)
        try:
            while True:
                start = self._wait_for_start(lease, operations)
                if start.kind == "release":
                    operations.release()
                    self.client.acknowledge(lease, start, {})
                    # A terminal status clears the fenced lease.  Acknowledge
                    # the release command while ownership is still valid, then
                    # close the run; doing this in the opposite order makes the
                    # worker reject its own acknowledgement and exit.
                    self.client.event(lease, "status", {"status": "completed", "detail": "Workspace released"})
                    return
                self._run_turn(lease, session, operations, start)
        finally:
            session.close()
            connector_config_path.unlink(missing_ok=True)

    def _wait_for_start(
        self,
        lease: RuntimeLease,
        operations: RuntimeWorkspaceOperations,
    ) -> RuntimeCommand:
        while True:
            selected: RuntimeCommand | None = None
            for command in self._take_commands(lease):
                if command.kind in {"start", "release"}:
                    if selected is None:
                        selected = command
                    else:
                        self._pending_commands.append(command)
                elif command.kind == "operation" and selected is None:
                    self._complete_operation(lease, operations, command)
                else:
                    self._pending_commands.append(command)
            if selected is not None:
                return selected
            self.client.event(lease, "checkpoint", {"waiting": "start", "workspaceReady": True})
            time.sleep(2)

    def _run_turn(
        self,
        lease: RuntimeLease,
        session: EngineSession,
        operations: RuntimeWorkspaceOperations,
        start: RuntimeCommand,
    ) -> None:
        self.client.event(lease, "status", {"status": "running"})
        turn_id = self._start_turn(lease, session, start)
        self.client.acknowledge(lease, start)
        if turn_id is None:
            self.client.event(lease, "turn_completed", {"status": "completed"})
            return
        stop = threading.Event()
        cancelled = threading.Event()
        command_thread = threading.Thread(
            target=self._route_commands,
            args=(lease, session, operations, turn_id, stop, cancelled),
            name=f"runtime-commands-{lease.run.id[:8]}",
            daemon=True,
        )
        command_thread.start()
        try:
            for event in session.events(turn_id):
                self.client.event(lease, "event", {"event": event.model_dump(mode="json")})
        finally:
            stop.set()
            command_thread.join(timeout=2)
        self.client.event(lease, "turn_completed", {
            "status": "cancelled" if cancelled.is_set() else "completed",
        })

    def _start_turn(
        self,
        lease: RuntimeLease,
        session: EngineSession,
        start: RuntimeCommand,
    ) -> str | None:
        command = str(start.payload.get("command") or "")
        goal_action = str(start.payload.get("goal_action") or "")
        objective = start.payload.get("goal_objective")
        goal_objective = str(objective) if objective is not None else None
        if self._run_immediate_command(lease, session, command, goal_action, goal_objective):
            return None
        if command == "goal":
            if goal_action == "set":
                return session.start_goal(goal_objective or "")
            if goal_action == "resume":
                return session.resume_goal()
            goal = session.goal_status() if goal_action == "view" else session.update_goal(goal_action, goal_objective)
            labels = {
                "view": "Goal status",
                "edit": "Goal updated",
                "pause": "Goal paused",
                "clear": "Goal cleared",
            }
            text = goal_status_text(goal) if goal else (
                "No goal is active for this coding task."
                if goal_action == "view"
                else "The task goal has been cleared."
            )
            self._emit_session_event(lease, labels[goal_action], text, {"goal": goal or {}})
            return None
        if command == "review":
            return session.start_review()
        return session.start_turn(str(
            start.payload.get("engine_prompt")
            or start.payload.get("prompt")
            or lease.task.prompt
        ))

    def _run_immediate_command(
        self,
        lease: RuntimeLease,
        session: EngineSession,
        command: str,
        goal_action: str,
        goal_objective: str | None,
    ) -> bool:
        if command == "compact":
            session.compact()
            self._emit_session_event(
                lease,
                "Compaction started",
                "Codex compacted this task's context for future turns.",
            )
            return True
        if command == "status":
            goal = session.goal_status()
            goal_line = goal_status_text(goal) if goal else "Goal: none"
            self._emit_session_event(
                lease,
                "Task status",
                "\n".join((
                    f"Model: {lease.execution.model}",
                    f"Permissions: {lease.execution.permission_mode.value}",
                    f"Network: {'on' if lease.execution.network_access else 'off'}",
                    goal_line,
                )),
                {"goal": goal or {}},
            )
            return True
        if command == "goal" and goal_action in {"view", "edit", "pause", "clear"}:
            goal = session.goal_status() if goal_action == "view" else session.update_goal(
                goal_action,
                goal_objective,
            )
            labels = {
                "view": "Goal status",
                "edit": "Goal updated",
                "pause": "Goal paused",
                "clear": "Goal cleared",
            }
            text = goal_status_text(goal) if goal else (
                "No goal is active for this coding task."
                if goal_action == "view"
                else "The task goal has been cleared."
            )
            self._emit_session_event(lease, labels[goal_action], text, {"goal": goal or {}})
            return True
        return False

    def _emit_session_event(
        self,
        lease: RuntimeLease,
        title: str,
        text: str,
        data: dict[str, object] | None = None,
    ) -> None:
        event = CodingEvent(
            type=EventType.session,
            title=title,
            text=text,
            phase="completed",
            data=data or {},
        )
        self.client.event(lease, "event", {"event": event.model_dump(mode="json")})

    def _route_commands(
        self,
        lease: RuntimeLease,
        session: EngineSession,
        operations: RuntimeWorkspaceOperations,
        turn_id: str,
        stop: threading.Event,
        cancelled: threading.Event,
    ) -> None:
        while not stop.wait(1):
            try:
                for command in self._take_commands(lease):
                    if command.kind == "steer":
                        session.steer(turn_id, str(command.payload.get("prompt") or ""))
                    elif command.kind == "agent_command":
                        action = str(command.payload.get("goal_action") or "")
                        objective = command.payload.get("goal_objective")
                        handled = self._run_immediate_command(
                            lease,
                            session,
                            str(command.payload.get("command") or ""),
                            action,
                            str(objective) if objective is not None else None,
                        )
                        if not handled:
                            self.client.acknowledge(
                                lease,
                                command,
                                error="The command cannot run during an active turn",
                            )
                            continue
                    elif command.kind == "cancel":
                        session.cancel(turn_id)
                        cancelled.set()
                    elif command.kind == "approve":
                        approval_id = str(command.payload.get("approvalId") or "")
                        with self._approval_lock:
                            waiter = self._approval_waiters.get(approval_id)
                        if waiter is None:
                            continue
                        waiter[1]["decision"] = str(command.payload.get("decision") or "decline")
                        waiter[0].set()
                    elif command.kind == "operation":
                        self._complete_operation(lease, operations, command)
                        continue
                    else:
                        continue
                    self.client.acknowledge(lease, command)
                self.client.event(lease, "checkpoint", {"activeTurn": turn_id})
            except RuntimeClientError:
                return

    def _take_commands(self, lease: RuntimeLease) -> list[RuntimeCommand]:
        pending, self._pending_commands = self._pending_commands, []
        return [*pending, *self.client.commands(lease)]

    def _complete_operation(
        self,
        lease: RuntimeLease,
        operations: RuntimeWorkspaceOperations,
        command: RuntimeCommand,
    ) -> None:
        result, error = operations.execute(command)
        self.client.acknowledge(lease, command, result, error)

    def _approval(self, lease: RuntimeLease, method: str, params: dict[str, Any] | None) -> dict[str, str]:
        approval_id = f"approval-{lease.run.id}-{int(time.time() * 1_000)}"
        resolved = threading.Event()
        decision: dict[str, str] = {}
        with self._approval_lock:
            self._approval_waiters[approval_id] = (resolved, decision)
        self.client.event(lease, "status", {"status": "awaiting_approval"})
        self.client.event(lease, "approval", {
            "approvalId": approval_id,
            "method": method,
            "params": params or {},
        })
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if resolved.wait(timeout=5):
                    self.client.event(lease, "status", {"status": "running"})
                    return {"decision": decision.get("decision", "decline")}
                self.client.event(lease, "checkpoint", {"waitingForApproval": approval_id})
            return {"decision": "decline"}
        finally:
            with self._approval_lock:
                self._approval_waiters.pop(approval_id, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect this computer to MindsHub Code")
    parser.add_argument("--server", default=os.environ.get("COWORK_RUNTIME_SERVER", ""))
    parser.add_argument("--code", default=os.environ.get("COWORK_RUNTIME_REGISTRATION_TOKEN", ""))
    parser.add_argument("--name", default=os.environ.get("COWORK_RUNTIME_NAME", platform.node() or "Code runtime"))
    parser.add_argument("--root", default=os.environ.get("COWORK_RUNTIME_ROOT", "~/.mindshub-code"))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    stored = load_runtime_identity(root)
    server_url = str(args.server).strip()
    if not server_url and stored:
        server_url = stored.server_url
    registration_token = str(args.code).strip()
    if registration_token:
        if not server_url:
            raise SystemExit("--server is required when connecting a computer")
        client = RemoteRuntimeClient.register(server_url, registration_token, str(args.name).strip(), root)
        save_runtime_identity(root, server_url, client.identity)
    elif stored:
        client = RemoteRuntimeClient(server_url or stored.server_url, stored.identity)
    else:
        raise SystemExit("Connect this computer from Settings → Code → Computers first")
    runtime = CodeOnlyRuntime(root, client)
    try:
        run_runtime_forever(runtime)
    finally:
        client.close()


if __name__ == "__main__":
    main()
