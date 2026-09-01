from __future__ import annotations

import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cowork.coding.contracts import CodingEvent, EventType
from cowork.coding.control_models import (
    RUNTIME_PROTOCOL_VERSION,
    ComputerCapabilities,
    RuntimeCommand,
    RuntimeEvent,
)
from cowork.coding.engines.base import (
    EngineCredentials,
    EngineMcpServer,
    EngineSession,
    EngineSessionConfig,
)
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.project_workspaces import PreparedProjectWorkspace, ProjectWorkspaceManager
from cowork.coding.remote_integration_mcp import (
    RemoteIntegrationConfig,
    write_remote_integration_config,
)
from cowork.coding.runtime_operations import RuntimeWorkspaceOperations
from cowork.coding.runtime_protocol import (
    ComputerRegistrationRequest,
    ComputerRegistrationResponse,
    RuntimeLease,
)
from cowork.coding.shells import shell_inventory
from cowork.coding.workspace import WorkspaceManager


class RuntimeClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeIdentity:
    computer_id: str
    runtime_token: str
    name: str


class RemoteRuntimeClient:
    """Authenticated outbound client for the versioned execution protocol."""

    def __init__(self, server_url: str, identity: RuntimeIdentity, client: httpx.Client | None = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.identity = identity
        self.client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._sequence_lock = threading.Lock()
        self._sequences: dict[str, int] = {}

    @classmethod
    def register(
        cls,
        server_url: str,
        registration_token: str,
        name: str,
        root: Path,
        registry: CodingEngineRegistry = engine_registry,
    ) -> RemoteRuntimeClient:
        persisted_id = cls._persisted_computer_id(root)
        shells = [item.id.value for item in shell_inventory().items]
        system = platform.system().lower()
        capabilities = ComputerCapabilities(
            platform="darwin" if system == "darwin" else "windows" if system == "windows" else "linux",
            architecture=platform.machine() or "unknown",
            runtime_version="cowork-code-runtime-1",
            agent_engines=registry.available_ids(),
            shells=shells,
            has_git=True,
            has_terminal=True,
            supports_local_folders=True,
            # One worker process retains one engine/workspace loop at a time.
            # Advertise that truthfully until the runtime supervisor fans out.
            max_concurrent_runs=1,
        )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{server_url.rstrip('/')}/api/v1/coding/runtime/register",
                json=ComputerRegistrationRequest(
                    registration_token=registration_token,
                    computer_id=persisted_id,
                    name=name,
                    capabilities=capabilities,
                ).model_dump(mode="json"),
            )
            cls._raise(response)
            registered = ComputerRegistrationResponse.model_validate(response.json())
        cls._save_computer_id(root, registered.computer.id)
        return cls(
            server_url,
            RuntimeIdentity(registered.computer.id, registered.runtime_token, registered.computer.name),
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def heartbeat(self, active_run_count: int = 0) -> None:
        response = self.client.post(
            self._url(f"computers/{self.identity.computer_id}/heartbeat"),
            headers=self._headers(),
            json={"protocol_version": RUNTIME_PROTOCOL_VERSION, "active_run_count": active_run_count},
        )
        self._raise(response)

    def lease(self, wait_seconds: float = 0) -> RuntimeLease | None:
        response = self.client.post(
            self._url(f"computers/{self.identity.computer_id}/lease"),
            headers=self._headers(),
            json={"protocol_version": RUNTIME_PROTOCOL_VERSION, "wait_seconds": wait_seconds},
        )
        self._raise(response)
        return RuntimeLease.model_validate(response.json()) if response.json() is not None else None

    def event(
        self,
        lease: RuntimeLease,
        kind: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        with self._sequence_lock:
            sequence = self._sequences.get(lease.run.id, lease.run.last_event_seq) + 1
            event = RuntimeEvent(
                run_id=lease.run.id,
                computer_id=self.identity.computer_id,
                lease_id=lease.lease_id,
                epoch=lease.run.epoch,
                seq=sequence,
                kind=kind,
                payload=payload or {},
            )
            response = self.client.post(
                self._url(f"runs/{lease.run.id}/events"),
                headers=self._headers(),
                json=event.model_dump(mode="json"),
            )
            self._raise(response)
            self._sequences[lease.run.id] = sequence

    def commands(self, lease: RuntimeLease) -> list[RuntimeCommand]:
        response = self.client.post(
            self._url(f"runs/{lease.run.id}/commands/claim"),
            params={"computer_id": self.identity.computer_id},
            headers=self._headers(),
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "lease_id": lease.lease_id,
                "epoch": lease.run.epoch,
            },
        )
        self._raise(response)
        return [RuntimeCommand.model_validate(item) for item in response.json().get("items", [])]

    def acknowledge(
        self,
        lease: RuntimeLease,
        command: RuntimeCommand,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        response = self.client.post(
            self._url(f"runs/{lease.run.id}/commands/ack"),
            params={"computer_id": self.identity.computer_id},
            headers=self._headers(),
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "lease_id": lease.lease_id,
                "epoch": lease.run.epoch,
                "command_id": command.id,
                "result": result,
                "error": error,
            },
        )
        self._raise(response)

    def inference_endpoint(self, lease: RuntimeLease) -> str:
        return self._url(
            f"computers/{self.identity.computer_id}/runs/{lease.run.id}/inference"
        )

    def _url(self, path: str) -> str:
        return f"{self.server_url}/api/v1/coding/runtime/{path}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.identity.runtime_token}"}

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = str(response.json().get("detail") or "")
            except (ValueError, AttributeError):
                pass
            raise RuntimeClientError(detail or f"Runtime request failed ({response.status_code})") from exc

    @staticmethod
    def _persisted_computer_id(root: Path) -> str | None:
        try:
            return (root / "computer-id").read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    @staticmethod
    def _save_computer_id(root: Path, computer_id: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        target = root / "computer-id"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(computer_id + "\n", encoding="utf-8")
        os.replace(temporary, target)


class CodeOnlyRuntime:
    """Prepare isolated workspaces and run an agent without a desktop UI."""

    def __init__(
        self,
        root: Path,
        client: RemoteRuntimeClient,
        registry: CodingEngineRegistry = engine_registry,
    ) -> None:
        self.root = root
        self.client = client
        self.registry = registry
        self.workspaces = ProjectWorkspaceManager(WorkspaceManager(root / "coding"))
        self._approval_lock = threading.Lock()
        self._approval_waiters: dict[str, tuple[threading.Event, dict[str, str]]] = {}
        self._pending_commands: list[RuntimeCommand] = []

    def run_once(self, wait_seconds: float = 0) -> bool:
        self.client.heartbeat()
        lease = self.client.lease(wait_seconds)
        if lease is None:
            return False
        try:
            self._execute(lease)
        except BaseException as exc:
            try:
                self.client.event(lease, "error", {"detail": str(exc)[:4_000]})
            except RuntimeClientError:
                pass
            raise
        return True

    def _execute(self, lease: RuntimeLease) -> None:
        if lease.project is None:
            raise RuntimeClientError("A remote Task Run requires portable project resources")
        can_restore = (
            lease.run.epoch > 1
            and bool(lease.workspaces)
            and all(
                item.computer_id == self.client.identity.computer_id
                for item in lease.workspaces
            )
        )
        prepared = (
            self.workspaces.restore(lease.task.id, lease.project, lease.workspaces)
            if can_restore
            else self.workspaces.prepare(lease.task.id, lease.project)
        )
        self.client.event(lease, "workspace", {
            "items": [item.model_dump(mode="json") for item in prepared.workspaces],
            "ports": prepared.ports,
        })
        setup_note = self._run_setup(lease, prepared)
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
                developer_instructions="\n\n".join(
                    item for item in (lease.execution.developer_instructions, setup_note) if item
                ),
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

    def _run_setup(self, lease: RuntimeLease, prepared: PreparedProjectWorkspace) -> str:
        assert lease.project is not None
        results = self.workspaces.run_commands(
            lease.project,
            list(prepared.workspaces),
            "setup",
            prepared.ports,
        )
        for result in results:
            event = CodingEvent(
                type=EventType.command,
                title=result.label,
                text=result.output,
                phase="completed" if result.return_code == 0 else "failed",
                data={
                    "folderId": result.folder_id,
                    "returnCode": result.return_code,
                    "phase": "setup",
                },
            )
            self.client.event(lease, "event", {"event": event.model_dump(mode="json")})
        failed = next((result for result in results if result.return_code != 0), None)
        if failed is None:
            return ""
        return (
            f"MindsHub Code setup note: {failed.label!r} failed in project resource "
            f"{failed.folder_id!r}. Inspect the workspace and recover as part of the task "
            f"when relevant.\nSetup output:\n{failed.output[:8_000]}"
        )

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
        turn_id = session.start_turn(str(start.payload.get("prompt") or lease.task.prompt))
        self.client.acknowledge(lease, start)
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
    server_url = os.environ.get("COWORK_RUNTIME_SERVER", "").strip()
    registration_token = os.environ.get("COWORK_RUNTIME_REGISTRATION_TOKEN", "").strip()
    root_value = os.environ.get("COWORK_RUNTIME_ROOT", "").strip()
    name = os.environ.get("COWORK_RUNTIME_NAME", platform.node() or "Code runtime")
    if not server_url or not registration_token or not root_value:
        raise SystemExit("Runtime server, registration token, and root are required")
    root = Path(root_value).expanduser().resolve()
    client = RemoteRuntimeClient.register(server_url, registration_token, name, root)
    runtime = CodeOnlyRuntime(root, client)
    try:
        while True:
            runtime.run_once(wait_seconds=20)
    finally:
        client.close()


if __name__ == "__main__":
    main()
