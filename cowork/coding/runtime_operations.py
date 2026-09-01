from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from cowork.coding.contracts import (
    DeliveryRecord,
    TerminalPage,
    TerminalShellPreference,
    WorkspaceKind,
)
from cowork.coding.control_models import RuntimeCommand
from cowork.coding.delivery import ProjectDeliveryService
from cowork.coding.engines.base import EngineSession
from cowork.coding.project_workspaces import (
    PreparedProjectWorkspace,
    ProjectWorkspaceManager,
)
from cowork.coding.runtime_protocol import RuntimeLease
from cowork.coding.terminal import TerminalBuffer
from cowork.coding.workspace import WorkspaceError


class RuntimeWorkspaceOperations:
    """Execute workspace-owned commands behind the fenced runtime protocol."""

    def __init__(
        self,
        lease: RuntimeLease,
        manager: ProjectWorkspaceManager,
        prepared: PreparedProjectWorkspace,
        engine: EngineSession,
    ) -> None:
        if lease.project is None:
            raise RuntimeError("Runtime workspace operations require a Code Project")
        self.lease = lease
        self.project = lease.project
        self.manager = manager
        self.prepared = prepared
        self.engine = engine
        self._lock = threading.RLock()
        self._terminals: dict[str, TerminalBuffer] = {}
        self._completed: dict[str, tuple[dict[str, object] | None, str | None]] = {}
        self._handlers: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
            "git_state": self._git_state,
            "git_states": self._git_states,
            "diff": self._diff,
            "review_file": self._review_file,
            "branch": self._branch,
            "commit": self._commit,
            "validate": self._validate,
            "delivery_plan": self._delivery_plan,
            "push_branch": self._push_branch,
            "terminal_page": self._terminal_page,
            "terminal_start": self._terminal_start,
            "terminal_input": self._terminal_input,
            "terminal_resize": self._terminal_resize,
            "terminal_stop": self._terminal_stop,
            "terminal_remove": self._terminal_remove,
        }

    def execute(self, command: RuntimeCommand) -> tuple[dict[str, object] | None, str | None]:
        """Run each command ID once, even when an acknowledgement is retried."""

        with self._lock:
            cached = self._completed.get(command.id)
            if cached is not None:
                return cached
            operation = str(command.payload.get("operation") or "")
            handler = self._handlers.get(operation)
            if handler is None:
                result = (None, "The selected computer does not support that operation")
            else:
                try:
                    result = (handler(command.payload), None)
                except Exception as exc:  # noqa: BLE001 - crosses a typed process boundary.
                    result = (None, str(exc)[:4_000])
            self._completed[command.id] = result
            return result

    def release(self) -> None:
        with self._lock:
            for terminal in self._terminals.values():
                if terminal.is_running:
                    try:
                        self.engine.stop_terminal(terminal.process_id)
                    except Exception:  # noqa: BLE001 - cleanup is best effort before workspace removal.
                        pass
            self._terminals.clear()
        self.manager.cleanup(self.lease.task.id, list(self.prepared.workspaces))

    def _git_state(self, _payload: dict[str, object]) -> dict[str, object]:
        states = self.manager.git_states(list(self.prepared.workspaces))
        if not states:
            raise WorkspaceError("This task does not contain a Git repository")
        return states[0].model_dump(mode="json")

    def _git_states(self, _payload: dict[str, object]) -> dict[str, object]:
        return {"items": [item.model_dump(mode="json") for item in self.manager.git_states(list(self.prepared.workspaces))]}

    def _diff(self, _payload: dict[str, object]) -> dict[str, object]:
        return {"files": [item.model_dump(mode="json") for item in self.manager.diff(list(self.prepared.workspaces))]}

    def _review_file(self, payload: dict[str, object]) -> dict[str, object]:
        folder_id = str(payload.get("folder_id") or "")
        workspace = next((item for item in self.prepared.workspaces if item.folder_id == folder_id), None)
        if workspace is None:
            raise WorkspaceError("Choose a file from this task")
        if workspace.workspace_kind != WorkspaceKind.git_worktree:
            raise WorkspaceError("Review actions require a Git task workspace")
        self.manager.review_file_action(workspace, str(payload.get("path") or ""), str(payload.get("action") or ""))
        return self._diff({})

    def _branch(self, payload: dict[str, object]) -> dict[str, object]:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise WorkspaceError("A branch name is required")
        states = []
        for workspace in self.prepared.workspaces:
            if workspace.workspace_kind != WorkspaceKind.git_worktree:
                continue
            state = self.manager.workspaces.create_branch(workspace.workspace_path, name)
            states.append(state.model_copy(update={
                "folder_id": workspace.folder_id,
                "folder_name": workspace.folder_name,
            }))
        if not states:
            raise WorkspaceError("This task does not contain a Git repository")
        return states[0].model_dump(mode="json")

    def _commit(self, payload: dict[str, object]) -> dict[str, object]:
        message = str(payload.get("message") or "").strip()
        if not message:
            raise WorkspaceError("A commit message is required")
        states = self.manager.commit(list(self.prepared.workspaces), message)
        if not states:
            raise WorkspaceError("This task does not contain a Git repository")
        return states[0].model_dump(mode="json")

    def _validate(self, _payload: dict[str, object]) -> dict[str, object]:
        results = self.manager.run_commands(
            self.project,
            list(self.prepared.workspaces),
            "validate",
            self.prepared.ports,
        )
        return {"items": [asdict(item) for item in results]}

    def _delivery_plan(self, payload: dict[str, object]) -> dict[str, object]:
        raw_deliveries = payload.get("deliveries")
        deliveries = [
            DeliveryRecord.model_validate(item)
            for item in raw_deliveries
        ] if isinstance(raw_deliveries, list) else []
        return ProjectDeliveryService(self.manager.workspaces.git).plan_workspaces(
            list(self.prepared.workspaces),
            deliveries,
        ).model_dump(mode="json")

    def _push_branch(self, payload: dict[str, object]) -> dict[str, object]:
        folder_id = str(payload.get("folder_id") or "")
        workspace = next(
            (item for item in self.prepared.workspaces if item.folder_id == folder_id),
            None,
        )
        if workspace is None or workspace.workspace_kind != WorkspaceKind.git_worktree:
            raise WorkspaceError("The selected project resource is not a Git workspace")
        root = Path(workspace.workspace_path)
        branch = self.manager.workspaces.git.run(
            root,
            "branch",
            "--show-current",
            check=False,
        ).stdout.strip()
        if not branch:
            raise WorkspaceError("Create a task branch before publishing this resource")
        self.manager.workspaces.git.run(root, "push", "origin", f"{branch}:{branch}")
        return {"folder_id": folder_id, "task_branch": branch}

    def _terminal_page(self, payload: dict[str, object]) -> dict[str, object]:
        terminal = self._terminals.get(self._terminal_id(payload))
        after = max(0, int(payload.get("after") or 0))
        wait = min(15.0, max(0.0, float(payload.get("wait") or 0)))
        page = terminal.wait(after, wait) if terminal is not None and wait else (
            terminal.page(after) if terminal is not None else TerminalPage()
        )
        return page.model_dump(mode="json")

    def _terminal_start(self, payload: dict[str, object]) -> dict[str, object]:
        terminal_id = self._terminal_id(payload)
        current = self._terminals.get(terminal_id)
        if current is not None and current.is_running:
            return current.page().model_dump(mode="json")
        terminal = TerminalBuffer(str(uuid.uuid4()))
        self._terminals[terminal_id] = terminal
        try:
            shell = TerminalShellPreference(str(payload.get("shell") or "auto"))
            self.engine.start_terminal(
                terminal.process_id,
                int(payload.get("cols") or 120),
                int(payload.get("rows") or 30),
                shell,
                terminal.append,
                terminal.finish,
            )
        except Exception:
            terminal.finish(None, "Terminal process failed to start")
            raise
        return terminal.page().model_dump(mode="json")

    def _terminal_input(self, payload: dict[str, object]) -> dict[str, object]:
        terminal = self._active_terminal(payload)
        self.engine.write_terminal(terminal.process_id, str(payload.get("data_base64") or ""))
        return terminal.page().model_dump(mode="json")

    def _terminal_resize(self, payload: dict[str, object]) -> dict[str, object]:
        terminal = self._active_terminal(payload)
        self.engine.resize_terminal(
            terminal.process_id,
            int(payload.get("cols") or 120),
            int(payload.get("rows") or 30),
        )
        return terminal.page().model_dump(mode="json")

    def _terminal_stop(self, payload: dict[str, object]) -> dict[str, object]:
        terminal = self._active_terminal(payload)
        self.engine.stop_terminal(terminal.process_id)
        return terminal.page().model_dump(mode="json")

    def _terminal_remove(self, payload: dict[str, object]) -> dict[str, object]:
        terminal = self._terminals.pop(self._terminal_id(payload), None)
        if terminal is not None and terminal.is_running:
            self.engine.stop_terminal(terminal.process_id)
        return {}

    def _active_terminal(self, payload: dict[str, object]) -> TerminalBuffer:
        terminal = self._terminals.get(self._terminal_id(payload))
        if terminal is None or not terminal.is_running:
            raise RuntimeError("There is no running terminal for this coding task")
        return terminal

    @staticmethod
    def _terminal_id(payload: dict[str, object]) -> str:
        terminal_id = str(payload.get("terminal_id") or "")
        if not terminal_id:
            raise ValueError("A terminal ID is required")
        return terminal_id
