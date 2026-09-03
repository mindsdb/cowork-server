from __future__ import annotations

import base64
import shlex
import socket
import subprocess
from collections.abc import Callable

from cowork.coding.contracts import CodingSession, TerminalShellPreference, TerminalStatus
from cowork.coding.control_models import Computer
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.project_models import (
    CodeProject,
    ProjectActionRunRequest,
    ProjectActionRunResponse,
    ProjectActionPage,
    ProjectActionSummary,
)
from cowork.coding.project_service import CodeProjectService
from cowork.coding.terminal_service import TaskTerminalService
from cowork.coding.workspace import WorkspaceError


PORT_PROBE_TIMEOUT_SECONDS = 0.15


def port_is_listening(port: int) -> bool:
    """True when something accepts TCP connections on the loopback port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=PORT_PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def terminal_command_line(
    *,
    argv: list[str],
    cwd: str,
    computer: Computer,
    shell: TerminalShellPreference,
) -> str:
    """Serialize one trusted structured project action for its chosen shell."""

    windows = computer.capabilities.platform == "windows"
    shells = set(computer.capabilities.shells)
    resolved = shell
    if windows and shell in {TerminalShellPreference.auto, TerminalShellPreference.system}:
        resolved = next(
            (
                candidate
                for candidate in (
                    TerminalShellPreference.bash,
                    TerminalShellPreference.pwsh,
                    TerminalShellPreference.powershell,
                    TerminalShellPreference.cmd,
                )
                if candidate.value in shells
            ),
            TerminalShellPreference.cmd,
        )
    posix = not windows or resolved == TerminalShellPreference.bash
    if posix:
        return f"cd {shlex.quote(cwd)} && {shlex.join(argv)}\n"
    if resolved == TerminalShellPreference.cmd:
        if any(any(character in value for character in "&|<>^%\r\n") for value in (cwd, *argv)):
            raise WorkspaceError("Managed project actions need Bash or PowerShell when values contain shell metacharacters")
        return f"cd /d {subprocess.list2cmdline([cwd])} && {subprocess.list2cmdline(argv)}\r\n"
    return (
        f"Set-Location -LiteralPath {_powershell_quote(cwd)}; "
        f"& {_powershell_quote(argv[0])} {' '.join(_powershell_quote(value) for value in argv[1:])}\r\n"
    )


class ProjectActionService:
    """Launch durable project run actions through the existing task terminal."""

    def __init__(
        self,
        *,
        get_session: Callable[[str], CodingSession],
        projects: CodeProjectService,
        terminals: TaskTerminalService,
        get_computer: Callable[[str], Computer],
        local_computer_id: str,
        execution_project: Callable[[CodingSession], CodeProject | None],
        port_open: Callable[[int], bool] = port_is_listening,
    ) -> None:
        self.get_session = get_session
        self.projects = projects
        self.terminals = terminals
        self.get_computer = get_computer
        self.local_computer_id = local_computer_id
        self.execution_project = execution_project
        self.port_open = port_open

    def list(self, session_id: str) -> ProjectActionPage:
        session = self.get_session(session_id)
        project = self.execution_project(session)
        if project is None and session.project_id:
            project = self.projects.get(session.project_id)
        if project is None:
            return ProjectActionPage()
        scoped = {item.folder_id for item in session.workspaces}
        items = [
            ProjectActionSummary(
                id=command.id,
                resource_id=resource.id,
                label=command.label,
                resource_name=resource.name,
            )
            for resource in project.resources
            if resource.id in scoped
            for command in resource.commands
            if command.phase == "run"
        ]
        active_actions = {
            (tab.project_resource_id, tab.project_action_id)
            for tab in self.terminals.list(session_id).items
            if tab.status == TerminalStatus.running
            and tab.project_resource_id
            and tab.project_action_id
        }
        has_active_run = any((item.resource_id, item.id) in active_actions for item in items)
        port = next(iter(session.allocated_ports.values()), None) if has_active_run else None
        # The action runs inside an interactive shell, so its terminal stays
        # "running" after the command has exited. Only offer a preview while
        # something actually answers on the port; until then it is pending.
        previewable = bool(port) and session.computer_is_local
        listening = previewable and self.port_open(port)
        return ProjectActionPage(
            items=items,
            preview_url=f"http://127.0.0.1:{port}" if listening else None,
            preview_pending=previewable and not listening,
        )

    def run(
        self,
        session_id: str,
        request: ProjectActionRunRequest,
        credentials: EngineCredentials,
    ) -> ProjectActionRunResponse:
        session = self.get_session(session_id)
        project = self.execution_project(session)
        if project is None:
            if not session.project_id:
                raise WorkspaceError("This task is not linked to a Code Project")
            project = self.projects.get(session.project_id)
        try:
            resource = next(item for item in project.resources if item.id == request.resource_id)
        except StopIteration as exc:
            raise WorkspaceError("That project resource is not available to this task") from exc
        try:
            command = next(
                item for item in resource.commands
                if item.id == request.command_id and item.phase == "run"
            )
        except StopIteration as exc:
            raise WorkspaceError("That run action is no longer available to this task") from exc
        try:
            workspace = next(item for item in session.workspaces if item.folder_id == resource.id)
        except StopIteration as exc:
            raise WorkspaceError("That project resource is outside this task's scope") from exc

        computer_id = session.computer_id or self.local_computer_id
        computer = self.get_computer(computer_id)
        line = terminal_command_line(
            argv=command.argv,
            cwd=workspace.workspace_path,
            computer=computer,
            shell=request.shell,
        )
        tab = self.terminals.create(
            session_id,
            command.label,
            project_action_id=command.id,
            project_resource_id=resource.id,
        )
        try:
            self.terminals.start(
                session_id,
                tab.id,
                credentials,
                request.cols,
                request.rows,
                request.shell,
            )
            self.terminals.write(
                session_id,
                tab.id,
                base64.b64encode(line.encode("utf-8")).decode("ascii"),
            )
        except Exception:
            try:
                self.terminals.delete(session_id, tab.id)
            except Exception:
                pass
            raise

        page = self.list(session_id)
        return ProjectActionRunResponse(
            terminal_id=tab.id,
            label=command.label,
            preview_url=page.preview_url,
            preview_pending=page.preview_pending,
        )
