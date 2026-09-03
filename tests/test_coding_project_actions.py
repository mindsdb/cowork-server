from __future__ import annotations

import base64
from pathlib import Path

import pytest

from cowork.coding.contracts import (
    CodingSession,
    TaskWorkspace,
    TerminalPage,
    TerminalShellPreference,
    TerminalTabPage,
    TerminalTabState,
    WorkspaceKind,
)
from cowork.coding.control_models import Computer, ComputerCapabilities
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.project_actions import ProjectActionService, terminal_command_line
from cowork.coding.project_models import CodeProject, ProjectActionRunRequest
from cowork.coding.workspace import WorkspaceError


def computer(platform: str = "darwin", shells: list[str] | None = None) -> Computer:
    return Computer(
        id="local",
        name="This computer",
        is_local=True,
        capabilities=ComputerCapabilities(
            platform=platform,
            architecture="arm64",
            runtime_version="test",
            agent_engines=["codex"],
            shells=shells or ["auto", "bash"],
        ),
    )


def project(path: Path) -> CodeProject:
    return CodeProject(
        id="project-1",
        name="Website",
        resources=[{
            "kind": "local_folder",
            "id": "web",
            "name": "Web",
            "path": str(path),
            "computer_id": "local",
            "commands": [{
                "id": "web-run",
                "label": "Dev server",
                "argv": ["npm", "run", "dev"],
                "phase": "run",
            }],
        }],
    )


def coding_session(path: Path) -> CodingSession:
    return CodingSession(
        id="session-1",
        title="Build site",
        engine_id="codex",
        engine_adapter_version="1",
        model="gpt",
        task_id="task-1",
        computer_id="local",
        project_id="project-1",
        source_path=str(path),
        workspace_path=str(path),
        workspace_kind=WorkspaceKind.local_copy,
        workspaces=[TaskWorkspace(
            folder_id="web",
            folder_name="Web",
            source_path=str(path),
            workspace_path=str(path),
            workspace_kind=WorkspaceKind.local_copy,
            source_dirty=False,
        )],
        allocated_ports={"PORT": 4173},
    )


class Terminals:
    def __init__(self) -> None:
        self.written = ""
        self.deleted: list[str] = []
        self.tabs: list[TerminalTabState] = []

    def list(self, _session_id: str) -> TerminalTabPage:
        return TerminalTabPage(items=self.tabs)

    def create(
        self,
        _session_id: str,
        label: str,
        *,
        project_action_id: str | None = None,
        project_resource_id: str | None = None,
    ) -> TerminalTabState:
        tab = TerminalTabState(
            id="terminal-1",
            label=label,
            project_action_id=project_action_id,
            project_resource_id=project_resource_id,
        )
        self.tabs.append(tab)
        return tab

    def start(self, *_args, **_kwargs) -> TerminalPage:
        self.tabs[0].status = "running"
        return TerminalPage(status="running")

    def write(self, _session_id: str, _terminal_id: str, data_base64: str) -> TerminalPage:
        self.written = base64.b64decode(data_base64).decode()
        return TerminalPage(status="running")

    def delete(self, _session_id: str, terminal_id: str) -> None:
        self.deleted.append(terminal_id)


def test_serializes_project_actions_for_supported_shells() -> None:
    assert terminal_command_line(
        argv=["npm", "run", "dev server"],
        cwd="/tmp/My App",
        computer=computer(),
        shell=TerminalShellPreference.auto,
    ) == "cd '/tmp/My App' && npm run 'dev server'\n"
    assert terminal_command_line(
        argv=["npm", "run", "dev"],
        cwd=r"C:\Code\My App",
        computer=computer("windows", ["auto", "pwsh"]),
        shell=TerminalShellPreference.auto,
    ) == "Set-Location -LiteralPath 'C:\\Code\\My App'; & 'npm' 'run' 'dev'\r\n"
    assert terminal_command_line(
        argv=["npm", "run", "dev"],
        cwd=r"C:\Code\My App",
        computer=computer("windows", ["auto", "cmd"]),
        shell=TerminalShellPreference.auto,
    ) == 'cd /d "C:\\Code\\My App" && npm run dev\r\n'


def test_runs_scoped_action_in_named_terminal(tmp_path: Path) -> None:
    session = coding_session(tmp_path)
    configured = project(tmp_path)
    terminals = Terminals()
    service = ProjectActionService(
        get_session=lambda _id: session,
        projects=object(),  # type: ignore[arg-type]
        terminals=terminals,  # type: ignore[arg-type]
        get_computer=lambda _id: computer(),
        local_computer_id="local",
        execution_project=lambda _session: configured,
        port_open=lambda port: port == 4173,
    )

    page = service.list(session.id)
    assert [(item.label, item.resource_name) for item in page.items] == [("Dev server", "Web")]
    assert page.preview_url is None
    assert page.preview_pending is False

    result = service.run(
        session.id,
        ProjectActionRunRequest(resource_id="web", command_id="web-run"),
        EngineCredentials("https://example.test", "secret"),
    )

    assert result.terminal_id == "terminal-1"
    assert result.label == "Dev server"
    assert result.preview_url == "http://127.0.0.1:4173"
    assert terminals.written == f"cd {tmp_path} && npm run dev\n"
    assert terminals.tabs[0].project_action_id == "web-run"
    assert terminals.tabs[0].project_resource_id == "web"

    assert service.list(session.id).preview_url == "http://127.0.0.1:4173"


def test_rejects_actions_outside_task_scope(tmp_path: Path) -> None:
    session = coding_session(tmp_path)
    session.workspaces = []
    service = ProjectActionService(
        get_session=lambda _id: session,
        projects=object(),  # type: ignore[arg-type]
        terminals=Terminals(),  # type: ignore[arg-type]
        get_computer=lambda _id: computer(),
        local_computer_id="local",
        execution_project=lambda _session: project(tmp_path),
    )

    with pytest.raises(WorkspaceError, match="outside this task's scope"):
        service.run(
            session.id,
            ProjectActionRunRequest(resource_id="web", command_id="web-run"),
            EngineCredentials("https://example.test", "secret"),
        )


def test_preview_waits_until_the_run_command_actually_listens(tmp_path: Path) -> None:
    # The action runs in an interactive shell, so the terminal stays "running"
    # after `npm run dev` has died; the preview must follow the port, not the tab.
    session = coding_session(tmp_path)
    configured = project(tmp_path)
    terminals = Terminals()
    listening = {"open": False}
    service = ProjectActionService(
        get_session=lambda _id: session,
        projects=object(),  # type: ignore[arg-type]
        terminals=terminals,  # type: ignore[arg-type]
        get_computer=lambda _id: computer(),
        local_computer_id="local",
        execution_project=lambda _session: configured,
        port_open=lambda _port: listening["open"],
    )

    result = service.run(
        session.id,
        ProjectActionRunRequest(resource_id="web", command_id="web-run"),
        EngineCredentials("https://example.test", "secret"),
    )
    assert result.preview_url is None
    assert result.preview_pending is True

    listening["open"] = True
    page = service.list(session.id)
    assert page.preview_url == "http://127.0.0.1:4173"
    assert page.preview_pending is False

    listening["open"] = False  # the dev server exited; the shell is still open
    page = service.list(session.id)
    assert page.preview_url is None
    assert page.preview_pending is True


def test_default_port_probe_reports_a_real_listener() -> None:
    import socket

    from cowork.coding.project_actions import port_is_listening

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert port_is_listening(port) is True
    assert port_is_listening(port) is False
