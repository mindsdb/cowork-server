from __future__ import annotations

from cowork.coding.contracts import (
    TerminalPage,
    TerminalShellPreference,
    TerminalTabPage,
    TerminalTabState,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.terminal_service import TaskTerminalService


class CodingTerminalOperations:
    """Thin compatibility facade over the task-owned terminal service."""

    task_terminals: TaskTerminalService

    def terminals(self, session_id: str) -> TerminalTabPage:
        return self.task_terminals.list(session_id)

    def create_terminal_tab(self, session_id: str, label: str | None = None) -> TerminalTabState:
        return self.task_terminals.create(session_id, label)

    def rename_terminal_tab(self, session_id: str, terminal_id: str, label: str) -> TerminalTabState:
        return self.task_terminals.rename(session_id, terminal_id, label)

    def delete_terminal_tab(self, session_id: str, terminal_id: str) -> None:
        self.task_terminals.delete(session_id, terminal_id)

    def terminal_tab(self, session_id: str, terminal_id: str, after: int = 0) -> TerminalPage:
        return self.task_terminals.page(session_id, terminal_id, after)

    def wait_for_terminal_tab(
        self,
        session_id: str,
        terminal_id: str,
        after: int,
        timeout: float = 15.0,
    ) -> TerminalPage:
        return self.task_terminals.wait(session_id, terminal_id, after, timeout)

    def start_terminal_tab(
        self,
        session_id: str,
        terminal_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        return self.task_terminals.start(session_id, terminal_id, credentials, cols, rows, shell)

    def write_terminal_tab(self, session_id: str, terminal_id: str, data_base64: str) -> TerminalPage:
        return self.task_terminals.write(session_id, terminal_id, data_base64)

    def resize_terminal_tab(self, session_id: str, terminal_id: str, cols: int, rows: int) -> TerminalPage:
        return self.task_terminals.resize(session_id, terminal_id, cols, rows)

    def stop_terminal_tab(self, session_id: str, terminal_id: str) -> TerminalPage:
        return self.task_terminals.stop(session_id, terminal_id)

    # Compatibility seam for the original one-terminal desktop. Deployments
    # can roll the server and renderer independently without breaking a task
    # that still speaks the singular endpoint contract.
    def terminal(self, session_id: str, after: int = 0) -> TerminalPage:
        return self.task_terminals.legacy_page(session_id, after)

    def wait_for_terminal(self, session_id: str, after: int, timeout: float = 15.0) -> TerminalPage:
        return self.task_terminals.legacy_wait(session_id, after, timeout)

    def start_terminal(
        self,
        session_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        return self.task_terminals.legacy_start(session_id, credentials, cols, rows, shell)

    def write_terminal(self, session_id: str, data_base64: str) -> TerminalPage:
        return self.task_terminals.legacy_write(session_id, data_base64)

    def resize_terminal(self, session_id: str, cols: int, rows: int) -> TerminalPage:
        return self.task_terminals.legacy_resize(session_id, cols, rows)

    def stop_terminal(self, session_id: str) -> TerminalPage:
        return self.task_terminals.legacy_stop(session_id)
