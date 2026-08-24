from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cowork.coding.contracts import CodingSession, TerminalPage
from cowork.coding.engines.base import (
    EngineCredentials,
    EngineSession,
    EngineSessionConfig,
)
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.terminal import TerminalBuffer

ApprovalRequest = Callable[[str, str, dict[str, Any] | None], dict[str, str]]


def engine_workspace_path(session: CodingSession) -> str:
    """Return the narrow task root presented to an agent runtime.

    The persisted ``workspace_path`` remains the primary folder for review and
    handoff. Project runtimes instead start at the common, task-owned parent so
    thread-level operations such as Codex goals can write every project folder.
    """
    if not session.workspaces:
        return session.workspace_path
    parents = {Path(item.workspace_path).resolve().parent for item in session.workspaces}
    if len(parents) != 1:
        return session.workspace_path
    root = parents.pop()
    return str(root) if root.name == session.id else session.workspace_path


class RuntimeManager:
    """Own persistent engine runtimes and task terminals.

    Per-task locks serialize runtime creation and teardown without holding the
    CodingService state lock around app-server I/O.
    """

    def __init__(
        self,
        root: Path,
        registry: CodingEngineRegistry,
        approval_request: ApprovalRequest,
    ) -> None:
        self._root = root
        self._registry = registry
        self._approval_request = approval_request
        self._lock = threading.RLock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._runtimes: dict[str, EngineSession] = {}
        self._terminals: dict[str, TerminalBuffer] = {}
        self._closed = False

    def session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def open(self, session: CodingSession, credentials: EngineCredentials) -> EngineSession:
        with self.session_lock(session.id):
            return self.open_locked(session, credentials)

    def open_locked(self, session: CodingSession, credentials: EngineCredentials) -> EngineSession:
        """Open or reuse a runtime while the caller owns ``session_lock``."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The coding runtime manager is shutting down")
            existing = self._runtimes.get(session.id)
            if existing is not None and not existing.is_closed:
                return existing
            self._runtimes.pop(session.id, None)

        engine = self._registry.get(session.engine_id)
        runtime_workspace = engine_workspace_path(session)
        opened = engine.open_session(
            cowork_root=str(self._root),
            workspace=runtime_workspace,
            config=EngineSessionConfig(
                model=session.model,
                permission_mode=session.permission_mode,
                reasoning_effort=session.reasoning_effort,
                service_tier=None if session.service_tier == "standard" else session.service_tier,
                personality=session.personality,
                network_access=session.network_access,
                web_search=session.web_search,
                additional_dirs=tuple(session.additional_dirs),
                developer_instructions=session.developer_instructions,
                skill_roots=tuple(session.skill_roots),
                environment=tuple(session.environment.items()),
                session_id=session.id,
                cowork_root=str(self._root),
            ),
            credentials=credentials,
            existing_session_id=session.engine_session_id,
            approval_handler=lambda method, params: self._approval_request(session.id, method, params),
        )
        with self._lock:
            closing = self._closed
            if not closing:
                self._runtimes[session.id] = opened
        if closing:
            opened.close()
            raise RuntimeError("The coding runtime manager is shutting down")
        return opened

    def close_session(self, session_id: str) -> None:
        with self.session_lock(session_id):
            self.close_locked(session_id)

    def close_locked(self, session_id: str) -> None:
        """Close a runtime while the caller owns ``session_lock``."""
        with self._lock:
            runtime = self._runtimes.pop(session_id, None)
            terminal = self._terminals.pop(session_id, None)
        if runtime is not None:
            runtime.close()
        if terminal is not None and terminal.is_running:
            terminal.finish(None, None)

    def discard_if_closed(self, session_id: str, runtime: EngineSession) -> None:
        if not runtime.is_closed:
            return
        with self._lock:
            if self._runtimes.get(session_id) is runtime:
                self._runtimes.pop(session_id, None)

    def terminal_page(self, session_id: str, after: int = 0) -> TerminalPage:
        with self._lock:
            terminal = self._terminals.get(session_id)
        return terminal.page(after) if terminal is not None else TerminalPage()

    def wait_for_terminal(self, session_id: str, after: int, timeout: float) -> TerminalPage:
        with self._lock:
            terminal = self._terminals.get(session_id)
        return terminal.wait(after, timeout) if terminal is not None else TerminalPage()

    def terminal_is_running(self, session_id: str) -> bool:
        with self._lock:
            terminal = self._terminals.get(session_id)
        return terminal is not None and terminal.is_running

    def start_terminal(
        self,
        session: CodingSession,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
    ) -> TerminalPage:
        with self.session_lock(session.id):
            return self.start_terminal_locked(session, credentials, cols, rows)

    def start_terminal_locked(
        self,
        session: CodingSession,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
    ) -> TerminalPage:
        """Start a terminal while the caller owns ``session_lock``."""
        with self._lock:
            current = self._terminals.get(session.id)
        if current is not None and current.is_running:
            return current.page()

        runtime = self.open_locked(session, credentials)
        process_id = str(uuid.uuid4())
        terminal = TerminalBuffer(process_id)
        with self._lock:
            self._terminals[session.id] = terminal
        try:
            runtime.start_terminal(process_id, cols, rows, terminal.append, terminal.finish)
        except Exception:
            terminal.finish(None, "Terminal process failed to start")
            raise
        return terminal.page()

    def write_terminal(self, session_id: str, data_base64: str) -> TerminalPage:
        runtime, terminal = self._active_terminal(session_id)
        runtime.write_terminal(terminal.process_id, data_base64)
        return terminal.page()

    def resize_terminal(self, session_id: str, cols: int, rows: int) -> TerminalPage:
        runtime, terminal = self._active_terminal(session_id)
        runtime.resize_terminal(terminal.process_id, cols, rows)
        return terminal.page()

    def stop_terminal(self, session_id: str) -> TerminalPage:
        runtime, terminal = self._active_terminal(session_id)
        runtime.stop_terminal(terminal.process_id)
        return terminal.page()

    def terminal_status(self, session_id: str) -> str:
        return self.terminal_page(session_id).status.value

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            runtimes = list(self._runtimes.values())
            terminals = list(self._terminals.values())
            self._runtimes.clear()
            self._terminals.clear()
        for runtime in runtimes:
            runtime.close()
        for terminal in terminals:
            if terminal.is_running:
                terminal.finish(None, None)

    def _active_terminal(self, session_id: str) -> tuple[EngineSession, TerminalBuffer]:
        with self._lock:
            runtime = self._runtimes.get(session_id)
            terminal = self._terminals.get(session_id)
        if terminal is None or not terminal.is_running:
            raise RuntimeError("There is no running terminal for this coding task")
        if runtime is None or runtime.is_closed:
            terminal.finish(None, "The coding runtime disconnected")
            raise RuntimeError("The coding runtime disconnected")
        return runtime, terminal
