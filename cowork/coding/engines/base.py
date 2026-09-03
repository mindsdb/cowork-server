from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from cowork.coding.contracts import (
    CodingEvent,
    EngineCapabilities,
    ExtensionInventory,
    PermissionMode,
    RuntimePlatformStatus,
    TerminalShellPreference,
)

ApprovalHandler = Callable[[str, dict[str, Any] | None], dict[str, str]]
TerminalOutputHandler = Callable[[str, str, bool], None]
TerminalExitHandler = Callable[[int | None, str | None], None]


@dataclass(frozen=True)
class EngineCredentials:
    minds_url: str
    minds_api_key: str


@dataclass(frozen=True)
class EngineMcpServer:
    """Agent-neutral process contract for a task-scoped MCP server."""

    name: str
    command: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class EngineSessionConfig:
    model: str
    permission_mode: PermissionMode
    reasoning_effort: str | None = None
    service_tier: str | None = None
    personality: str | None = None
    network_access: bool = False
    web_search: bool = False
    additional_dirs: tuple[str, ...] = ()
    developer_instructions: str = ""
    skill_roots: tuple[str, ...] | None = None
    environment: tuple[tuple[str, str], ...] = ()
    session_id: str = ""
    cowork_root: str = ""
    workspace_label: str = ""
    inference_base_url: str = ""
    inference_api_key: str = ""
    mcp_servers: tuple[EngineMcpServer, ...] = ()


@dataclass(frozen=True)
class EngineInputReference:
    name: str
    path: str
    kind: str
    resource_id: str | None = None
    relative_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    content_hash: str | None = None



class SteerOutcome:
    """A steer the engine has not confirmed within the deadline.

    The instruction may still be applied once the engine answers, so the
    caller records it as unconfirmed and learns the result through
    ``on_settled`` instead of reporting a rejection that could turn out wrong.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settled: tuple[bool, str] | None = None
        self._callbacks: list[Callable[[bool, str], None]] = []

    @property
    def settled(self) -> tuple[bool, str] | None:
        with self._lock:
            return self._settled

    def settle(self, ok: bool, detail: str = "") -> None:
        with self._lock:
            if self._settled is not None:
                return
            self._settled = (ok, detail)
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(ok, detail)

    def on_settled(self, callback: Callable[[bool, str], None]) -> None:
        with self._lock:
            if self._settled is None:
                self._callbacks.append(callback)
                return
            ok, detail = self._settled
        callback(ok, detail)

class EngineSession(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def is_closed(self) -> bool: ...

    def start_turn(
        self,
        prompt: str,
        attachments: tuple[EngineInputReference, ...] = (),
    ) -> str: ...

    def start_goal(self, objective: str) -> str: ...

    def resume_goal(self) -> str: ...

    def update_goal(self, action: str, objective: str | None = None) -> dict[str, Any] | None: ...

    def start_review(self) -> str: ...

    def events(self, turn_id: str) -> Iterator[CodingEvent]: ...

    def steer(
        self,
        turn_id: str,
        prompt: str,
        attachments: tuple[EngineInputReference, ...] = (),
    ) -> SteerOutcome | None:
        """Deliver guidance to the active turn.

        Returns None once the engine confirmed it, or a SteerOutcome when the
        engine has not answered within its deadline.
        """
        ...

    def cancel(self, turn_id: str) -> None: ...

    def compact(self) -> None: ...

    def goal_status(self) -> dict[str, Any] | None: ...

    def extension_inventory(self) -> ExtensionInventory: ...

    def fork(self, workspace: str, additional_dirs: tuple[str, ...] = ()) -> str: ...

    def platform_status(self) -> RuntimePlatformStatus: ...

    def setup_windows_sandbox(self) -> RuntimePlatformStatus: ...

    def start_terminal(
        self,
        process_id: str,
        cols: int,
        rows: int,
        shell: TerminalShellPreference,
        output_handler: TerminalOutputHandler,
        exit_handler: TerminalExitHandler,
    ) -> None: ...

    def write_terminal(self, process_id: str, data_base64: str) -> None: ...

    def resize_terminal(self, process_id: str, cols: int, rows: int) -> None: ...

    def stop_terminal(self, process_id: str) -> None: ...

    def close(self) -> None: ...


class CodingEngine(Protocol):
    id: str

    def capabilities(self) -> EngineCapabilities: ...

    def open_session(
        self,
        *,
        cowork_root: str,
        workspace: str,
        config: EngineSessionConfig,
        credentials: EngineCredentials,
        existing_session_id: str | None,
        approval_handler: ApprovalHandler,
    ) -> EngineSession: ...

    def discover_models(self, credentials: EngineCredentials) -> list[str]: ...
