from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from cowork.coding.contracts import (
    CodingEvent,
    EngineCapabilities,
    ExtensionInventory,
    PermissionMode,
    RuntimePlatformStatus,
)

ApprovalHandler = Callable[[str, dict[str, Any] | None], dict[str, str]]
TerminalOutputHandler = Callable[[str, str, bool], None]
TerminalExitHandler = Callable[[int | None, str | None], None]


@dataclass(frozen=True)
class EngineCredentials:
    minds_url: str
    minds_api_key: str


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


@dataclass(frozen=True)
class EngineInputReference:
    name: str
    path: str
    kind: str


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
    ) -> None: ...

    def cancel(self, turn_id: str) -> None: ...

    def compact(self) -> None: ...

    def goal_status(self) -> dict[str, Any] | None: ...

    def extension_inventory(self) -> ExtensionInventory: ...

    def fork(self, workspace: str) -> str: ...

    def platform_status(self) -> RuntimePlatformStatus: ...

    def setup_windows_sandbox(self) -> RuntimePlatformStatus: ...

    def start_terminal(
        self,
        process_id: str,
        cols: int,
        rows: int,
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
