from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from cowork.coding.contracts import (
    CodingEvent,
    EngineCapabilities,
    EngineCommand,
    EventType,
    PermissionMode,
    SessionStatus,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.service import CodingService

CREDS = EngineCredentials(minds_url="https://example.invalid", minds_api_key="test-key")


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "cowork@example.invalid")
    git(repo, "config", "user.name", "Cowork Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def wait_for_status(service: CodingService, session_id: str, status: SessionStatus) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if service.get_session(session_id).status == status:
            return
        time.sleep(0.01)
    assert service.get_session(session_id).status == status


def wait_for_steers(engine: FakeEngine) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not engine.steers:
        time.sleep(0.01)


class FakeSession:
    def __init__(self, engine: FakeEngine, existing: str | None) -> None:
        self.engine = engine
        self._session_id = existing or "engine-session-1"
        self.cancelled = threading.Event()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_closed(self) -> bool:
        return self.engine.is_closed

    def start_turn(self, prompt: str, attachments=()) -> str:
        self.engine.prompts.append(prompt)
        self.engine.attachments.append(attachments)
        self.engine.started.set()
        return f"turn-{len(self.engine.prompts)}"

    def start_goal(self, objective: str) -> str:
        self.engine.goals.append(objective)
        self.engine.goal = {"objective": objective, "status": "active"}
        self.engine.started.set()
        return f"goal-{len(self.engine.goals)}"

    def resume_goal(self) -> str:
        self.engine.goal_resumes += 1
        if self.engine.goal is not None:
            self.engine.goal["status"] = "active"
        self.engine.started.set()
        return f"goal-resume-{self.engine.goal_resumes}"

    def update_goal(self, action: str, objective: str | None = None):
        if action == "clear":
            self.engine.goal = None
        elif action == "pause":
            if self.engine.goal is None:
                raise RuntimeError("There is no goal to update")
            self.engine.goal["status"] = "paused"
        elif action == "edit":
            if self.engine.goal is None:
                raise RuntimeError("There is no goal to update")
            self.engine.goal["objective"] = objective
        return self.engine.goal

    def start_review(self) -> str:
        self.engine.reviews += 1
        self.engine.started.set()
        return f"review-{self.engine.reviews}"

    def events(self, turn_id: str):
        if self.engine.events_error:
            raise RuntimeError("adapter stream disconnected")
        if self.engine.block_until_release:
            self.engine.release_events.wait(timeout=2)
        if self.engine.block_until_cancel:
            self.cancelled.wait(timeout=2)
            yield CodingEvent(type=EventType.session, data={"status": "interrupted"})
            return
        yield CodingEvent(type=EventType.agent_message, text="done", item_id="message-1")
        yield CodingEvent(type=EventType.session, data={"status": "completed"})

    def steer(self, turn_id: str, prompt: str, attachments=()) -> None:
        if self.engine.steer_error:
            raise RuntimeError("adapter rejected steer")
        self.engine.steers.append((turn_id, prompt))
        self.engine.steer_attachments.append(attachments)

    def cancel(self, turn_id: str) -> None:
        self.engine.cancels.append(turn_id)
        self.cancelled.set()

    def compact(self) -> None:
        if self.engine.block_compact:
            self.engine.compact_started.set()
            self.engine.release_compact.wait(timeout=2)
        self.engine.compactions += 1

    def goal_status(self):
        return self.engine.goal

    def fork(self, workspace: str, additional_dirs: tuple[str, ...] = ()) -> str:
        self.engine.forked_workspaces.append(workspace)
        self.engine.forked_additional_dirs.append(additional_dirs)
        return f"forked-engine-session-{len(self.engine.forked_workspaces)}"

    def start_terminal(self, process_id, cols, rows, output_handler, exit_handler) -> None:
        if self.engine.terminal_start_error:
            raise RuntimeError("terminal secret-ish failure")
        self.engine.terminal_process_id = process_id
        self.engine.terminal_size = (cols, rows)
        self.engine.terminal_output = output_handler
        self.engine.terminal_exit = exit_handler

    def write_terminal(self, process_id: str, data_base64: str) -> None:
        self.engine.terminal_writes.append((process_id, data_base64))

    def resize_terminal(self, process_id: str, cols: int, rows: int) -> None:
        self.engine.terminal_size = (cols, rows)

    def stop_terminal(self, process_id: str) -> None:
        self.engine.terminal_stops.append(process_id)

    def close(self) -> None:
        if not self.engine.is_closed:
            self.engine.is_closed = True
            self.engine.closed += 1
            self.engine.release_events.set()


class FakeEngine:
    id = "fake"

    def __init__(self, *, block_open: bool = False, block_until_cancel: bool = False) -> None:
        self.block_open = block_open
        self.block_until_cancel = block_until_cancel
        self.block_until_release = False
        self.events_error = False
        self.block_compact = False
        self.compact_started = threading.Event()
        self.release_compact = threading.Event()
        self.release_events = threading.Event()
        self.opened = threading.Event()
        self.release_open = threading.Event()
        self.started = threading.Event()
        self.existing_ids: list[str | None] = []
        self.permission_modes: list[PermissionMode] = []
        self.configs = []
        self.prompts: list[str] = []
        self.attachments = []
        self.goals: list[str] = []
        self.goal_resumes = 0
        self.reviews = 0
        self.goal: dict | None = None
        self.forked_workspaces: list[str] = []
        self.forked_additional_dirs: list[tuple[str, ...]] = []
        self.opened_workspaces: list[str] = []
        self.compactions = 0
        self.steers: list[tuple[str, str]] = []
        self.steer_error = False
        self.steer_attachments = []
        self.cancels: list[str] = []
        self.closed = 0
        self.is_closed = False
        self.terminal_process_id: str | None = None
        self.terminal_size: tuple[int, int] | None = None
        self.terminal_output = None
        self.terminal_exit = None
        self.terminal_writes: list[tuple[str, str]] = []
        self.terminal_stops: list[str] = []
        self.terminal_start_error = False

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            label="Fake",
            adapter_version="1",
            available=True,
            commands=[
                EngineCommand(name="goal", label="Goal", description="Goal", action="goal"),
                EngineCommand(name="review", label="Review", description="Review", action="turn"),
                EngineCommand(name="compact", label="Compact", description="Compact", action="compact"),
                EngineCommand(name="status", label="Status", description="Status", action="status"),
                EngineCommand(name="init", label="Init", description="Init", action="turn"),
                EngineCommand(
                    name="permissions",
                    label="Permissions",
                    description="Permissions",
                    action="client",
                    client_action="controls",
                ),
            ],
        )

    def open_session(self, *, existing_session_id: str | None, **kwargs) -> FakeSession:
        self.existing_ids.append(existing_session_id)
        self.permission_modes.append(kwargs["config"].permission_mode)
        self.configs.append(kwargs["config"])
        self.opened_workspaces.append(kwargs["workspace"])
        self.opened.set()
        if self.block_open:
            self.release_open.wait(timeout=2)
        return FakeSession(self, existing_session_id)

    def discover_models(self, _credentials: EngineCredentials) -> list[str]:
        return ["fake-model"]


def service_with(tmp_path: Path, engine: FakeEngine) -> CodingService:
    registry = CodingEngineRegistry()
    registry.register(engine)
    return CodingService(tmp_path / "coding", registry=registry)
