from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from cowork.coding.contracts import (
    CodingEvent,
    EngineCapabilities,
    EngineCommand,
    EventType,
    InputReference,
    PermissionMode,
    SessionCreateRequest,
    SessionStatus,
    SessionUpdateRequest,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.service import CodingService
from cowork.coding.turns import EventBuffer, terminal_status

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

    def fork(self, workspace: str) -> str:
        self.engine.forked_workspaces.append(workspace)
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
        self.compactions = 0
        self.steers: list[tuple[str, str]] = []
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
                EngineCommand(name="permissions", label="Permissions", description="Permissions", action="client", client_action="controls"),
            ],
        )

    def open_session(self, *, existing_session_id: str | None, **_kwargs) -> FakeSession:
        self.existing_ids.append(existing_session_id)
        self.permission_modes.append(_kwargs["config"].permission_mode)
        self.configs.append(_kwargs["config"])
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


def test_event_buffer_preserves_terminal_phase_when_coalescing_deltas() -> None:
    emitted: list[CodingEvent] = []
    buffer = EventBuffer(emitted.append)
    buffer.add(CodingEvent(type=EventType.command, title="Run tests", phase="started", item_id="cmd-1"))
    buffer.add(CodingEvent(type=EventType.command, text="tests passed", phase="progress", item_id="cmd-1"))
    buffer.add(CodingEvent(type=EventType.command, title="Run tests", phase="completed", item_id="cmd-1"))
    buffer.flush()

    assert len(emitted) == 1
    assert emitted[0].phase == "completed"
    assert emitted[0].title == "Run tests"
    assert emitted[0].text == "tests passed"


def test_terminal_status_distinguishes_interruption_from_user_cancel() -> None:
    assert terminal_status("interrupted", cancel_requested=False) == SessionStatus.interrupted
    assert terminal_status("interrupted", cancel_requested=True) == SessionStatus.cancelled


def test_completed_task_persists_events_and_reuses_live_engine_runtime(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    service.submit_turn(created.id, "Second turn", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)

    assert engine.existing_ids == [None]
    assert engine.prompts == ["First turn", "Second turn"]
    assert engine.closed == 0
    events = service.events(created.id).items
    assert [event.text for event in events if event.type == EventType.user_message] == ["First turn", "Second turn"]
    assert service.get_session(created.id).event_count == len(events)


def test_failed_adapter_stream_closes_the_runtime_before_another_turn_can_reuse_it(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.events_error = True
    service = service_with(tmp_path, engine)

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Start work"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.failed)

    assert engine.closed == 1
    assert service.get_session(created.id).last_error == "adapter stream disconnected"


def test_direct_folder_session_reports_the_workspace_it_actually_uses(tmp_path: Path) -> None:
    folder = tmp_path / "plain-folder"
    folder.mkdir()
    service = service_with(tmp_path, FakeEngine())

    created = service.create_session(
        SessionCreateRequest(
            path=str(folder),
            prompt="Use the folder",
            allow_direct_folder=True,
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    ready = service.events(created.id).items[0]
    assert ready.title == "Task workspace ready"
    assert ready.text == "Using the selected local folder for this task."
    assert ready.data["workspaceKind"] == "direct_folder"


def test_git_mutation_cannot_race_a_new_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    entered_mutation = threading.Event()
    release_mutation = threading.Event()
    original = service.workspaces.create_branch

    def blocking_branch(workspace_path: str, name: str):
        entered_mutation.set()
        assert release_mutation.wait(timeout=1)
        return original(workspace_path, name)

    monkeypatch.setattr(service.workspaces, "create_branch", blocking_branch)
    branch_thread = threading.Thread(target=lambda: service.create_branch(created.id, "review-race"))
    branch_thread.start()
    assert entered_mutation.wait(timeout=1)

    submit_thread = threading.Thread(target=lambda: service.submit_turn(created.id, "Second turn", CREDS))
    submit_thread.start()
    time.sleep(0.05)
    assert submit_thread.is_alive()
    assert engine.prompts == ["First turn"]

    release_mutation.set()
    branch_thread.join(timeout=1)
    submit_thread.join(timeout=1)
    wait_for_status(service, created.id, SessionStatus.completed)
    assert engine.prompts == ["First turn", "Second turn"]


def test_permission_mode_persists_and_reaches_engine(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)

    created = service.create_session(
        SessionCreateRequest(
            path=str(repo),
            prompt="Work within the repository",
            permission_mode=PermissionMode.workspace,
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    assert service.get_session(created.id).permission_mode == PermissionMode.workspace
    assert engine.permission_modes == [PermissionMode.workspace]


def test_cancel_during_engine_start_is_not_lost(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_open=True, block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.opened.wait(timeout=1)

    service.cancel(created.id)
    engine.release_open.set()
    wait_for_status(service, created.id, SessionStatus.cancelled)

    assert engine.cancels == ["turn-1"]
    assert engine.closed == 0
    assert service.get_session(created.id).last_error is None
    events = service.events(created.id).items
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert service.get_session(created.id).event_count == len(events)


def test_task_runtime_closes_on_delete_and_service_shutdown(tmp_path: Path) -> None:
    first_repo = repository(tmp_path)
    second_repo = tmp_path / "second-repo"
    second_repo.mkdir()
    git(second_repo, "init")
    git(second_repo, "config", "user.email", "cowork@example.invalid")
    git(second_repo, "config", "user.name", "Cowork Test")
    (second_repo / "README.md").write_text("second\n", encoding="utf-8")
    git(second_repo, "add", ".")
    git(second_repo, "commit", "-m", "base")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)

    first = service.create_session(
        SessionCreateRequest(path=str(first_repo), prompt="First task"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, first.id, SessionStatus.completed)
    service.delete_session(first.id)
    assert engine.closed == 1

    # A closed adapter is replaced lazily for the next task and is reaped at
    # application shutdown even when its last turn is idle.
    engine.is_closed = False
    second = service.create_session(
        SessionCreateRequest(path=str(second_repo), prompt="Second task"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, second.id, SessionStatus.completed)
    service.close_all()
    assert engine.closed == 2


def test_terminal_replays_output_and_shares_runtime_across_turns(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    started = service.start_terminal(created.id, CREDS, 120, 40)
    assert started.status.value == "running"
    assert engine.terminal_size == (120, 40)
    assert engine.terminal_output is not None
    engine.terminal_output("aGVsbG8=", "stdout", False)

    replay = service.terminal(created.id)
    assert [item.data_base64 for item in replay.items] == ["aGVsbG8="]
    assert service.terminal(created.id, after=replay.next_seq).items == []

    service.submit_turn(created.id, "Second turn", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    assert engine.existing_ids == [None]
    assert service.terminal(created.id).status.value == "running"

    service.write_terminal(created.id, "bHMK")
    service.resize_terminal(created.id, 90, 24)
    service.stop_terminal(created.id)
    assert engine.terminal_writes == [(engine.terminal_process_id, "bHMK")]
    assert engine.terminal_size == (90, 24)
    assert engine.terminal_stops == [engine.terminal_process_id]


def test_terminal_start_failure_is_not_left_running(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    engine.terminal_start_error = True

    with pytest.raises(RuntimeError, match="terminal secret-ish failure"):
        service.start_terminal(created.id, CREDS, 100, 30)
    failed = service.terminal(created.id)
    assert failed.status.value == "failed"
    assert failed.error == "Terminal process failed to start"


def test_goal_command_uses_engine_goal_lifecycle(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    service.submit_turn(created.id, "/goal Finish the migration and keep tests green", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)

    assert engine.goals == ["Finish the migration and keep tests green"]
    assert engine.prompts == ["First turn"]


def test_goal_command_supports_view_edit_pause_resume_and_clear(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    service.submit_turn(created.id, "/goal set Ship the migration", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    service.submit_turn(created.id, "/goal edit Ship the migration with Windows tests", CREDS)
    service.submit_turn(created.id, "/goal pause", CREDS)
    service.submit_turn(created.id, "/goal", CREDS)
    assert engine.goal == {
        "objective": "Ship the migration with Windows tests",
        "status": "paused",
    }

    service.submit_turn(created.id, "/goal resume", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    assert engine.goal_resumes == 1
    service.submit_turn(created.id, "/goal clear", CREDS)
    assert engine.goal is None
    assert engine.prompts == ["First turn"]


def test_status_and_compact_commands_do_not_start_model_turns(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.goal = {"objective": "Ship it", "status": "active", "tokensUsed": 42}
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    service.submit_turn(created.id, "/status", CREDS)
    service.submit_turn(created.id, "/compact", CREDS)

    assert engine.prompts == ["First turn"]
    assert engine.compactions == 1
    text = "\n".join(event.text for event in service.events(created.id).items)
    assert "Goal (active): Ship it" in text
    assert "compacting the task context" in text


def test_immediate_command_reservation_prevents_a_new_turn_from_racing_compaction(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    engine.block_compact = True

    compact_thread = threading.Thread(target=lambda: service.submit_turn(created.id, "/compact", CREDS))
    compact_thread.start()
    assert engine.compact_started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="being updated"):
        service.submit_turn(created.id, "Do not race compaction", CREDS)

    engine.release_compact.set()
    compact_thread.join(timeout=1)
    assert engine.compactions == 1
    assert engine.prompts == ["First turn"]


def test_unsupported_slash_command_is_explained_without_starting_a_turn(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    with pytest.raises(ValueError, match=r"/teleport is not supported by Fake"):
        service.submit_turn(created.id, "/teleport", CREDS)
    assert engine.prompts == ["First turn"]


def test_client_commands_and_unexpected_arguments_never_leak_into_model_turns(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    with pytest.raises(ValueError, match="opens a Code workspace control"):
        service.submit_turn(created.id, "/permissions please", CREDS)
    with pytest.raises(ValueError, match="does not accept an argument"):
        service.submit_turn(created.id, "/status verbose", CREDS)
    service.submit_turn(created.id, "/goal\tset Ship with Windows coverage", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)

    assert engine.goals == ["Ship with Windows coverage"]
    assert engine.prompts == ["First turn"]


def test_review_command_uses_codex_native_review_lifecycle(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    service.submit_turn(created.id, "/review", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)

    assert engine.reviews == 1
    assert engine.prompts == ["First turn"]


def test_task_controls_persist_and_restart_idle_runtime(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    updated = service.update_session_config(
        created.id,
        SessionUpdateRequest(
            model="other-model",
            permission_mode=PermissionMode.full_access,
            reasoning_effort="high",
            service_tier="priority",
            personality="friendly",
            network_access=False,
            web_search=True,
        ),
    )
    assert updated.model == "other-model"
    assert updated.permission_mode == PermissionMode.full_access
    assert updated.network_access is True
    assert updated.web_search is True
    assert engine.closed == 1

    engine.is_closed = False
    service.submit_turn(created.id, "Second turn", CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    config = engine.configs[-1]
    assert config.model == "other-model"
    assert config.reasoning_effort == "high"
    assert config.service_tier == "priority"
    assert config.personality == "friendly"
    assert config.network_access is True
    assert config.web_search is True


def test_steering_reaches_active_engine_and_is_recorded(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    service.steer(created.id, "Focus on tests")
    wait_for_steers(engine)
    assert engine.steers == [("turn-1", "Focus on tests")]
    assert service.events(created.id).items[-1].text == "Focus on tests"
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_steering_during_engine_start_is_queued(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_open=True, block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.opened.wait(timeout=1)

    service.steer(created.id, "Focus on tests")
    engine.release_open.set()
    assert engine.started.wait(timeout=1)

    wait_for_steers(engine)
    assert engine.steers == [("turn-1", "Focus on tests")]
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_native_turn_commands_must_be_queued_while_another_turn_runs(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match=r"Queue /review"):
        service.steer(created.id, "/review")
    with pytest.raises(RuntimeError, match=r"Queue /init"):
        service.steer(created.id, "/init")
    with pytest.raises(RuntimeError, match="Queue this goal command"):
        service.steer(created.id, "/goal set Ship it")

    assert engine.steers == []
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_attachmentless_command_is_rejected_before_it_enters_the_queue(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    attachment = repo / "notes.txt"
    attachment.write_text("context\n", encoding="utf-8")
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    with pytest.raises(ValueError, match=r"/review does not accept file attachments"):
        service.queue_turn(
            created.id,
            "/review",
            attachments=[InputReference(name="notes.txt", path=str(attachment))],
        )

    assert service.get_session(created.id).queued_instructions == []
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_queued_instruction_runs_as_the_next_turn_and_is_persisted(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    engine.block_until_release = True
    engine.started.clear()
    service.submit_turn(created.id, "Long second turn", CREDS)
    assert engine.started.wait(timeout=1)

    queued = service.queue_turn(created.id, "Run this after the current work")
    assert [item.prompt for item in queued.queued_instructions] == ["Run this after the current work"]
    assert service.events(created.id).items[-1].title == "Queued next"

    engine.release_events.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and len(engine.prompts) < 3:
        time.sleep(0.01)
    assert engine.prompts == ["First turn", "Long second turn", "Run this after the current work"]
    wait_for_status(service, created.id, SessionStatus.completed)
    assert service.get_session(created.id).queued_instructions == []


def test_queued_instruction_can_be_removed_before_it_runs(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    queued = service.queue_turn(created.id, "No longer needed")
    instruction_id = queued.queued_instructions[0].id
    updated = service.remove_queued_turn(created.id, instruction_id)
    assert updated.queued_instructions == []

    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)
    assert engine.prompts == ["Long turn"]


def test_turn_accepts_native_file_references_and_workspace_mentions(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    source = repo / "src"
    source.mkdir()
    referenced = source / "feature.py"
    referenced.write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "add feature")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(
            path=str(repo),
            prompt="Inspect the referenced file",
            attachments=[InputReference(name="src/feature.py", path=str(referenced))],
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    assert len(engine.attachments[0]) == 1
    assert engine.attachments[0][0].name == "src/feature.py"
    assert engine.attachments[0][0].path == str(Path(created.workspace_path, "src", "feature.py"))
    assert service.workspace_files(created.id, "feature") == [{
        "name": "src/feature.py",
        "path": str(Path(created.workspace_path, "src", "feature.py")),
        "kind": "mention",
    }]

    service.submit_turn(
        created.id,
        "Inspect the source folder",
        CREDS,
        [InputReference(name="src/", path=str(source), kind="mention")],
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    assert engine.attachments[1][0].name == "src/"
    assert engine.attachments[1][0].path == str(Path(created.workspace_path, "src"))
    assert service.workspace_files(created.id, "src")[0] == {
        "name": "src/",
        "path": str(Path(created.workspace_path, "src")),
        "kind": "mention",
    }


def test_missing_attachment_is_rejected_before_a_turn_starts(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)

    with pytest.raises(ValueError, match="Attached file is unavailable"):
        service.create_session(
            SessionCreateRequest(
                path=str(repo),
                prompt="Read it",
                attachments=[InputReference(name="missing.txt", path=str(repo / "missing.txt"))],
            ),
            CREDS,
            "fake",
            "fake-model",
        )
    assert engine.prompts == []


def test_task_can_be_renamed_archived_and_restored(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    renamed = service.rename_session(created.id, "  Clear   task name  ")
    assert renamed.title == "Clear task name"
    assert service.set_archived(created.id, True).archived is True
    assert service.list_sessions().items == []
    assert [item.id for item in service.list_sessions(include_archived=True).items] == [created.id]
    assert service.set_archived(created.id, False).archived is False
    assert [item.id for item in service.list_sessions().items] == [created.id]


def test_fork_copies_conversation_and_working_changes_to_an_independent_worktree(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    parent = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build the feature"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, parent.id, SessionStatus.completed)
    changed = Path(parent.workspace_path, "README.md")
    changed.write_text("forked work\n", encoding="utf-8")

    child = service.fork_session(parent.id, CREDS)

    assert child.id != parent.id
    assert child.workspace_path != parent.workspace_path
    assert Path(child.workspace_path, "README.md").read_text(encoding="utf-8") == "forked work\n"
    assert child.engine_session_id == "forked-engine-session-1"
    child_events = service.events(child.id).items
    assert any(event.text == "Build the feature" for event in child_events)
    assert child_events[-1].title == "Task forked"

    service.delete_session(parent.id)
    assert Path(child.workspace_path).is_dir()
