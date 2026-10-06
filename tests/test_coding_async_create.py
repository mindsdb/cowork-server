"""A new local task returns at once and prepares its workspace in the background."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
from coding_service_fakes import (
    CREDS,
    FakeEngine,
    repository,
    service_with,
    wait_for_status,
)

from cowork.coding import project_workspaces
from cowork.coding.contracts import (
    EventType,
    InputReference,
    SessionCreateRequest,
    SessionStatus,
)
from cowork.coding.control_models import RunStatus
from cowork.coding.project_models import (
    ProjectCommand,
    ProjectCreateRequest,
    ProjectFolder,
)
from cowork.coding.repository_setup_models import TaskRepositorySetup
from cowork.coding.service import CodingService
from cowork.coding.workspace import WorkspaceError


def hold_preparation(monkeypatch: pytest.MonkeyPatch, service: CodingService) -> tuple[threading.Event, threading.Event]:
    """Pause workspace preparation until the test releases it."""
    started, release = threading.Event(), threading.Event()
    prepare = service.session_factory._prepare_local_session

    def held(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(service.session_factory, "_prepare_local_session", held)
    return started, release


def create(service: CodingService, repo: Path):
    return service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build the feature"), CREDS, "fake", "fake-model"
    )


def test_a_new_task_is_returned_before_its_workspace_is_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    started, release = hold_preparation(monkeypatch, service)

    created = create(service, repo)

    assert started.wait(timeout=3)
    assert created.status == SessionStatus.running
    assert created.run_status == RunStatus.preparing.value
    assert created.workspace_path == ""
    events = service.events(created.id).items
    assert [(event.type, event.title) for event in events] == [
        (EventType.user_message, "You"),
        (EventType.session, "Preparing task workspace"),
    ]
    for operation in (service.git_state, service.diff, service.terminals):
        with pytest.raises(RuntimeError, match="still being prepared"):
            operation(created.id)
    with pytest.raises(RuntimeError, match="still being prepared"):
        service.workspace_files(created.id, "README")
    with pytest.raises(RuntimeError, match="already has a running turn"):
        service.submit_turn(created.id, "And another thing", CREDS)
    with pytest.raises(RuntimeError, match="once the task workspace is ready"):
        service.steer(created.id, "Look at this", [InputReference(name="README.md", path=str(repo / "README.md"))])
    # A plain steer waits for the first turn.
    service.steer(created.id, "Also update the docs")

    release.set()
    done = wait_for_status(service, created.id, SessionStatus.completed)

    assert done.workspace_path
    assert Path(done.workspace_path, "README.md").exists()
    assert engine.prompts == ["Build the feature"]
    assert [prompt for _, prompt in engine.steers] == ["Also update the docs"]
    titles = [event.title for event in service.events(created.id).items]
    # The prompt is shown once, when the task is created.
    assert titles.count("You") == 1
    assert titles.index("Task workspace ready") < titles.index("Starting agent")


def test_stopping_a_task_while_it_prepares_never_starts_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    started, release = hold_preparation(monkeypatch, service)
    created = create(service, repo)
    assert started.wait(timeout=3)

    service.cancel(created.id)
    assert service.get_session(created.id).run_status == RunStatus.preparing.value
    release.set()
    stopped = wait_for_status(service, created.id, SessionStatus.cancelled)

    assert engine.prompts == []
    assert stopped.run_status == RunStatus.cancelled.value
    assert stopped.workspace_path
    assert "Task stopped" in [event.title for event in service.events(created.id).items]


def test_a_task_whose_workspace_cannot_be_prepared_stays_visible_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)

    def fail(*_args, **_kwargs):
        raise RuntimeError("disk is full")

    monkeypatch.setattr(service.session_factory.workspaces, "prepare", fail)

    created = create(service, repo)
    failed = wait_for_status(service, created.id, SessionStatus.failed)

    assert engine.prompts == []
    assert failed.run_status == RunStatus.failed.value
    assert "disk is full" in (failed.last_error or "")
    error = service.events(created.id).items[-1]
    assert (error.type, error.title) == (EventType.error, "Task did not start")
    with pytest.raises(RuntimeError, match="still being prepared"):
        service.git_state(created.id)
    with pytest.raises(RuntimeError, match="still being prepared"):
        service.fork_session(created.id, CREDS)
    # A follow-up must never open the agent without a task workspace.
    with pytest.raises(RuntimeError, match="did not start"):
        service.submit_turn(created.id, "Try again", CREDS)
    assert engine.prompts == []


def test_a_task_left_preparing_by_a_stopped_app_fails_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    first = service_with(tmp_path, FakeEngine())
    started, release = hold_preparation(monkeypatch, first)
    created = create(first, repo)
    assert started.wait(timeout=3)

    # A second service over the same root stands in for the next app launch;
    # the first process's preparation thread never finishes.
    restarted = service_with(tmp_path, FakeEngine())
    recovered = restarted.get_session(created.id)

    assert recovered.status == SessionStatus.failed
    assert recovered.run_status == RunStatus.failed.value
    assert "stopped before the task workspace was ready" in (recovered.last_error or "")
    # The orphaned preparation then finds its Run already failed.
    orphan = first._running[created.id].thread
    release.set()
    orphan.join(timeout=5)
    assert not orphan.is_alive()


def test_invalid_attachments_are_rejected_before_the_task_is_created(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
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
    assert service.list_sessions(True).items == []


@pytest.mark.parametrize(
    ("folder", "allow_direct_folder", "message"),
    [
        ("missing", True, "Choose an existing local folder"),
        ("plain", False, "Local folder isolation was not enabled"),
    ],
)
def test_a_folder_preparation_would_refuse_is_rejected_before_the_task_is_created(
    folder: str, allow_direct_folder: bool, message: str, tmp_path: Path
) -> None:
    (tmp_path / "plain").mkdir()
    service = service_with(tmp_path, FakeEngine())
    with pytest.raises(WorkspaceError, match=message):
        service.create_session(
            SessionCreateRequest(path=str(tmp_path / folder), prompt="Go", allow_direct_folder=allow_direct_folder),
            CREDS,
            "fake",
            "fake-model",
        )
    assert service.list_sessions(True).items == []


def test_a_command_sent_as_the_first_prompt_is_shown_once(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt="/status"), CREDS, "fake", "fake-model")
    wait_for_status(service, created.id, SessionStatus.ready)

    titles = [event.title for event in service.events(created.id).items]
    assert titles.count("You") == 1
    assert "Task status" in titles


@pytest.mark.parametrize("send", ["steer", "queue_turn"])
def test_a_follow_up_behind_a_command_first_prompt_still_runs(
    send: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    started, release = hold_preparation(monkeypatch, service)
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt="/status"), CREDS, "fake", "fake-model")
    assert started.wait(timeout=3)

    getattr(service, send)(created.id, "Now build the feature")
    if send == "steer":
        service.steer(created.id, "And update the docs")
    release.set()
    wait_for_status(service, created.id, SessionStatus.completed)

    # /status starts no turn, so the first follow-up becomes the turn.
    assert engine.prompts == ["Now build the feature"]
    if send == "steer":
        assert [prompt for _, prompt in engine.steers] == ["And update the docs"]
    assert service.get_session(created.id).queued_instructions == []


def test_a_follow_up_queued_while_preparing_runs_after_the_first_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    started, release = hold_preparation(monkeypatch, service)
    created = create(service, repo)
    assert started.wait(timeout=3)

    service.queue_turn(created.id, "Then add tests")
    with pytest.raises(RuntimeError, match="once the task workspace is ready"):
        service.queue_turn(created.id, "Read this", [InputReference(name="README.md", path=str(repo / "README.md"))])
    release.set()
    wait_for_status(service, created.id, SessionStatus.completed)
    deadline = time.monotonic() + 3
    while len(engine.prompts) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert engine.prompts == ["Build the feature", "Then add tests"]
    assert service.get_session(created.id).queued_instructions == []
    # Events emitted while preparing never try to move the Run past preparing.
    assert "Could not synchronize Task Run state" not in caplog.text


def test_setup_commands_report_progress_while_they_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Setup",
            folders=[ProjectFolder(
                id="app",
                name="App",
                path=str(repo),
                commands=[ProjectCommand(id="install", label="Install dependencies", argv=["true"], phase="setup")],
            )],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    running, finish = threading.Event(), threading.Event()
    run_one = project_workspaces.ProjectCommandRunner._run_one

    def slow(command, workspace, environment):
        running.set()
        assert finish.wait(timeout=5)
        return run_one(command, workspace, environment)

    monkeypatch.setattr(project_workspaces.ProjectCommandRunner, "_run_one", staticmethod(slow))
    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Build it"), CREDS, "fake", "fake-model"
    )
    assert running.wait(timeout=3)

    live = [event for event in service.events(created.id).items if event.title == "Install dependencies"]
    assert [(event.type, event.phase) for event in live] == [(EventType.command, "started")]
    finish.set()
    wait_for_status(service, created.id, SessionStatus.completed)
    setup = [event for event in service.events(created.id).items if event.title == "Install dependencies"]
    assert [event.phase for event in setup] == ["started", "completed"]
    assert setup[0].item_id == setup[1].item_id


def test_deleting_a_task_while_it_prepares_removes_it_and_its_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    started, release = hold_preparation(monkeypatch, service)
    created = create(service, repo)
    assert started.wait(timeout=3)
    thread = service._running[created.id].thread

    service.delete_session(created.id)

    with pytest.raises(KeyError):
        service.get_session(created.id)
    assert service.list_sessions(True).items == []
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert engine.prompts == []
    # The worktree preparation made after the delete was released again.
    assert git_output(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_runtime_settings_wait_for_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    started, release = hold_preparation(monkeypatch, service)
    created = create(service, repo)
    assert started.wait(timeout=3)

    for operation in (service.extension_inventory, service.platform_status):
        with pytest.raises(RuntimeError, match="still being prepared"):
            operation(created.id, CREDS)
    release.set()
    wait_for_status(service, created.id, SessionStatus.completed)


def test_deleting_a_task_before_its_first_turn_rolls_back_its_new_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Branch",
            folders=[ProjectFolder(id="app", name="App", path=str(repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    prepared, handoff = threading.Event(), threading.Event()
    complete = service.session_factory.complete

    def held(*args, **kwargs):
        # Hold the task after preparation finishes, before its first turn.
        session = complete(*args, **kwargs)
        prepared.set()
        assert handoff.wait(timeout=5)
        return session

    monkeypatch.setattr(service.session_factory, "complete", held)
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            prompt="Build it",
            repository_setup=TaskRepositorySetup(branch="feat/aborted"),
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    assert prepared.wait(timeout=3)
    assert "feat/aborted" in git_output(repo, "branch", "--list")
    thread = service._running[created.id].thread

    service.delete_session(created.id)
    handoff.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert engine.prompts == []
    assert "feat/aborted" not in git_output(repo, "branch", "--list")
    assert git_output(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout
