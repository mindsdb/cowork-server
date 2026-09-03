from __future__ import annotations

import contextlib
import itertools
import logging
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from coding_service_fakes import (
    CREDS,
    FakeEngine,
    git,
    repository,
    service_with,
    wait_for_status,
    wait_for_steers,
)

from cowork.coding.contracts import (
    CodingEvent,
    CodingSession,
    DeliveryRecord,
    EventType,
    InputReference,
    PendingApproval,
    PermissionMode,
    QueuedInstruction,
    QueueRunRequest,
    SessionCreateRequest,
    SessionStatus,
    SessionUpdateRequest,
    SourceContext,
    TaskCapabilities,
    WorkspaceKind,
    utc_now,
)
from cowork.api.v1.endpoints import coding
from cowork.api.v1.endpoints.guards import require_local
from cowork.coding.context import is_context_exhaustion_error
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_models import (
    CodeTask,
    ComputerCapabilities,
    RunStatus,
    RuntimeEvent,
    TaskCapability,
    TaskRun,
    WorkspaceStatus,
)
from cowork.db.scoped import get_tenant_scope
from cowork.db.session import get_session
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    DraftPullRequestSpec,
    ProjectCommand,
    ProjectCreateRequest,
    ProjectFolder,
    ProjectUpdateRequest,
    PublishRequest,
    PullRequestActionRequest,
    RepositoryResource,
)
from cowork.coding.project_service import CodeProjectService
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.store import CodingStore
from cowork.coding.turns import EventBuffer, finish_turn, mark_running, terminal_status
from cowork.coding.workspace import WorkspaceError


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


@pytest.mark.parametrize(
    "message",
    [
        "maximum context length exceeded",
        "context_length_exceeded",
        "too many tokens for this request",
        "the model context window limit reached",
        "ran out of room in the model's context window",
    ],
)
def test_context_exhaustion_errors_are_recognized(message: str) -> None:
    assert is_context_exhaustion_error(message)


def test_unrelated_engine_failures_are_not_treated_as_context_exhaustion() -> None:
    assert not is_context_exhaustion_error("adapter stream disconnected")


def test_task_creation_requires_exactly_one_project_or_folder() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SessionCreateRequest(prompt="Build it")
    with pytest.raises(ValueError, match="exactly one"):
        SessionCreateRequest(path="/folder", project_id="project", prompt="Build it")


def test_service_startup_survives_a_task_referencing_an_invalid_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "coding"
    CodingStore(root).save_session(CodingSession(
        id="task-with-invalid-project",
        title="Recover this task",
        engine_id="fake",
        engine_adapter_version="1",
        model="fake-model",
        status=SessionStatus.completed,
        project_id="invalid-project",
        source_path=str(workspace),
        workspace_path=str(workspace),
        workspace_kind=WorkspaceKind.local_copy,
    ))
    original_get = CodeProjectService.get

    def invalid_project(self, project_id: str):
        if project_id == "invalid-project":
            raise ValueError("stored project no longer passes validation")
        return original_get(self, project_id)

    monkeypatch.setattr(CodeProjectService, "get", invalid_project)

    service = service_with(tmp_path, FakeEngine())

    assert service.get_session("task-with-invalid-project").project_id == "invalid-project"


def stored_local_session(tmp_path: Path, status: SessionStatus) -> CodingSession:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = CodingSession(
        id="stored-task",
        title="Stored task",
        engine_id="fake",
        engine_adapter_version="1",
        model="fake-model",
        status=status,
        source_path=str(workspace),
        workspace_path=str(workspace),
        workspace_kind=WorkspaceKind.local_copy,
    )
    CodingStore(tmp_path / "coding").save_session(session)
    return session


def test_finishing_a_turn_with_a_pending_approval_completes_the_run(tmp_path: Path) -> None:
    session = stored_local_session(tmp_path, SessionStatus.ready)
    service = service_with(tmp_path, FakeEngine())
    pending = PendingApproval(id="approval-1", method="command", kind="command", title="Run it", detail="ls", risk="low", scope="once")
    service._emit(session.id, CodingEvent(type=EventType.session, phase="started"), mark_running)
    service._emit(session.id, CodingEvent(type=EventType.approval, phase="pending"), lambda current: service._open_approval(current, pending))
    run_id = service.get_session(session.id).run_id
    assert service.control.store.get_run(run_id).status == RunStatus.awaiting_approval

    service._emit(session.id, CodingEvent(type=EventType.session, phase="completed"), lambda current: finish_turn(current, SessionStatus.completed))

    assert service.control.store.get_run(run_id).status == RunStatus.completed
    assert service.get_session(session.id).pending_approval is None


def test_invalid_run_transition_during_emit_is_logged_with_both_statuses(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session = stored_local_session(tmp_path, SessionStatus.completed)
    service = service_with(tmp_path, FakeEngine())

    with caplog.at_level(logging.ERROR, logger="cowork.coding.service"):
        service._emit(session.id, CodingEvent(type=EventType.session, phase="started"), mark_running)

    (record,) = [item for item in caplog.records if "synchronize" in item.getMessage()]
    assert record.levelno == logging.ERROR
    assert "from completed to running" in record.getMessage()
    assert service.control.store.get_run(service.get_session(session.id).run_id).status == RunStatus.completed


def test_emit_swallows_only_missing_runs_and_state_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = stored_local_session(tmp_path, SessionStatus.ready)
    service = service_with(tmp_path, FakeEngine())
    failures = iter([StateConflict("Coding session has not been linked to a Task Run"), ValueError("unexpected")])

    def failing_sync(_session) -> None:
        raise next(failures)

    monkeypatch.setattr(service.control, "sync_session", failing_sync)

    service._emit(session.id, CodingEvent(type=EventType.session, phase="started"), mark_running)
    with pytest.raises(ValueError, match="unexpected"):
        service._emit(session.id, CodingEvent(type=EventType.session, phase="progress"))


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


def test_remote_runtime_claims_a_portable_task_without_receiving_local_paths(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    code_project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=code_project.id,
            computer_id=remote.id,
            prompt="Build remotely",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )

    assert engine.prompts == []
    assert service.control.store.get_run(created.run_id or "").status == RunStatus.queued
    leased = service.control.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased
    task = service.control.store.get_task(run.task_id)
    runtime_project = service.control.runtime_project(code_project, task.resource_scope, remote.id)
    repository_resource = runtime_project.resources[0]
    assert isinstance(repository_resource, RepositoryResource)
    assert repository_resource.local_path is None
    assert repository_resource.source_url == "https://example.test/repo.git"

    def send(seq: int, status: str) -> None:
        service.accept_runtime_event(RuntimeEvent(
            run_id=run.id,
            computer_id=remote.id,
            lease_id=lease_id,
            epoch=run.epoch,
            seq=seq,
            kind="status",
            payload={"status": status},
        ))

    send(1, "ready")
    send(2, "running")
    service.accept_runtime_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=3,
        kind="event",
        payload={"event": {
            "type": "agent_message",
            "title": "Agent",
            "text": "Remote work is complete.",
            "phase": "completed",
        }},
    ))
    send(4, "completed")
    assert service.get_session(created.id).status == SessionStatus.completed
    assert any(event.text == "Remote work is complete." for event in service.events(created.id).items)

    continued = service.submit_turn(created.id, "One more change", CREDS)
    assert continued.run_id != run.id
    assert service.control.store.get_run(continued.run_id or "").status == RunStatus.queued


def test_compatibility_events_cannot_overwrite_a_remote_runtime_lifecycle(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=remote.id,
            prompt="Build remotely",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    assert created.task_capabilities.files is False
    assert created.task_capabilities.review is False  # Legacy runtimes fail closed until they advertise support.
    with pytest.raises(RuntimeError, match="Task files stay on the computer"):
        service.workspace_files(created.id)
    with pytest.raises(RuntimeError, match="Task files stay on the computer"):
        service.workspace_resources(created.id)
    with pytest.raises(RuntimeError, match="Task files stay on the computer"):
        service.workspace_entries(created.id, "repo")
    with pytest.raises(RuntimeError, match="Task files stay on the computer"):
        service.workspace_file(created.id, "repo", "README.md")
    with pytest.raises(RuntimeError, match="Task files stay on the computer"):
        service.workspace_search(created.id, "README")
    leased = service.control.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased
    service.accept_runtime_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=1,
        kind="status",
        payload={"status": "ready"},
    ))
    service.accept_runtime_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=2,
        kind="status",
        payload={"status": "running"},
    ))

    queued = service.queue_turn(created.id, "No longer needed")
    service.accept_runtime_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=3,
        kind="status",
        payload={"status": "ready"},
    ))
    session = service.store.load_session(created.id)
    session.status = SessionStatus.completed
    service.store.save_session(session)
    service.remove_queued_turn(created.id, queued.queued_instructions[0].id)

    canonical = service.control.store.get_run(run.id)
    assert canonical.status == RunStatus.ready
    assert canonical.lease_id == lease_id
    assert canonical.last_event_seq == 3


def test_remote_approval_text_is_redacted_before_reaching_the_ui() -> None:
    run = TaskRun(id="run", task_id="task", computer_id="remote", status=RunStatus.awaiting_approval)
    pending, event = RemoteExecutionCoordinator._coding_event(run, RuntimeEvent(
        run_id=run.id,
        computer_id=run.computer_id,
        lease_id="lease",
        epoch=1,
        seq=1,
        kind="approval",
        payload={
            "approvalId": "approval",
            "params": {
                "title": "Use token=secret-value",
                "command": "Authorization: Bearer secret-value",
                "cwd": "/tmp/password=secret-value",
            },
        },
    ))

    assert pending is not None
    assert pending.title == "Use token=[redacted]"
    assert pending.detail == "Authorization: Bearer [redacted]"
    assert pending.cwd == "/tmp/password=[redacted]"
    assert "secret-value" not in event.model_dump_json()


def test_remote_cancel_is_deduplicated_per_turn_not_per_runtime_epoch(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=remote.id,
            prompt="First turn",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    leased = service.control.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased
    initial = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    service.control.acknowledge_command(run.id, initial[0].id, remote.id, lease_id, run.epoch)

    def status(seq: int, value: str) -> None:
        service.accept_runtime_event(RuntimeEvent(
            run_id=run.id,
            computer_id=remote.id,
            lease_id=lease_id,
            epoch=run.epoch,
            seq=seq,
            kind="status",
            payload={"status": value},
        ))

    status(1, "ready")
    status(2, "running")
    service.cancel(created.id)
    service.cancel(created.id)
    first_cancel = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [command.kind for command in first_cancel] == ["cancel"]
    service.control.acknowledge_command(run.id, first_cancel[0].id, remote.id, lease_id, run.epoch)

    status(3, "ready")
    service.submit_turn(created.id, "Second turn", CREDS)
    next_start = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [command.kind for command in next_start] == ["start"]
    service.control.acknowledge_command(run.id, next_start[0].id, remote.id, lease_id, run.epoch)
    status(4, "running")
    service.cancel(created.id)
    second_cancel = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [command.kind for command in second_cancel] == ["cancel"]
    assert second_cancel[0].idempotency_key != first_cancel[0].idempotency_key


def test_active_remote_task_must_stop_before_deletion(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=remote.id,
            prompt="Keep working",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    run = service.control.store.get_run(created.run_id or "")
    service.control.set_run_status(run.id, RunStatus.preparing)
    service.control.set_run_status(run.id, RunStatus.ready)
    service.control.set_run_status(run.id, RunStatus.running)
    service.store.update_session(
        created.id,
        lambda current: setattr(current, "run_status", RunStatus.running.value),
    )

    with pytest.raises(RuntimeError, match="Wait for the remote agent"):
        service.delete_session(created.id)

    assert service.get_session(created.id).id == created.id
    assert service.control.store.get_run(run.id).task_id == created.task_id


def test_deleting_task_survives_a_moved_source_repository(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Inspect it"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    workspace = Path(service.get_session(created.id).workspace_path)
    repo.rename(tmp_path / "moved")

    service.delete_session(created.id)

    with pytest.raises(KeyError):
        service.get_session(created.id)
    with pytest.raises(KeyError):
        service.control.store.get_run(created.run_id or "")
    assert not workspace.exists()


def test_remote_runtime_reuses_one_run_for_follow_ups_and_persisted_queue(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
            task_capabilities=[TaskCapability.slash_commands],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=remote.id,
            prompt="Build remotely",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    leased = service.control.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased
    initial = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [item.kind for item in initial] == ["start"]
    service.control.acknowledge_command(run.id, initial[0].id, remote.id, lease_id, run.epoch)

    def event(seq: int, kind: str, payload: dict[str, object]) -> None:
        service.accept_runtime_event(RuntimeEvent(
            run_id=run.id,
            computer_id=remote.id,
            lease_id=lease_id,
            epoch=run.epoch,
            seq=seq,
            kind=kind,
            payload=payload,
        ))

    event(1, "status", {"status": "ready"})
    event(2, "status", {"status": "running"})
    service.steer(created.id, "/status")
    immediate = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [item.kind for item in immediate] == ["agent_command"]
    assert immediate[0].payload["command"] == "status"
    service.control.acknowledge_command(run.id, immediate[0].id, remote.id, lease_id, run.epoch)
    queued = service.queue_turn(created.id, "Run this after the current work")
    assert [item.prompt for item in queued.queued_instructions] == ["Run this after the current work"]
    event(3, "turn_completed", {"status": "completed"})

    resumed = service.get_session(created.id)
    assert resumed.run_id == run.id
    assert resumed.queued_instructions == []
    event_count = len(service.events(created.id).items)
    event(4, "checkpoint", {"waiting": "start", "workspaceReady": True})
    assert len(service.events(created.id).items) == event_count
    assert service.get_session(created.id).status == SessionStatus.completed
    commands = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert len(commands) == 1
    assert commands[0].kind == "start"
    assert commands[0].payload == {
        "prompt": "Run this after the current work",
        "engine_prompt": "Run this after the current work",
        "command": "",
        "goal_action": "",
        "goal_objective": None,
    }

    service.control.acknowledge_command(run.id, commands[0].id, remote.id, lease_id, run.epoch)
    event(5, "status", {"status": "running"})
    event(6, "turn_completed", {"status": "completed"})
    follow_up = service.submit_turn(created.id, "/review", CREDS)
    assert follow_up.run_id == run.id
    commands = service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert len(commands) == 1
    assert commands[0].payload == {
        "prompt": "/review",
        "engine_prompt": "/review",
        "command": "review",
        "goal_action": "",
        "goal_objective": None,
    }


def test_remote_run_state_is_projected_and_can_be_restored(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    registration = service.control.issue_registration_token()
    remote, _ = service.control.register_runtime(
        registration,
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            computer_id=remote.id,
            prompt="Build remotely",
            engine_id="fake",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    assert created.run_status == "queued"
    assert created.computer_name == "Build computer"
    assert created.computer_status == "online"

    leased = service.control.acquire_lease(remote.id)
    assert leased is not None
    run, _ = leased
    run.lease_expires_at = utc_now() - timedelta(seconds=1)
    service.control.store.save_run(run)
    service.expire_stale_state()
    interrupted = service.get_session(created.id)
    assert interrupted.run_status == "recovering"
    assert interrupted.last_error == "The computer stopped responding. The task can be resumed safely."

    restored = service.recover(created.id)
    assert restored.run_status == "recovering"
    assert restored.runtime_epoch == 3
    assert restored.computer_id == remote.id


def remote_task(tmp_path: Path, service, prompt: str = "Build remotely"):
    """Register a connected computer and queue one portable task on it."""
    repo = repository(tmp_path)
    remote, _ = service.control.register_runtime(
        service.control.issue_registration_token(),
        "Build computer",
        ComputerCapabilities(
            platform="linux",
            architecture="test",
            runtime_version="test-runtime",
            agent_engines=["fake"],
            shells=["bash"],
        ),
    )
    project = service.projects.create(ProjectCreateRequest(
        name="Portable project",
        resources=[RepositoryResource(
            id="repo",
            name="Repo",
            source_url="https://example.test/repo.git",
            local_path=str(repo),
        )],
    ))
    created = service.create_session(
        SessionCreateRequest(project_id=project.id, computer_id=remote.id, prompt=prompt, engine_id="fake"),
        CREDS,
        "fake",
        "fake-model",
    )
    return created, remote


def runtime_event(service, run: TaskRun, lease_id: str, seq: int, kind: str, payload: dict[str, object]) -> TaskRun:
    return service.accept_runtime_event(RuntimeEvent(
        run_id=run.id,
        computer_id=run.computer_id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=seq,
        kind=kind,
        payload=payload,
    ))


def test_cancel_during_remote_approval_completes_the_turn(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    runtime_event(service, run, lease_id, 3, "status", {"status": "awaiting_approval"})
    runtime_event(service, run, lease_id, 4, "approval", {"approvalId": "approval-1", "method": "command"})

    service.cancel(created.id)
    runtime_event(service, run, lease_id, 5, "turn_completed", {"status": "cancelled"})

    assert {command.kind for command in service.control.store.list_commands(run.id)} == {"start", "cancel"}
    assert service.control.store.get_run(run.id).status == RunStatus.ready
    session = service.get_session(created.id)
    assert session.status == SessionStatus.cancelled
    assert session.pending_approval is None


def test_release_workspace_timeout_reports_the_offline_computer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})

    def never_answers(_run_id: str, _command_id: str, timeout: float = 20.0):
        raise RuntimeError("The selected computer did not answer in time")

    monkeypatch.setattr(service.control, "wait_for_command", never_answers)
    with pytest.raises(RuntimeError, match="did not release this task workspace; retry when it is online"):
        service.remote.release_workspace(service.get_session(created.id))

    assert service.control.store.get_run(run.id).status == RunStatus.ready
    assert {command.kind for command in service.control.store.list_commands(run.id)} == {"start", "release"}


def spy_on_control_store_writes(store, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    writes: list[str] = []

    def recording(name: str, original):
        def spy(*args, **kwargs):
            writes.append(name)
            return original(*args, **kwargs)

        return spy

    for name in dir(store):
        if name.startswith(("save_", "update_", "claim_", "create_", "delete_", "consume_", "prune")):
            monkeypatch.setattr(store, name, recording(name, getattr(store, name)))
    return writes


def test_listing_sessions_performs_no_control_plane_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = service_with(tmp_path, FakeEngine(), expiry_interval_seconds=3600)
    created, remote = remote_task(tmp_path, service)
    run, _ = service.control.acquire_lease(remote.id)
    service.control.store.update_run(run.id, lambda current: setattr(current, "lease_expires_at", utc_now() - timedelta(seconds=1)))
    stale = service.control.store.get_computer(remote.id)
    stale.last_seen_at = utc_now() - timedelta(minutes=5)
    service.control.store.save_computer(stale)
    template = service.store.load_session(created.id)
    for index in range(49):
        task = CodeTask(id=f"task-{index}", title=f"Task {index}")
        clone = TaskRun(id=f"run-{index}", task_id=task.id, computer_id=service.control.local_computer.id)
        service.control.store.create_task_run(task, clone, [])
        service.store.save_session(template.model_copy(update={"id": task.id, "task_id": task.id, "run_id": clone.id}))
    writes = spy_on_control_store_writes(service.control.store, monkeypatch)

    listed = service.list_sessions()

    assert len(listed.items) == 50
    assert writes == []
    assert service.control.store.get_run(run.id).status == RunStatus.preparing

    service.expire_stale_state()
    assert service.control.store.get_run(run.id).status == RunStatus.recovering
    assert service.control.store.get_computer(remote.id).status.value == "offline"


def test_expiry_runs_on_a_timer_and_stops_with_the_service(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine(), expiry_interval_seconds=0.01)
    _, remote = remote_task(tmp_path, service)
    run, _ = service.control.acquire_lease(remote.id)
    service.control.store.update_run(run.id, lambda current: setattr(current, "lease_expires_at", utc_now() - timedelta(seconds=1)))

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and service.control.store.get_run(run.id).status != RunStatus.recovering:
        time.sleep(0.01)
    assert service.control.store.get_run(run.id).status == RunStatus.recovering

    service.close_all()
    service._expiry_thread.join(timeout=1)
    assert not service._expiry_thread.is_alive()


def test_close_all_leaves_no_expiry_thread_behind(tmp_path: Path) -> None:
    before = {thread.ident for thread in threading.enumerate() if thread.name == "coding-expiry"}
    service = service_with(tmp_path, FakeEngine())
    assert service._expiry_thread.is_alive()

    service.close_all()

    survivors = [thread for thread in threading.enumerate() if thread.name == "coding-expiry" and thread.ident not in before]
    assert survivors == []


def acknowledge(service, run: TaskRun, lease_id: str, command, error: str | None = None):
    return service.acknowledge_runtime_command(
        run.id, command.id, run.computer_id, lease_id, run.epoch, None, error,
    )


def command_results(service, session_id: str) -> list[CodingEvent]:
    return [item for item in service.events(session_id).items if item.type == EventType.command_result]


def test_concurrent_duplicate_acks_surface_one_command_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    service.steer(created.id, "Change direction")
    (steer,) = [item for item in service.control.claim_commands(run.id, remote.id, lease_id, run.epoch) if item.kind == "steer"]
    store = service.control.store
    original_get_command = store.get_command
    both_read_before_either_acks = threading.Barrier(2, timeout=0.5)
    reads = itertools.count()

    def get_command(command_id: str):
        if next(reads) < 2:
            with contextlib.suppress(threading.BrokenBarrierError):
                both_read_before_either_acks.wait()
        return original_get_command(command_id)

    monkeypatch.setattr(store, "get_command", get_command)
    acks = [threading.Thread(target=acknowledge, args=(service, run, lease_id, steer, "rejected")) for _ in range(2)]
    for thread in acks:
        thread.start()
    for thread in acks:
        thread.join(timeout=5)

    assert service.control.store.get_command(steer.id).acked_at is not None
    assert len(command_results(service, created.id)) == 1


def test_rejected_remote_command_is_surfaced_once_as_a_redacted_command_result(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    service.steer(created.id, "Change direction")
    (steer,) = [item for item in service.control.claim_commands(run.id, remote.id, lease_id, run.epoch) if item.kind == "steer"]

    acknowledge(service, run, lease_id, steer, error="adapter rejected steer: Authorization: Bearer secret-token")
    acknowledge(service, run, lease_id, steer, error="adapter rejected steer: Authorization: Bearer secret-token")

    (result,) = command_results(service, created.id)
    assert result.title == "Steer rejected"
    assert result.phase == "failed"
    assert result.data == {"command": "steer", "commandId": steer.id}
    assert "secret-token" not in result.text
    assert result.text == "adapter rejected steer: Authorization: Bearer [redacted]"
    assert result.turn_id is None


def test_successful_turn_commands_report_completion_but_operations_stay_silent(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    service.cancel(created.id)
    operation = service.control.queue_command(run.id, "operation", {"operation": "diff"}, "operation-diff-1")
    claimed = {item.kind: item for item in service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)}

    acknowledge(service, run, lease_id, claimed["start"])
    acknowledge(service, run, lease_id, claimed["cancel"])
    acknowledge(service, run, lease_id, operation, error="The task execution workspace is no longer available")

    (result,) = command_results(service, created.id)
    assert (result.title, result.phase, result.text) == ("Cancel accepted", "completed", "")
    assert result.data == {"command": "cancel", "commandId": claimed["cancel"].id}


def test_failed_queued_start_is_reported_without_requeuing_the_instruction(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    queued = service.queue_turn(created.id, "Run this after the current work")
    instruction_id = queued.queued_instructions[0].id
    runtime_event(service, run, lease_id, 3, "turn_completed", {"status": "completed"})
    assert service.get_session(created.id).queued_instructions == []
    (start,) = [
        item for item in service.control.claim_commands(run.id, remote.id, lease_id, run.epoch)
        if item.idempotency_key == f"queued-turn-{instruction_id}"
    ]

    acknowledge(service, run, lease_id, start, error="The engine could not start the turn")

    (result,) = command_results(service, created.id)
    assert (result.title, result.phase) == ("Turn rejected", "failed")
    assert result.data["commandId"] == start.id
    assert service.get_session(created.id).queued_instructions == []


def test_queue_run_accepts_an_optional_body_over_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    queued = service.queue_turn(created.id, "Run this after the current work")
    instruction_id = queued.queued_instructions[0].id
    runtime_event(service, run, lease_id, 3, "turn_completed", {"status": "cancelled"})
    monkeypatch.setattr(coding, "_service", lambda: service)
    monkeypatch.setattr(coding, "_settings", lambda _session, _scope: None)
    monkeypatch.setattr(coding, "_credentials", lambda _settings: CREDS)
    app = FastAPI()
    app.include_router(coding.router, prefix="/api/v1/coding")
    app.dependency_overrides[require_local] = lambda: None
    app.dependency_overrides[get_session] = lambda: None
    app.dependency_overrides[get_tenant_scope] = lambda: None

    with TestClient(app) as client:
        stale = client.post(f"/api/v1/coding/sessions/{created.id}/queue/run", json={"instruction_id": "not-the-head"})
        started = client.post(f"/api/v1/coding/sessions/{created.id}/queue/run", json={"instruction_id": instruction_id})
        bodiless = client.post(f"/api/v1/coding/sessions/{created.id}/queue/run")

    assert stale.status_code == 409
    assert started.status_code == 200
    assert CodingSession.model_validate(started.json()).queued_instructions == []
    assert bodiless.status_code == 200
    assert len([item for item in service.control.store.list_commands(run.id) if item.kind == "start"]) == 2


def test_queue_run_with_an_instruction_id_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "running"})
    queued = service.queue_turn(created.id, "Run this after the current work")
    instruction_id = queued.queued_instructions[0].id
    runtime_event(service, run, lease_id, 3, "turn_completed", {"status": "cancelled"})
    assert [item.id for item in service.get_session(created.id).queued_instructions] == [instruction_id]
    monkeypatch.setattr(coding, "_service", lambda: service)
    monkeypatch.setattr(coding, "_settings", lambda _session, _scope: None)
    monkeypatch.setattr(coding, "_credentials", lambda _settings: CREDS)

    def run_queued(body: QueueRunRequest | None) -> CodingSession:
        return coding.run_next_queued(created.id, session=None, scope=None, body=body)

    with pytest.raises(HTTPException) as stale:
        run_queued(QueueRunRequest(instruction_id="not-the-head"))
    assert stale.value.status_code == 409
    assert [item.id for item in service.get_session(created.id).queued_instructions] == [instruction_id]

    started = run_queued(QueueRunRequest(instruction_id=instruction_id))
    assert started.queued_instructions == []
    starts = [item for item in service.control.store.list_commands(run.id) if item.idempotency_key == f"queued-turn-{instruction_id}"]
    assert len(starts) == 1

    with pytest.raises(HTTPException) as repeated:
        run_queued(QueueRunRequest(instruction_id=instruction_id))
    assert repeated.value.status_code == 409
    assert service.get_session(created.id).queued_instructions == []
    assert len([item for item in service.control.store.list_commands(run.id) if item.kind == "start"]) == 2
    assert run_queued(None).queued_instructions == []


def test_continuing_a_task_persists_the_durable_session_not_the_projected_view(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 2, "status", {"status": "completed"})
    projected = service.get_session(created.id)
    assert (projected.run_status, projected.computer_name) == ("completed", "Build computer")

    continued = service.submit_turn(created.id, "One more change", CREDS)

    durable = service.store.load_session(created.id)
    assert durable.run_id == continued.run_id != run.id
    assert durable.status == SessionStatus.ready
    assert (durable.run_status, durable.computer_name, durable.computer_status) == (None, None, None)
    assert durable.computer_is_local is True
    assert durable.task_capabilities == TaskCapabilities()
    assert continued.run_status == "queued"


def test_follow_up_after_a_released_remote_run_reprovisions_its_workspaces(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    created, remote = remote_task(tmp_path, service)
    run, lease_id = service.control.acquire_lease(remote.id)
    runtime_event(service, run, lease_id, 1, "workspace", {"items": [{
        "folder_id": "repo",
        "folder_name": "Repo",
        "source_path": "/remote/source",
        "workspace_path": "/remote/worktrees/repo",
        "workspace_kind": "git_worktree",
        "base_revision": "abc123",
    }]})
    runtime_event(service, run, lease_id, 2, "status", {"status": "ready"})
    runtime_event(service, run, lease_id, 3, "status", {"status": "completed"})
    released = service.control.store.list_workspaces(run.id)
    assert [(item.status, item.path) for item in released] == [(WorkspaceStatus.released, "")]

    continued = service.submit_turn(created.id, "One more change", CREDS)
    lease = service.remote.acquire_lease(remote.id)

    assert lease is not None
    assert lease.run.id == continued.run_id != run.id
    assert lease.run.workspace_resume_mode == "prepare"
    assert [(item.resource_id, item.status, item.path) for item in lease.workspaces] == [
        ("repo", WorkspaceStatus.pending, ""),
    ]


def test_interrupted_local_task_resumes_on_a_follow_up_run(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.block_until_release = True
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long-running turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)
    service.prepare_shutdown()
    engine.release_events.set()
    wait_for_status(service, created.id, SessionStatus.interrupted)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and created.id in service._running:
        time.sleep(0.01)
    interrupted_run = service.control.store.get_run(created.run_id or "")
    assert interrupted_run.status == RunStatus.interrupted

    restarted = service_with(tmp_path, FakeEngine())
    resumed = restarted.submit_turn(created.id, "Pick up where you left off", CREDS)
    wait_for_status(restarted, created.id, SessionStatus.completed)

    assert resumed.run_id != interrupted_run.id
    finished = restarted.get_session(created.id)
    assert finished.run_status == "completed"
    assert finished.runtime_epoch == interrupted_run.epoch + 1
    assert restarted.control.store.get_run(interrupted_run.id).status == RunStatus.interrupted


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


def test_credit_exhaustion_is_reported_with_a_stable_code_and_the_technical_detail(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.events_error = True
    engine.events_error_message = (
        '{"error": {"message": "Your wallet has no balance to cover the model \'gpt\'.", '
        '"type": "invalid_request_error", "code": "insufficient_credits", "upstream_status": 402}}'
    )
    service = service_with(tmp_path, engine)

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Start work"), CREDS, "fake", "gpt"
    )
    wait_for_status(service, created.id, SessionStatus.failed)

    failed = service.get_session(created.id)
    error = [event for event in service.events(created.id).items if event.type == EventType.error][-1]
    assert failed.last_error == "This model needs credits. Add credits or choose another model."
    assert error.text == failed.last_error
    assert error.data == {
        "code": "insufficient_credits",
        "detail": engine.events_error_message,
        "model": "gpt",
    }
    assert engine.closed == 1


def test_a_turn_codex_reports_as_failed_is_classified_at_its_terminal_event(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.turn_failure_message = (
        '{"error": {"message": "Invalid API key", "type": "invalid_request_error", '
        '"code": "model_authentication_failed", "upstream_status": 401}}'
    )
    service = service_with(tmp_path, engine)

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Start work"), CREDS, "fake", "gpt"
    )
    wait_for_status(service, created.id, SessionStatus.failed)

    failed = service.get_session(created.id)
    events = service.events(created.id).items
    terminal = [event for event in events if event.type == EventType.session and event.data.get("status") == "failed"][-1]
    raw = [event for event in events if event.type == EventType.error][-1]
    assert failed.last_error == (
        "Your sign-in does not match this server. Sign in again, or switch back to the environment you signed into."
    )
    assert terminal.text == failed.last_error
    assert terminal.data == {
        "status": "failed",
        "code": "model_authentication_failed",
        "detail": engine.turn_failure_message,
        "model": "gpt",
    }
    assert raw.text == engine.turn_failure_message
    assert raw.data == {}


def test_local_context_exhaustion_recovers_ready_with_a_fresh_agent_session(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.events_error = True
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Use all the context"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.failed)
    failed = service.get_session(created.id)
    original_workspace = failed.workspace_path
    assert failed.engine_session_id == "engine-session-1"
    context_error = (
        "Codex ran out of room in the model's context window. "
        "Start a new thread or clear earlier history before retrying."
    )
    service.store.update_session(
        created.id,
        lambda current: setattr(current, "last_error", "The coding agent stopped unexpectedly"),
    )
    run = service.control.store.get_run(created.run_id or "")
    run.last_error = context_error
    service.control.store.save_run(run)

    restored = service.recover(created.id)

    assert restored.status == SessionStatus.ready
    assert restored.run_status == RunStatus.ready.value
    assert restored.engine_session_id is None
    assert restored.last_error is None
    assert restored.workspace_path == original_workspace
    assert any(
        event.title == "Agent context refreshed"
        and "workspace" in event.text
        and "preserved" in event.text
        for event in service.events(created.id).items
    )


def test_non_git_session_reports_its_isolated_local_copy(tmp_path: Path) -> None:
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
    assert ready.text == "Created an isolated task workspace."
    assert ready.data["workspaceKind"] == "local_copy"
    assert created.workspace_path != str(folder.resolve())
    assert created.source_path == str(folder.resolve())


def test_repository_without_commits_can_start_a_coding_session(tmp_path: Path) -> None:
    repo = tmp_path / "unborn-repo"
    repo.mkdir()
    git(repo, "init")
    (repo / "README.md").write_text("uncommitted source\n", encoding="utf-8")
    service = service_with(tmp_path, FakeEngine())

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Start from this folder"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)

    assert created.workspace_kind == WorkspaceKind.local_copy
    assert created.workspace_path != str(repo.resolve())
    assert Path(created.workspace_path, "README.md").read_text(encoding="utf-8") == "uncommitted source\n"


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

    try:
        with pytest.raises(RuntimeError, match="being updated"):
            service.submit_turn(created.id, "Second turn", CREDS)
        assert engine.prompts == ["First turn"]
    finally:
        release_mutation.set()
        branch_thread.join(timeout=1)

    assert not branch_thread.is_alive()


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


def test_service_shutdown_marks_an_active_turn_interrupted(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.block_until_release = True
    service = service_with(tmp_path, engine)

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long-running turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    assert service.prepare_shutdown() == 1
    assert service.prepare_shutdown() == 0
    service.close_all()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and created.id in service._running:
        time.sleep(0.01)

    restored = service.get_session(created.id)
    assert restored.status == SessionStatus.interrupted
    assert restored.last_error is None
    assert created.id not in service._running
    assert sum(event.title == "Task interrupted" for event in service.events(created.id).items) == 1


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


def test_task_terminals_run_independently_and_keep_their_names(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    task_updated_at = service.get_session(created.id).updated_at

    first = service.create_terminal_tab(created.id)
    second = service.create_terminal_tab(created.id)
    assert (first.label, second.label) == ("Terminal 1", "Terminal 2")
    second = service.rename_terminal_tab(created.id, second.id, "Dev server")
    assert second.label == "Dev server"
    restored = service_with(tmp_path, FakeEngine())
    assert [item.label for item in restored.terminals(created.id).items] == [
        "Terminal 1", "Dev server"
    ]

    service.start_terminal_tab(created.id, first.id, CREDS, 100, 30)
    service.start_terminal_tab(created.id, second.id, CREDS, 120, 40)
    first_process, second_process = engine.terminal_process_ids
    assert first_process != second_process
    assert engine.terminal_sizes == {first_process: (100, 30), second_process: (120, 40)}

    engine.terminal_outputs[first_process]("Zmlyc3Q=", "stdout", False)
    engine.terminal_outputs[second_process]("c2Vjb25k", "stdout", False)
    assert [item.data_base64 for item in service.terminal_tab(created.id, first.id).items] == ["Zmlyc3Q="]
    assert [item.data_base64 for item in service.terminal_tab(created.id, second.id).items] == ["c2Vjb25k"]

    service.write_terminal_tab(created.id, first.id, "bHMK")
    service.resize_terminal_tab(created.id, second.id, 90, 24)
    assert engine.terminal_writes[-1] == (first_process, "bHMK")
    assert engine.terminal_sizes[second_process] == (90, 24)

    engine.terminal_exits[first_process](0, None)
    states = {item.id: item for item in service.terminals(created.id).items}
    assert states[first.id].status.value == "exited"
    assert states[second.id].status.value == "running"

    service.delete_terminal_tab(created.id, first.id)
    service.delete_terminal_tab(created.id, second.id)
    assert service.terminals(created.id).items == []
    assert second_process in engine.terminal_stops
    assert service.get_session(created.id).updated_at == task_updated_at


def test_terminal_start_reads_config_after_acquiring_runtime_lock(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    tab = service.create_terminal_tab(created.id)

    runtime_lock = service.runtimes.session_lock(created.id)
    runtime_lock.acquire()
    errors: list[BaseException] = []

    def start_terminal() -> None:
        try:
            service.start_terminal_tab(created.id, tab.id, CREDS, 100, 30)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=start_terminal)
    thread.start()
    time.sleep(0.05)
    service.runtimes.close_locked(created.id)
    service.store.update_session(
        created.id,
        lambda current: setattr(current, "model", "updated-model"),
    )
    engine.is_closed = False
    runtime_lock.release()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert errors == []
    assert engine.configs[-1].model == "updated-model"


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


def test_queue_continues_after_an_immediate_command_without_reusing_its_id(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="First turn"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    first = QueuedInstruction(id="queued-status", prompt="/status")
    second = QueuedInstruction(id="queued-turn", prompt="Second turn")
    service.store.update_session(
        created.id,
        lambda session: session.queued_instructions.extend([first, second]),
    )

    started = service.run_next_queued(created.id, CREDS, first.id)
    assert started.status == SessionStatus.running
    wait_for_status(service, created.id, SessionStatus.completed)

    assert engine.prompts == ["First turn", "Second turn"]
    assert service.get_session(created.id).queued_instructions == []


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
    finished = service.get_session(created.id)
    assert finished.queued_instructions == []
    assert finished.run_id != queued.run_id
    assert finished.run_status == "completed"


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


def test_queued_instruction_can_be_promoted_to_the_active_turn(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    queued = service.queue_turn(created.id, "Focus on the Windows build now")
    instruction_id = queued.queued_instructions[0].id
    updated = service.steer_queued_turn(created.id, instruction_id)

    wait_for_steers(engine)
    assert updated.queued_instructions == []
    assert engine.steers == [("turn-1", "Focus on the Windows build now")]
    assert service.events(created.id).items[-1].text == "Focus on the Windows build now"
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_non_steerable_queued_command_remains_queued(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    queued = service.queue_turn(created.id, "/review")
    instruction_id = queued.queued_instructions[0].id
    with pytest.raises(RuntimeError, match=r"Queue /review"):
        service.steer_queued_turn(created.id, instruction_id)

    assert [item.id for item in service.get_session(created.id).queued_instructions] == [instruction_id]
    assert engine.steers == []
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


def test_failed_queue_promotion_restores_the_instruction(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine(block_until_cancel=True)
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Long turn"), CREDS, "fake", "fake-model"
    )
    assert engine.started.wait(timeout=1)

    queued = service.queue_turn(created.id, "Try this immediately")
    instruction_id = queued.queued_instructions[0].id
    engine.steer_error = True
    with pytest.raises(RuntimeError, match="adapter rejected steer"):
        service.steer_queued_turn(created.id, instruction_id)

    restored = service.get_session(created.id).queued_instructions
    assert [item.id for item in restored] == [instruction_id]
    assert engine.steers == []
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)


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


def test_task_pin_is_persisted_without_changing_activity_order(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Pin this task"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    before = service.get_session(created.id).updated_at

    pinned = service.set_pinned(created.id, True)

    assert pinned.pinned is True
    assert pinned.updated_at == before
    assert service.get_session(created.id).pinned is True
    assert service.set_pinned(created.id, False).pinned is False


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
    service.set_pinned(parent.id, True)

    child = service.fork_session(parent.id, CREDS)

    assert child.id != parent.id
    assert child.task_id != parent.task_id
    assert child.run_id != parent.run_id
    assert service.control.store.get_task(child.task_id).id == child.task_id
    assert service.control.store.get_run(child.run_id).task_id == child.task_id
    assert child.workspace_path != parent.workspace_path
    assert child.pinned is False
    assert Path(child.workspace_path, "README.md").read_text(encoding="utf-8") == "forked work\n"
    assert child.engine_session_id == "forked-engine-session-1"
    assert engine.closed == 1
    child_events = service.events(child.id).items
    assert any(event.text == "Build the feature" for event in child_events)
    assert child_events[-1].title == "Task forked"

    service.delete_session(parent.id)
    assert Path(child.workspace_path).is_dir()


def test_fork_refuses_to_stop_a_running_parent_terminal(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    parent = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build the feature"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, parent.id, SessionStatus.completed)
    terminal = service.create_terminal_tab(parent.id)
    service.start_terminal_tab(parent.id, terminal.id, CREDS, 100, 30)

    with pytest.raises(StateConflict, match="Stop running terminals before forking"):
        service.fork_session(parent.id, CREDS)

    assert service.terminal_tab(parent.id, terminal.id).status.value == "running"
    assert engine.closed == 0


def test_fork_writes_the_durable_session_without_the_control_plane_projection(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    parent = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build the feature"), CREDS, "fake", "fake-model"
    )
    wait_for_status(service, parent.id, SessionStatus.completed)

    child = service.fork_session(parent.id, CREDS)

    stored = service.store.load_session(child.id)
    assert (stored.run_status, stored.computer_name, stored.computer_status) == (None, None, None)
    assert stored.task_capabilities == service.store.load_session(parent.id).task_capabilities
    assert child.run_status == "completed"


def test_failed_fork_preparation_does_not_leave_control_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Fork cleanup",
        folders=[ProjectFolder(id="repo", name="Repo", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    parent = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Prepare parent"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, parent.id, SessionStatus.completed)
    before = {task.id for task in service.control.store.list_tasks()}

    def fail_fork(*_args, **_kwargs):
        raise WorkspaceError("simulated fork failure")

    monkeypatch.setattr(service.project_workspaces, "fork", fail_fork)

    with pytest.raises(WorkspaceError, match="simulated fork failure"):
        service.fork_session(parent.id, CREDS)

    assert {task.id for task in service.control.store.list_tasks()} == before
    assert {session.id for session in service.store.list_sessions()} == {parent.id}


def test_project_fork_keeps_every_folder_change_isolated_and_reviewable(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "plan.txt").write_text("base\n", encoding="utf-8")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[
                ProjectFolder(id="app", name="App", path=str(repo)),
                ProjectFolder(id="notes", name="Notes", path=str(notes)),
            ],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    parent = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Build across both folders"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, parent.id, SessionStatus.completed)
    (Path(parent.workspaces[0].workspace_path) / "README.md").write_text("parent app\n", encoding="utf-8")
    (Path(parent.workspaces[1].workspace_path) / "plan.txt").write_text("parent notes\n", encoding="utf-8")

    child = service.fork_session(parent.id, CREDS)

    assert child.task_id != parent.task_id
    assert child.run_id != parent.run_id
    assert service.control.store.get_task(child.task_id).id == child.task_id
    assert service.control.store.get_run(child.run_id).task_id == child.task_id
    assert child.project_id == project.id
    assert len(child.workspaces) == 2
    assert all(item.workspace_path != parent.workspaces[index].workspace_path for index, item in enumerate(child.workspaces))
    assert (Path(child.workspaces[0].workspace_path) / "README.md").read_text(encoding="utf-8") == "parent app\n"
    assert (Path(child.workspaces[1].workspace_path) / "plan.txt").read_text(encoding="utf-8") == "parent notes\n"
    assert {(item.folder_name, item.path) for item in service.diff(child.id)} == {
        ("App", "README.md"),
        ("Notes", "plan.txt"),
    }
    assert child.allocated_ports["PORT"] != parent.allocated_ports["PORT"]
    assert child.workspaces[1].workspace_path in child.additional_dirs
    assert child.workspaces[1].workspace_path in child.developer_instructions
    assert engine.forked_workspaces[-1] == child.workspaces[0].workspace_path
    assert engine.forked_additional_dirs[-1] == tuple(child.additional_dirs)


def test_scoped_task_validation_and_fork_use_immutable_project_snapshot(tmp_path: Path) -> None:
    app = repository(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("base\n", encoding="utf-8")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Scoped",
            folders=[
                ProjectFolder(
                    id="app",
                    name="App",
                    path=str(app),
                    commands=[ProjectCommand(
                        id="snapshot-check",
                        label="Snapshot check",
                        argv=[sys.executable, "-c", "print('snapshot')"],
                        phase="validate",
                    )],
                ),
                ProjectFolder(
                    id="docs",
                    name="Docs",
                    path=str(docs),
                    commands=[ProjectCommand(
                        id="unscoped-check",
                        label="Unscoped check",
                        argv=[sys.executable, "-c", "raise SystemExit(1)"],
                        phase="validate",
                    )],
                ),
            ],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    parent = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            resource_ids=["app"],
            prompt="Work only on the app",
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, parent.id, SessionStatus.completed)

    live = service.projects.get(project.id)
    changed_resources = [
        resource.model_copy(update={
            "commands": [ProjectCommand(
                id="live-check",
                label="Live check",
                argv=[sys.executable, "-c", "raise SystemExit(1)"],
                phase="validate",
            )]
        })
        if resource.id == "app"
        else resource
        for resource in live.resources
    ]
    service.projects.update(project.id, ProjectUpdateRequest(resources=changed_resources))

    results = service.validate_project(parent.id)
    child = service.fork_session(parent.id, CREDS)

    assert [(result["label"], result["return_code"]) for result in results] == [("Snapshot check", 0)]
    assert parent.resource_ids == ["app"]
    assert [workspace.folder_id for workspace in parent.workspaces] == ["app"]
    assert child.resource_ids == ["app"]
    assert [workspace.folder_id for workspace in child.workspaces] == ["app"]
    assert child.task_id != parent.task_id
    assert child.run_id != parent.run_id


def test_project_runtime_opens_in_primary_workspace_and_keeps_other_folders_available(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "plan.txt").write_text("base\n", encoding="utf-8")
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Product",
            folders=[
                ProjectFolder(id="app", name="App", path=str(repo)),
                ProjectFolder(id="notes", name="Notes", path=str(notes)),
            ],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )

    task = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Build across both folders"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)

    primary = task.workspaces[0].workspace_path
    secondary = task.workspaces[1].workspace_path
    assert engine.opened_workspaces[-1] == primary
    assert engine.configs[-1].additional_dirs == (secondary,)


def test_project_delivery_is_planned_then_explicitly_publishes_a_draft_pr(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Delivery",
            folders=[ProjectFolder(id="app", name="App", path=str(repo))],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    task = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Prepare delivery"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)
    (Path(task.workspace_path) / "README.md").write_text("delivery\n", encoding="utf-8")

    assert service.delivery_plan(task.id).items[0].status == "needs_commit"
    service.commit(task.id, "Prepare delivery")
    assert service.delivery_plan(task.id).items[0].status == "ready"

    class Integrations:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create_draft_pull_request(self, _project, **kwargs):
            self.calls.append(kwargs)
            return "https://github.example/draft/1"

        def git_push_credentials(self, _project, repository_url, _connection_name):
            return SimpleNamespace(remote_url=repository_url, environment={})

    integrations = Integrations()
    with pytest.raises(WorkspaceError, match="Confirm"):
        service.create_draft_pull_requests(
            task.id,
            DraftPullRequestRequest(title="Delivery"),
            integrations,  # type: ignore[arg-type]
        )
    records = service.create_draft_pull_requests(
        task.id,
        DraftPullRequestRequest(title="Delivery", confirmed=True),
        integrations,  # type: ignore[arg-type]
    )

    assert records[0].status == "published"
    assert records[0].external_url == "https://github.example/draft/1"
    assert integrations.calls[0]["head"] == task.workspaces[0].task_branch
    assert service.get_session(task.id).deliveries[0].action == "draft_pull_request"
    assert service.delivery_plan(task.id).items[0].status == "published"


def test_pull_request_actions_are_limited_to_deliveries_linked_to_the_task(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(ProjectCreateRequest(
        name="Delivery authorization",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    task = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Prepare delivery"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)

    class Integrations:
        def __init__(self) -> None:
            self.requests: list[PullRequestActionRequest] = []

        def pull_request_action(self, _project, request):
            self.requests.append(request)
            return SimpleNamespace(url=request.target_url)

    integrations = Integrations()
    request = PullRequestActionRequest(
        target_url="https://github.example/pulls/7",
        action="ready",
        confirmed=True,
    )
    with pytest.raises(WorkspaceError, match="not linked"):
        service.pull_request_action(task.id, request, integrations)  # type: ignore[arg-type]
    assert integrations.requests == []

    service.record_delivery(task.id, DeliveryRecord(
        provider="github",
        action="draft_pull_request",
        target_url="https://github.example/repository",
        status="published",
        external_url="https://github.example/pulls/7/",
        connection_name="Team GitHub",
    ))
    service.pull_request_action(task.id, request, integrations)  # type: ignore[arg-type]

    assert integrations.requests[0].connection_name == "Team GitHub"


def test_external_updates_are_limited_to_work_items_linked_to_the_task(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Source authorization",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    task = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            prompt="Implement the issue",
            source_contexts=[SourceContext(
                provider="linear",
                kind="issue",
                url="https://linear.example/ENG-7",
                connection_name="Team Linear",
            )],
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)

    class Integrations:
        def __init__(self) -> None:
            self.requests: list[PublishRequest] = []

        def publish(self, _project, request):
            self.requests.append(request)
            return DeliveryRecord(
                provider=request.provider,
                action=request.action,
                target_url=request.target_url,
                status="published",
            )

    integrations = Integrations()
    with pytest.raises(WorkspaceError, match="not linked"):
        service.publish_task_update(
            task.id,
            PublishRequest(
                provider="linear",
                action="progress",
                target_url="https://linear.example/ENG-99",
                text="Progress",
                confirmed=True,
            ),
            integrations,  # type: ignore[arg-type]
        )
    assert integrations.requests == []

    service.publish_task_update(
        task.id,
        PublishRequest(
            provider="linear",
            action="progress",
            target_url="https://linear.example/ENG-7/",
            text="Progress",
            confirmed=True,
        ),
        integrations,  # type: ignore[arg-type]
    )
    assert integrations.requests[0].connection_name == "Team Linear"


def test_project_delivery_can_publish_a_selected_repository_with_its_own_copy(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = repository(first_root)
    second = repository(second_root)
    for index, repo in enumerate((first, second), start=1):
        remote = tmp_path / f"remote-{index}.git"
        git(tmp_path, "init", "--bare", str(remote))
        git(repo, "remote", "add", "origin", str(remote))
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Multi-repository delivery",
        folders=[
            ProjectFolder(id="frontend", name="Frontend", path=str(first)),
            ProjectFolder(id="server", name="Server", path=str(second)),
        ],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    task = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Prepare both repositories"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)
    for workspace in task.workspaces:
        (Path(workspace.workspace_path) / "README.md").write_text(f"{workspace.folder_name}\n", encoding="utf-8")
    service.commit(task.id, "Prepare delivery")

    class Integrations:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create_draft_pull_request(self, _project, **kwargs):
            self.calls.append(kwargs)
            return "https://github.example/draft/selected"

        def git_push_credentials(self, _project, repository_url, _connection_name):
            return SimpleNamespace(remote_url=repository_url, environment={})

    integrations = Integrations()
    records = service.create_draft_pull_requests(
        task.id,
        DraftPullRequestRequest(
            title="Shared title",
            body="Shared body",
            confirmed=True,
            drafts=[DraftPullRequestSpec(
                folder_id="server",
                title="Server-specific title",
                body="Server-specific reviewer context",
            )],
        ),
        integrations,  # type: ignore[arg-type]
    )

    assert [(record.folder_id, record.status) for record in records] == [("server", "published")]
    assert integrations.calls[0]["title"] == "Server-specific title"
    assert integrations.calls[0]["body"] == "Server-specific reviewer context"


def test_deleting_a_project_removes_its_managed_playbook_cache(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    (repo / "AGENTS.md").write_text("Keep tests green.\n", encoding="utf-8")
    git(repo, "add", "AGENTS.md")
    git(repo, "commit", "-m", "guidance")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Playbook",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    service.playbooks.configure(project.id, str(repo), branch)
    cache = service.playbooks.root / project.id

    service.delete_project(project.id)

    assert not cache.exists()
    with pytest.raises(KeyError):
        service.projects.get(project.id)


def test_failed_project_setup_is_visible_to_the_agent_without_stranding_the_task(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    project = service.projects.create(
        ProjectCreateRequest(
            name="Recoverable",
            folders=[ProjectFolder(
                id="app",
                name="App",
                path=str(repo),
                commands=[ProjectCommand(
                    id="setup",
                    label="Install dependencies",
                    argv=["git", "definitely-not-a-command"],
                    phase="setup",
                )],
            )],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )

    task = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Recover and finish the feature"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(service, task.id, SessionStatus.completed)

    restored = service.get_session(task.id)
    assert engine.prompts == ["Recover and finish the feature"]
    assert "Install dependencies" in engine.configs[0].developer_instructions
    assert Path(restored.workspace_path).is_dir()
    assert any(event.title == "Install dependencies" and event.phase == "failed" for event in service.events(task.id).items)


def test_project_task_resumes_same_workspaces_and_ports_after_service_restart(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    notes = tmp_path / "notes"
    notes.mkdir()
    engine = FakeEngine()
    first = service_with(tmp_path, engine)
    project = first.projects.create(
        ProjectCreateRequest(
            name="Restart",
            folders=[
                ProjectFolder(id="app", name="App", path=str(repo)),
                ProjectFolder(id="notes", name="Notes", path=str(notes)),
            ],
            default_engine_id="fake",
            default_model="fake-model",
        )
    )
    task = first.create_session(
        SessionCreateRequest(project_id=project.id, prompt="First turn"),
        CREDS,
        "fake",
        "fake-model",
    )
    wait_for_status(first, task.id, SessionStatus.completed)
    original_paths = [item.workspace_path for item in first.get_session(task.id).workspaces]
    original_ports = first.get_session(task.id).allocated_ports
    original_engine_session_id = first.get_session(task.id).engine_session_id
    first.close_all()

    restarted_engine = FakeEngine()
    restarted = service_with(tmp_path, restarted_engine)
    loaded = restarted.get_session(task.id)
    restarted.submit_turn(task.id, "Continue", CREDS)
    wait_for_status(restarted, task.id, SessionStatus.completed)
    next_task = restarted.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Parallel task"),
        CREDS,
        "fake",
        "fake-model",
    )

    assert [item.workspace_path for item in loaded.workspaces] == original_paths
    assert loaded.allocated_ports == original_ports
    assert restarted_engine.existing_ids[0] == original_engine_session_id
    assert next_task.allocated_ports["PORT"] != original_ports["PORT"]
