from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from cowork.coding.contracts import utc_now
from cowork.coding.control_models import (
    CodeTask,
    ComputerCapabilities,
    ExecutionWorkspace,
    RunStatus,
    RuntimeEvent,
    TaskRun,
)
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_service import ControlPlaneService, StaleRuntimeEvent
from cowork.coding.control_store import ControlPlaneStore, LocalControlPlaneStore
from cowork.coding.project_models import CodeProject, RepositoryResource
from cowork.coding.sql_control_store import SqlControlPlaneStore
from cowork.db.session import get_session_factory


def sql_store(tmp_path: Path) -> SqlControlPlaneStore:
    engine = create_engine(f"sqlite:///{tmp_path / 'control.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return SqlControlPlaneStore(get_session_factory(engine), "org-conformance")


@pytest.fixture(params=["local", "sql"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ControlPlaneStore:
    if request.param == "local":
        return LocalControlPlaneStore(tmp_path / "local")
    return sql_store(tmp_path)


def capabilities() -> ComputerCapabilities:
    return ComputerCapabilities(
        platform="linux",
        architecture="test",
        runtime_version="test-runtime",
        agent_engines=["codex"],
        shells=["bash"],
    )


def leased_run(store: ControlPlaneStore, tmp_path: Path) -> tuple[ControlPlaneService, TaskRun, str]:
    service = ControlPlaneService(tmp_path / "service", capabilities(), store)
    remote, _ = service.register_runtime(service.issue_registration_token(), "Remote", capabilities())
    service.create_task_run(
        task_id="portable",
        title="Portable",
        prompt="Run remotely",
        project=CodeProject(
            id="portable-project",
            name="Portable",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    leased = service.acquire_lease(remote.id)
    assert leased is not None
    return service, *leased


def status_event(run: TaskRun, lease_id: str, seq: int, status: str) -> RuntimeEvent:
    return RuntimeEvent(
        run_id=run.id,
        computer_id=run.computer_id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=seq,
        kind="status",
        payload={"status": status},
    )


def test_update_run_persists_the_operation_and_leaves_nothing_behind_on_failure(store: ControlPlaneStore) -> None:
    store.create_task_run(CodeTask(id="task", title="Task"), TaskRun(id="run", task_id="task", computer_id="c"), [])

    updated = store.update_run("run", lambda run: run.checkpoint.update({"phase": "one"}))

    assert updated.checkpoint == {"phase": "one"}
    assert store.get_run("run").checkpoint == {"phase": "one"}

    def half_applied(run: TaskRun) -> None:
        run.checkpoint = {"phase": "two"}
        raise RuntimeError("side effect failed")

    with pytest.raises(RuntimeError, match="side effect failed"):
        store.update_run("run", half_applied)
    assert store.get_run("run").checkpoint == {"phase": "one"}
    with pytest.raises(KeyError):
        store.update_run("missing", lambda run: None)


def test_sql_update_run_rolls_back_the_operation_s_other_writes_with_the_run(tmp_path: Path) -> None:
    store = sql_store(tmp_path)
    store.create_task_run(CodeTask(id="task", title="Task"), TaskRun(id="run", task_id="task", computer_id="c"), [])

    def half_applied(run: TaskRun) -> None:
        store.save_workspace(ExecutionWorkspace(id="workspace", run_id=run.id, resource_id="repo", computer_id="c"))
        assert [item.id for item in store.list_workspaces(run.id)] == ["workspace"]
        run.checkpoint = {"phase": "two"}
        raise RuntimeError("side effect failed")

    with pytest.raises(RuntimeError, match="side effect failed"):
        store.update_run("run", half_applied)

    assert store.list_workspaces("run") == []
    assert store.get_run("run").checkpoint == {}


def test_creating_a_task_run_twice_is_a_state_conflict(store: ControlPlaneStore) -> None:
    store.create_task_run(CodeTask(id="task", title="Task"), TaskRun(id="run", task_id="task", computer_id="c"), [])

    with pytest.raises(StateConflict, match="already exists"):
        store.create_task_run(CodeTask(id="other", title="Other"), TaskRun(id="run", task_id="other", computer_id="c"), [])
    assert store.get_run("run").task_id == "task"


def test_update_run_without_changes_does_not_rewrite_the_record(store: ControlPlaneStore) -> None:
    created = TaskRun(id="run", task_id="task", computer_id="c", updated_at=utc_now() - timedelta(minutes=5))
    store.create_task_run(CodeTask(id="task", title="Task"), created, [])

    store.update_run("run", lambda run: None)

    assert store.get_run("run").updated_at == created.updated_at


def test_accept_event_applies_the_event_and_its_sequence_together(store: ControlPlaneStore, tmp_path: Path) -> None:
    service, run, lease_id = leased_run(store, tmp_path)
    event = status_event(run, lease_id, 1, "ready")
    seen: list[RunStatus] = []

    def failing_side_effect(current: TaskRun) -> None:
        seen.append(current.status)
        raise RuntimeError("read model unavailable")

    with pytest.raises(RuntimeError, match="read model unavailable"):
        service.accept_event(event, failing_side_effect)
    untouched = store.get_run(run.id)
    assert seen == [RunStatus.ready]
    assert (untouched.status, untouched.last_event_seq, untouched.last_event_id) == (RunStatus.preparing, 0, None)

    redelivered = service.accept_event(event, lambda current: seen.append(current.status))

    assert (redelivered.status, redelivered.last_event_seq, redelivered.last_event_id) == (RunStatus.ready, 1, event.id)
    assert store.get_run(run.id).status == RunStatus.ready


def test_redelivered_event_is_acknowledged_once_and_lower_sequences_are_stale(
    store: ControlPlaneStore,
    tmp_path: Path,
) -> None:
    service, run, lease_id = leased_run(store, tmp_path)
    applied: list[int] = []
    first = status_event(run, lease_id, 1, "ready")
    second = status_event(run, lease_id, 2, "running")

    acknowledged = service.accept_event(first, lambda current: applied.append(1))
    again = service.accept_event(first, lambda current: applied.append(1))
    assert applied == [1]
    assert (again.status, again.last_event_seq) == (acknowledged.status, acknowledged.last_event_seq)
    assert again.lease_expires_at >= acknowledged.lease_expires_at

    with pytest.raises(StaleRuntimeEvent, match="sequence is stale"):
        service.accept_event(first.model_copy(update={"id": "event-reusing-seq-1"}))
    service.accept_event(second, lambda current: applied.append(2))
    with pytest.raises(StaleRuntimeEvent, match="sequence is stale"):
        service.accept_event(first)
    assert applied == [1, 2]
    assert store.get_run(run.id).status == RunStatus.running


def test_terminal_event_redelivery_is_acknowledged_after_the_event_clears_its_lease(
    store: ControlPlaneStore,
    tmp_path: Path,
) -> None:
    service, run, lease_id = leased_run(store, tmp_path)
    terminal = status_event(run, lease_id, 1, "failed")
    applied: list[RunStatus] = []

    accepted = service.accept_event(terminal, lambda current: applied.append(current.status))
    redelivered = service.accept_event(terminal, lambda current: applied.append(current.status))

    assert accepted.lease_id is None
    assert redelivered.status == RunStatus.failed
    assert (redelivered.last_event_seq, redelivered.last_event_id) == (1, terminal.id)
    assert applied == [RunStatus.failed]


def test_expired_lease_prepares_a_workspace_that_was_never_published(
    store: ControlPlaneStore,
    tmp_path: Path,
) -> None:
    service, run, _ = leased_run(store, tmp_path)
    store.update_run(run.id, lambda current: setattr(current, "lease_expires_at", utc_now() - timedelta(seconds=1)))

    service.expire_leases()

    recovered = store.get_run(run.id)
    assert recovered.status == RunStatus.recovering
    assert recovered.workspace_resume_mode == "prepare"


def test_lifecycle_mutations_write_the_run_only_under_the_store_lock(
    store: ControlPlaneStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, run, _ = leased_run(store, tmp_path)
    locked: list[str] = []
    unlocked_writes: list[str] = []
    real_update_run = store.update_run
    real_save_run = store.save_run

    def recording_update_run(run_id, operation):
        locked.append(run_id)
        return real_update_run(run_id, operation)

    def recording_save_run(candidate):
        if not locked or locked[-1] != candidate.id:
            unlocked_writes.append(candidate.id)
        return real_save_run(candidate)

    monkeypatch.setattr(store, "update_run", recording_update_run)
    monkeypatch.setattr(store, "save_run", recording_save_run)

    assert service.set_run_status(run.id, RunStatus.ready).status == RunStatus.ready
    service.queue_command(run.id, "start", {"prompt": "Build"}, "start-1")
    claimed = service.claim_commands(run.id, run.computer_id, service.store.get_run(run.id).lease_id, run.epoch)
    assert [item.kind for item in claimed] == ["start"]
    store.update_run(run.id, lambda current: setattr(current, "lease_expires_at", utc_now() - timedelta(seconds=1)))
    service.expire_leases()
    assert store.get_run(run.id).status == RunStatus.recovering
    recovered = service.recover_run(run.id, run.computer_id)
    assert recovered.epoch == run.epoch + 2

    assert locked == [run.id] * 5
    assert unlocked_writes == []
