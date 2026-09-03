from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from cowork.coding.contracts import utc_now
from cowork.coding.control_models import (
    CodeTask,
    ComputerCapabilities,
    ConnectorGrant,
    ExecutionWorkspace,
    RunStatus,
    RuntimeCommand,
    SecurityAuditEvent,
    TaskRun,
)
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.sql_control_store import SqlControlPlaneStore
from cowork.db.session import get_session_factory


def capabilities() -> ComputerCapabilities:
    return ComputerCapabilities(
        platform="linux",
        architecture="test",
        runtime_version="test",
        agent_engines=["codex"],
        shells=["bash"],
    )


def stores() -> tuple[SqlControlPlaneStore, SqlControlPlaneStore]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    factory = get_session_factory(engine)
    return (
        SqlControlPlaneStore(factory, "org-a"),
        SqlControlPlaneStore(factory, "org-b"),
    )


def test_sql_control_store_is_transactional_and_tenant_isolated() -> None:
    first, second = stores()
    task = CodeTask(id="task", title="Task")
    run = TaskRun(id="run", task_id=task.id, computer_id="computer")
    workspace = ExecutionWorkspace(
        id="workspace",
        run_id=run.id,
        resource_id="repo",
        computer_id="computer",
    )

    first.create_task_run(task, run, [workspace])

    assert first.get_task(task.id).title == "Task"
    assert first.list_workspaces(run.id)[0].resource_id == "repo"
    assert second.list_tasks() == []


def test_sql_registration_and_leasing_survive_service_instances(tmp_path: Path) -> None:
    first, _ = stores()
    issuer = ControlPlaneService(tmp_path / "issuer", capabilities(), first)
    token = issuer.issue_registration_token()
    registrar = ControlPlaneService(tmp_path / "registrar", capabilities(), first)
    computer, _ = registrar.register_runtime(token, "Build computer", capabilities())
    registrar.create_task_run(
        task_id="task",
        title="Task",
        prompt="Run",
        project=None,
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
        standalone_computer_id=registrar.local_computer.id,
    )

    assert computer.name == "Build computer"
    assert registrar.acquire_lease(registrar.local_computer.id) is not None


def test_sql_scheduler_claims_only_the_target_computers_oldest_run() -> None:
    store, _ = stores()
    for suffix, computer_id in [("a", "computer-a"), ("b", "computer-b"), ("c", "computer-a")]:
        task = CodeTask(id=f"task-{suffix}", title=f"Task {suffix}")
        run = TaskRun(id=f"run-{suffix}", task_id=task.id, computer_id=computer_id)
        store.create_task_run(task, run, [])

    claimed = store.claim_run("computer-a", "lease", utc_now() + timedelta(minutes=1))

    assert claimed is not None
    assert claimed.id == "run-a"
    assert claimed.status == RunStatus.preparing
    assert store.get_run("run-b").status == RunStatus.queued
    assert store.get_run("run-c").status == RunStatus.queued


def test_sql_security_audit_is_append_only() -> None:
    store, _ = stores()
    event = SecurityAuditEvent(
        id="audit-one",
        action="runtime.register",
        outcome="completed",
        actor_type="runtime",
        target_id="computer",
    )

    store.save_audit_event(event)

    with pytest.raises(ValueError, match="append-only"):
        store.save_audit_event(event.model_copy(update={"detail": "rewritten"}))


def test_sql_grant_consumption_validates_the_locked_task_run() -> None:
    store, _ = stores()
    task = CodeTask(id="task", title="Task")
    run = TaskRun(id="run", task_id=task.id, computer_id="computer", epoch=3)
    store.create_task_run(task, run, [])
    grant = ConnectorGrant(
        id="grant",
        run_id=run.id,
        computer_id=run.computer_id,
        epoch=run.epoch,
        provider="github",
        connection_name="work",
        actions=["read_source"],
        token_hash="0" * 64,
        expires_at=utc_now() + timedelta(minutes=5),
    )
    store.save_grant(grant)

    def consume(locked_grant: ConnectorGrant, locked_run: TaskRun) -> None:
        assert locked_run.epoch == locked_grant.epoch == 3
        locked_grant.use_count += 1

    updated = store.update_grant(grant.id, consume)

    assert updated.use_count == 1
    assert store.get_grant(grant.id).use_count == 1


def test_sql_parent_projection_scopes_runtime_commands_and_cascade_delete() -> None:
    store, _ = stores()
    task = CodeTask(id="task", title="Task")
    run = TaskRun(id="run", task_id=task.id, computer_id="computer")
    store.create_task_run(task, run, [])
    store.save_command(RuntimeCommand(id="command", run_id=run.id, epoch=1, kind="start"))
    store.save_command(RuntimeCommand(id="other-command", run_id="other-run", epoch=1, kind="start"))

    assert [item.id for item in store.list_commands(run.id)] == ["command"]

    store.delete_task(task.id)

    with pytest.raises(KeyError):
        store.get_task(task.id)
    assert store.list_commands(run.id) == []
    assert store.get_command("other-command").run_id == "other-run"
