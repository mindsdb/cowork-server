from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from cowork.coding.contracts import utc_now
from cowork.coding.control_models import (
    RUNTIME_PROTOCOL_VERSION,
    CodeTask,
    Computer,
    ComputerCapabilities,
    ComputerStatus,
    ConnectorGrant,
    ExecutionWorkspace,
    RunStatus,
    RuntimeCommand,
    RuntimeEvent,
    RuntimeRegistrationCredential,
    SecurityAuditEvent,
    TaskRun,
    WorkspaceStatus,
)
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_service import (
    ControlPlaneService,
    NoEligibleComputer,
    RuntimeAuthenticationError,
    StaleRuntimeEvent,
)
from cowork.coding.control_store import LocalControlPlaneStore
from cowork.coding.project_models import (
    CodeProject,
    LocalFolderResource,
    RepositoryResource,
)
from cowork.coding.run_state import InvalidRunTransition


def capabilities(*, max_runs: int = 4) -> ComputerCapabilities:
    return ComputerCapabilities(
        platform="linux",
        architecture="test",
        runtime_version="test-runtime",
        protocol_versions=[RUNTIME_PROTOCOL_VERSION],
        agent_engines=["codex"],
        shells=["bash"],
        max_concurrent_runs=max_runs,
    )


def register(service: ControlPlaneService, name: str = "Remote", *, max_runs: int = 4):
    token = service.issue_registration_token()
    return service.register_runtime(token, name, capabilities(max_runs=max_runs))


def test_task_run_bundle_rolls_back_if_local_persistence_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalControlPlaneStore(tmp_path)
    task = CodeTask(id="atomic-task", title="Atomic task")
    run = TaskRun(id="atomic-run", task_id=task.id, computer_id="computer")
    workspace = ExecutionWorkspace(
        id="atomic-workspace",
        run_id=run.id,
        resource_id="resource",
        computer_id="computer",
    )
    real_replace = os.replace

    def interrupted_replace(source: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if target.name == f"{task.id}.json":
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr("cowork.coding.control_store.os.replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        store.create_task_run(task, run, [workspace])

    with pytest.raises(KeyError):
        store.get_task(task.id)
    assert not (tmp_path / "control" / ".transaction.json").exists()


def test_separate_store_instances_can_save_the_same_document_concurrently(tmp_path: Path) -> None:
    stores = [LocalControlPlaneStore(tmp_path) for _ in range(12)]
    computer = Computer(
        id="shared-computer",
        name="Shared computer",
        capabilities=capabilities(),
    )

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        saved = list(pool.map(
            lambda store: store.save_computer(computer.model_copy(deep=True)),
            stores,
        ))

    assert len(saved) == len(stores)
    assert stores[0].get_computer(computer.id).name == computer.name
    assert not list((tmp_path / "control" / "computers").glob("*.tmp"))


def test_runtime_command_polling_uses_the_run_index_not_a_collection_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalControlPlaneStore(tmp_path)
    store.save_command(RuntimeCommand(id="command-a", run_id="run-a", epoch=1, kind="start"))
    store.save_command(RuntimeCommand(id="command-b", run_id="run-b", epoch=1, kind="start"))
    monkeypatch.setattr(store, "_list", lambda *_args, **_kwargs: pytest.fail("full collection scan"))

    assert [item.id for item in store.list_commands("run-a")] == ["command-a"]
    assert store.get_command("command-b").run_id == "run-b"


def test_deleting_a_control_task_cascades_through_its_run_records(tmp_path: Path) -> None:
    store = LocalControlPlaneStore(tmp_path)
    task = CodeTask(id="task", title="Task")
    run = TaskRun(id="run", task_id=task.id, computer_id="computer")
    workspace = ExecutionWorkspace(id="workspace", run_id=run.id, resource_id="repo", computer_id="computer")
    store.create_task_run(task, run, [workspace])
    store.save_command(RuntimeCommand(id="command", run_id=run.id, epoch=1, kind="start"))
    store.save_audit_event(SecurityAuditEvent(
        id="audit", action="task.start", outcome="completed", actor_type="system", run_id=run.id,
    ))

    store.delete_task(task.id)

    with pytest.raises(KeyError):
        store.get_task(task.id)
    with pytest.raises(KeyError):
        store.get_run(run.id)
    assert store.list_workspaces(run.id) == []
    assert store.list_commands(run.id) == []
    assert store.list_audit_events(run.id) == []


def test_control_store_prunes_expired_auxiliary_history_only(tmp_path: Path) -> None:
    store = LocalControlPlaneStore(tmp_path)
    old = utc_now() - timedelta(days=120)
    task = store.save_task(CodeTask(id="task", title="Task"))
    store.save_command(RuntimeCommand(
        id="command", run_id="run", epoch=1, kind="start", created_at=old, acked_at=old,
    ))
    store.save_registration_credential(RuntimeRegistrationCredential(
        id="0" * 64, token_hash="0" * 64, expires_at=old,
    ))
    store.save_audit_event(SecurityAuditEvent(
        id="audit", action="task.start", outcome="completed", actor_type="system", created_at=old,
    ))

    removed = store.prune(utc_now() - timedelta(days=30), utc_now() - timedelta(days=90))

    assert removed == 3
    assert store.get_task(task.id).title == "Task"
    assert store.list_commands() == []
    assert store.list_audit_events() == []


def project(local_computer_id: str) -> CodeProject:
    return CodeProject(
        id="product",
        name="Product",
        resources=[
            RepositoryResource(
                id="api",
                name="API",
                source_url="https://github.com/example/api.git",
            ),
            LocalFolderResource(
                id="notes",
                name="Notes",
                path="/example/notes",
                computer_id=local_computer_id,
            ),
        ],
    )


def test_registration_is_one_use_and_computers_survive_restart(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    registration_token = service.issue_registration_token()
    # Registration authority is durable so a runtime may reach another API
    # worker (or the service may restart) after the pairing code is shown.
    service = ControlPlaneService(tmp_path, capabilities())
    computer, runtime_token = service.register_runtime(
        registration_token,
        "Remote",
        capabilities(),
    )
    assert service.authenticate_runtime(computer.id, runtime_token).id == computer.id
    with pytest.raises(RuntimeAuthenticationError, match="already used"):
        service.register_runtime("invalid-token-that-is-long-enough-for-testing", "Other", capabilities())

    restarted = ControlPlaneService(tmp_path, capabilities())
    persisted = {item.id: item for item in restarted.list_computers().items}
    assert computer.id in persisted
    assert service.local_computer.id == restarted.local_computer.id
    assert restarted.authenticate_runtime(computer.id, runtime_token).id == computer.id


def test_computers_can_be_named_and_remote_access_revoked(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, runtime_token = register(service, "Build computer")

    renamed = service.rename_computer(remote.id, "  Release   runner  ")
    assert renamed.name == "Release runner"
    assert not renamed.is_local

    service.revoke_computer(remote.id)
    assert remote.id not in {item.id for item in service.list_computers().items}
    with pytest.raises(RuntimeAuthenticationError, match="revoked"):
        service.authenticate_runtime(remote.id, runtime_token)


def test_this_computer_can_be_renamed_but_not_revoked(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    local = service.rename_computer(service.local_computer.id, "Ian's Mac")
    assert local.name == "Ian's Mac"
    assert local.is_local

    restarted = ControlPlaneService(tmp_path, capabilities())
    assert restarted.local_computer.name == "Ian's Mac"
    with pytest.raises(ValueError, match="cannot be revoked"):
        restarted.revoke_computer(restarted.local_computer.id)


def test_every_registration_mints_a_distinct_computer(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    first, first_token = register(service, "Remote")
    second, second_token = register(service, "Remote")

    assert first.id != second.id
    assert first_token != second_token
    assert service.authenticate_runtime(first.id, first_token).id == first.id
    assert service.authenticate_runtime(second.id, second_token).id == second.id


def test_revoked_computer_cannot_lease_again(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.revoke_computer(remote.id)

    with pytest.raises(RuntimeAuthenticationError, match="revoked"):
        service.acquire_lease(remote.id)
    assert service.store.get_computer(remote.id).revoked_at is not None
    assert remote.id not in {item.id for item in service.list_computers().items}


def test_reconnection_requires_the_computers_existing_credential(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, runtime_token = register(service)

    restarted = ControlPlaneService(tmp_path, capabilities())
    with pytest.raises(RuntimeAuthenticationError, match="failed"):
        restarted.authenticate_runtime(remote.id, "not-the-credential-issued-at-registration")
    assert restarted.authenticate_runtime(remote.id, runtime_token).id == remote.id


def test_queued_work_moves_between_computers_only_by_audited_reassignment(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    first, _ = register(service, "First")
    second, _ = register(service, "Second")
    snapshot = service.create_task_run(
        task_id="queued-move",
        title="Queued move",
        prompt="Run wherever",
        project=CodeProject(
            id="portable-project",
            name="Portable",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=first.id,
        engine_id="codex",
    )
    bound = service.create_task_run(
        task_id="bound-task",
        title="Bound task",
        prompt="Update notes",
        project=project(service.local_computer.id),
        requested_resource_ids=None,
        computer_id=service.local_computer.id,
        engine_id="codex",
    )

    moved = service.reassign_queued_run(snapshot.run.id, second.id)

    assert moved.computer_id == second.id
    assert moved.status == RunStatus.queued
    assert {item.computer_id for item in service.store.list_workspaces(moved.id)} == {second.id}
    assert any(
        item.action == "run.reassign" and item.computer_id == second.id and first.id in item.detail
        for item in service.store.list_audit_events(moved.id)
    )
    assert service.acquire_lease(first.id) is None
    leased = service.acquire_lease(second.id)
    assert leased is not None
    assert leased[0].id == moved.id
    with pytest.raises(StateConflict, match="queued"):
        service.reassign_queued_run(moved.id, first.id)
    with pytest.raises(NoEligibleComputer, match="cannot access"):
        service.reassign_queued_run(bound.run.id, second.id)


def test_local_folders_are_owner_bound_but_repositories_are_portable(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    code_project = project(service.local_computer.id)

    repo_run = service.create_task_run(
        task_id="repo-task",
        title="Repository task",
        prompt="Change the API",
        project=code_project,
        requested_resource_ids=["api"],
        computer_id=remote.id,
        engine_id="codex",
    )
    assert repo_run.computer.id == remote.id

    with pytest.raises(NoEligibleComputer, match="cannot access"):
        service.create_task_run(
            task_id="folder-task",
            title="Folder task",
            prompt="Update notes",
            project=code_project,
            requested_resource_ids=["notes"],
            computer_id=remote.id,
            engine_id="codex",
        )


def test_task_resource_snapshot_is_immutable_when_project_changes(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    code_project = project(service.local_computer.id)
    snapshot = service.create_task_run(
        task_id="scoped-task",
        title="Scoped task",
        prompt="Only change the API",
        project=code_project,
        requested_resource_ids=["api"],
        computer_id=remote.id,
        engine_id="codex",
    )

    code_project.resources[0].name = "Renamed after task creation"
    code_project.resources.append(RepositoryResource(
        id="web",
        name="Web",
        source_url="https://github.com/example/web.git",
    ))

    task = service.store.get_task(snapshot.task.id)
    assert task.execution_project is not None
    assert [(item.id, item.name) for item in task.execution_project.resources] == [("api", "API")]
    assert [item.id for item in service.runtime_project_for_task(task, remote.id).resources] == ["api"]


def test_no_project_folder_task_is_kept_on_its_owning_computer(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    assert remote.id != service.local_computer.id

    snapshot = service.create_task_run(
        task_id="standalone",
        title="Standalone",
        prompt="Inspect this folder",
        project=None,
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
        standalone_computer_id=service.local_computer.id,
    )
    assert snapshot.computer.id == service.local_computer.id


def test_leases_fence_stale_and_duplicate_runtime_events(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    snapshot = service.create_task_run(
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
    run, lease_id = leased
    assert run.status == RunStatus.preparing

    event = RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=1,
        kind="status",
        payload={"status": "ready"},
    )
    assert service.accept_event(event).status == RunStatus.ready
    assert service.accept_event(event).last_event_seq == 1
    with pytest.raises(StaleRuntimeEvent, match="sequence is stale"):
        service.accept_event(event.model_copy(update={"id": "event-reusing-a-sequence"}))

    recovered = service.recover_run(
        snapshot.run.id,
        service.local_computer.id,
        allow_recreate=True,
    )
    assert recovered.epoch == run.epoch + 1
    with pytest.raises(StaleRuntimeEvent, match="ownership changed"):
        service.accept_event(event.model_copy(update={"seq": 2}))


def test_runtime_checkpoint_and_error_payloads_are_sanitized_before_persistence(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.create_task_run(
        task_id="sanitized",
        title="Sanitized",
        prompt="Run remotely",
        project=CodeProject(
            id="sanitized-project",
            name="Sanitized",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    leased = service.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased

    checkpoint = service.accept_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=1,
        kind="checkpoint",
        payload={"authorization": "Bearer secret-token", "detail": "token=secret-token"},
    ))
    failed = service.accept_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=2,
        kind="error",
        payload={"detail": "Authorization: Bearer secret-token"},
    ))

    assert checkpoint.checkpoint == {"authorization": "[redacted]", "detail": "token=[redacted]"}
    assert failed.last_error == "Authorization: Bearer [redacted]"


def test_a_classified_runtime_failure_keeps_its_user_message_as_the_run_error(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.create_task_run(
        task_id="task-credits",
        title="Credits",
        prompt="Build",
        project=CodeProject(
            id="credits-project",
            name="Credits",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    leased = service.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased

    failed = service.accept_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=1,
        kind="error",
        payload={
            "message": "This model needs credits. Add credits or choose another model.",
            "detail": "unexpected status 402 Payment Required: wallet empty",
            "code": "insufficient_credits",
            "model": "gpt",
        },
    ))

    assert failed.status is RunStatus.failed
    assert failed.last_error == "This model needs credits. Add credits or choose another model."


def test_only_one_concurrent_runtime_claim_can_acquire_a_queued_run(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.create_task_run(
        task_id="single-lease-task",
        title="Single lease",
        prompt="Run once",
        project=CodeProject(
            id="single-lease-project",
            name="Single lease project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://github.com/acme/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(lambda _: service.acquire_lease(remote.id), range(8)))

    assert len([claim for claim in claims if claim is not None]) == 1


def test_recovered_run_can_be_leased_for_workspace_preparation(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    snapshot = service.create_task_run(
        task_id="recoverable",
        title="Recoverable",
        prompt="Resume safely",
        project=CodeProject(
            id="recoverable-project",
            name="Recoverable",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    first = service.acquire_lease(remote.id)
    assert first is not None
    service.set_run_status(snapshot.run.id, RunStatus.interrupted)

    recovered = service.recover_run(snapshot.run.id, remote.id)
    assert recovered.status == RunStatus.recovering
    assert recovered.epoch == 2
    assert recovered.last_event_seq == 0
    assert recovered.checkpoint == {}

    resumed = service.acquire_lease(remote.id)
    assert resumed is not None
    run, lease_id = resumed
    assert run.status == RunStatus.preparing
    assert run.epoch == recovered.epoch
    assert lease_id != first[1]
    assert service.accept_event(RuntimeEvent(
        run_id=run.id,
        computer_id=remote.id,
        lease_id=lease_id,
        epoch=run.epoch,
        seq=1,
        kind="status",
        payload={"status": "ready"},
    )).status == RunStatus.ready


def test_recovery_plan_describes_reopening_not_resuming_the_turn(tmp_path: Path) -> None:
    # Recovery re-provisions the run to ready; it never replays the interrupted
    # turn, so the option copy must not promise that the work resumes.
    service = ControlPlaneService(tmp_path, capabilities())
    computer, _ = register(service, "Laptop")
    snapshot = service.create_task_run(
        task_id="reopen-run",
        title="Reopen run",
        prompt="Keep going",
        project=None,
        requested_resource_ids=None,
        computer_id=computer.id,
        engine_id="codex",
    )
    assert service.acquire_lease(computer.id) is not None
    service.set_run_status(snapshot.run.id, RunStatus.interrupted)

    plan = service.recovery_plan(snapshot.run.id)

    assert [option.mode for option in plan.options] == ["restore"]
    assert plan.options[0].detail.startswith("Reopen the saved working copy")
    assert "Resume" not in plan.options[0].detail


def test_a_named_computer_is_listed_as_pending_until_its_runtime_connects(tmp_path: Path) -> None:
    # "Connect a computer" names the computer up front; the entry must survive
    # closing the dialog and turn into the real computer when the code is used.
    service = ControlPlaneService(tmp_path, capabilities())

    token, pending = service.invite_computer("Build box", "linux")
    assert pending.name == "Build box"
    assert pending.platform == "linux"
    assert pending.expired is False
    page = service.list_computers()
    assert [item.name for item in page.pending] == ["Build box"]
    assert all(item.is_local for item in page.items)

    computer, _runtime_token = service.register_runtime(token, "runtime-host-name", capabilities())

    assert computer.name == "Build box"  # the typed name wins over the runtime's host name
    page = service.list_computers()
    assert page.pending == []
    assert computer.id in {item.id for item in page.items}


def test_a_pending_computer_can_get_a_fresh_code_or_be_removed(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    first_token, first = service.invite_computer("Laptop", "darwin")

    second_token, second = service.invite_computer("Laptop", "darwin", replaces_id=first.id)
    assert second.id != first.id
    assert [item.id for item in service.list_computers().pending] == [second.id]
    with pytest.raises(RuntimeAuthenticationError):
        service.register_runtime(first_token, "Laptop", capabilities())  # the replaced code is dead

    service.remove_pending_computer(second.id)
    assert service.list_computers().pending == []
    with pytest.raises(RuntimeAuthenticationError):
        service.register_runtime(second_token, "Laptop", capabilities())
    with pytest.raises(KeyError):
        service.remove_pending_computer(second.id)


def test_an_expired_pending_computer_is_flagged_not_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    monkeypatch.setattr("cowork.coding.control_service.REGISTRATION_CODE_LIFETIME", timedelta(seconds=-1))
    _token, pending = service.invite_computer("Old code", "windows")

    listed = service.list_computers().pending
    assert [item.id for item in listed] == [pending.id]
    assert listed[0].expired is True


def test_anonymous_registration_tokens_do_not_show_as_pending_computers(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    service.issue_registration_token()
    assert service.list_computers().pending == []


def test_cross_computer_recovery_recreates_only_portable_scoped_resources(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    first_computer, _ = register(service, "First")
    second_computer, _ = register(service, "Second")
    snapshot = service.create_task_run(
        task_id="migrate-run",
        title="Migrate run",
        prompt="Continue elsewhere",
        project=CodeProject(
            id="portable-project",
            name="Portable",
            resources=[RepositoryResource(
                id="repo",
                name="Repo",
                source_url="https://example.test/repo.git",
            )],
        ),
        requested_resource_ids=None,
        computer_id=first_computer.id,
        engine_id="codex",
    )
    workspace = service.store.list_workspaces(snapshot.run.id)[0]
    workspace.status = WorkspaceStatus.ready
    workspace.path = "/old-computer/workspace"
    service.store.save_workspace(workspace)
    assert service.acquire_lease(first_computer.id) is not None
    service.set_run_status(snapshot.run.id, RunStatus.interrupted)

    recovered = service.recover_run(
        snapshot.run.id,
        second_computer.id,
        allow_recreate=True,
    )

    assert recovered.workspace_resume_mode == "recreate"
    assert recovered.recovery_count == 1
    migrated = service.store.list_workspaces(snapshot.run.id)[0]
    assert migrated.computer_id == second_computer.id
    assert migrated.status == "pending"
    assert migrated.path == ""


def test_capacity_counts_only_computer_work_not_idle_task_history(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities(max_runs=1))
    remote, _ = register(service, max_runs=1)
    code_project = CodeProject(
        id="project",
        name="Project",
        resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
    )
    first = service.create_task_run(
        task_id="first",
        title="First",
        prompt="First",
        project=code_project,
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    # Queued metadata does not consume execution capacity until leased.
    second = service.create_task_run(
        task_id="second",
        title="Second",
        prompt="Second",
        project=code_project,
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    assert first.run.status == second.run.status == RunStatus.queued
    assert service.acquire_lease(remote.id) is not None
    assert service.store.get_computer(remote.id).status == ComputerStatus.online
    with pytest.raises(NoEligibleComputer, match="cannot access"):
        service.create_task_run(
            task_id="third",
            title="Third",
            prompt="Third",
            project=code_project,
            requested_resource_ids=None,
            computer_id=remote.id,
            engine_id="codex",
        )


def test_terminal_run_cannot_be_reopened_without_explicit_recovery(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    snapshot = service.create_task_run(
        task_id="done",
        title="Done",
        prompt="Done",
        project=None,
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
        standalone_computer_id=service.local_computer.id,
    )
    service.set_run_status(snapshot.run.id, RunStatus.preparing)
    service.set_run_status(snapshot.run.id, RunStatus.ready)
    service.set_run_status(snapshot.run.id, RunStatus.running)
    service.set_run_status(snapshot.run.id, RunStatus.completed)
    with pytest.raises(InvalidRunTransition, match="completed to running"):
        service.set_run_status(snapshot.run.id, RunStatus.running)


def test_connector_capabilities_are_short_lived_scoped_and_revocable(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    snapshot = service.create_task_run(
        task_id="connector-task",
        title="Connector task",
        prompt="Read one repository",
        project=CodeProject(
            id="connector-project",
            name="Connector project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://github.com/acme/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
    )
    grant, token = service.issue_connector_grant(
        snapshot.run.id,
        "github",
        "work-account",
        ["read_source"],
        {"repository": "acme/repo"},
    )
    assert service.authorize_connector(
        grant.id,
        token,
        "read_source",
        {"repository": "acme/repo"},
    ).connection_name == "work-account"
    assert token not in grant.model_dump_json()
    assert grant.computer_id == snapshot.run.computer_id
    with pytest.raises(RuntimeAuthenticationError, match="does not allow"):
        service.authorize_connector(grant.id, token, "search_work")
    with pytest.raises(RuntimeAuthenticationError, match="outside its resource scope"):
        service.authorize_connector(
            grant.id,
            token,
            "read_source",
            {"repository": "acme/other"},
        )
    service.revoke_connector_grants(snapshot.run.id)
    with pytest.raises(RuntimeAuthenticationError, match="expired"):
        service.authorize_connector(grant.id, token, "read_source")


def test_connector_constraints_compare_non_ascii_values_in_constant_time(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    snapshot = service.create_task_run(
        task_id="unicode-constraint",
        title="Unicode constraint",
        prompt="Read one repository",
        project=None,
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
        standalone_computer_id=service.local_computer.id,
    )
    grant, token = service.issue_connector_grant(
        snapshot.run.id,
        "github",
        "work-account",
        ["read_source"],
        {"repository": "acme/dépôt"},
    )

    assert service.authorize_connector(grant.id, token, "read_source", {"repository": "acme/dépôt"}).use_count == 1
    with pytest.raises(RuntimeAuthenticationError, match="outside its resource scope"):
        service.authorize_connector(grant.id, token, "read_source", {"repository": "acme/depot"})


def test_legacy_unbound_connector_grants_load_but_fail_closed(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    snapshot = service.create_task_run(
        task_id="legacy-grant",
        title="Legacy grant",
        prompt="Read source",
        project=None,
        requested_resource_ids=None,
        computer_id=None,
        engine_id="codex",
        standalone_computer_id=service.local_computer.id,
    )
    token = "legacy-secret"
    grant = ConnectorGrant.model_validate({
        "id": "grant-legacy",
        "run_id": snapshot.run.id,
        "provider": "github",
        "connection_name": "work-account",
        "actions": ["read_source"],
        "token_hash": service._digest(token),
        "expires_at": utc_now() + timedelta(minutes=5),
    })
    service.store.save_grant(grant)

    assert grant.computer_id == "legacy-unbound"
    with pytest.raises(RuntimeAuthenticationError, match="another computer"):
        service.authorize_connector(grant.id, token, "read_source")
    audit = service.store.list_audit_events(snapshot.run.id)
    assert any(item.action == "connector.invoke" and item.outcome == "denied" for item in audit)
    assert all(token not in item.model_dump_json() for item in audit)


def test_agent_token_is_scoped_to_one_leased_run_epoch(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.create_task_run(
        task_id="agent-token-task",
        title="Agent token task",
        prompt="Run remotely",
        project=CodeProject(
            id="agent-token-project",
            name="Agent token project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    leased = service.acquire_lease(remote.id)
    assert leased is not None
    run, _ = leased
    token = service.issue_run_token(run.id)
    assert service.authenticate_run_token(run.id, remote.id, token).epoch == run.epoch
    assert token not in service.store.get_run_credential(run.id).model_dump_json()

    run.last_event_seq = 7
    run.checkpoint = {"phase": "running"}
    run.lease_expires_at = utc_now() - timedelta(seconds=1)
    service.store.save_run(run)
    service.expire_leases()
    expired = service.store.get_run(run.id)
    assert expired.epoch == 2
    assert expired.last_event_seq == 0
    assert expired.checkpoint == {}
    with pytest.raises(RuntimeAuthenticationError, match="failed"):
        service.authenticate_run_token(run.id, remote.id, token)


def test_runtime_commands_are_idempotent_retryable_and_acknowledged(tmp_path: Path) -> None:
    service = ControlPlaneService(tmp_path, capabilities())
    remote, _ = register(service)
    service.create_task_run(
        task_id="command-task",
        title="Command task",
        prompt="Start",
        project=CodeProject(
            id="command-project",
            name="Command project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.test/repo.git")],
        ),
        requested_resource_ids=None,
        computer_id=remote.id,
        engine_id="codex",
    )
    leased = service.acquire_lease(remote.id)
    assert leased is not None
    run, lease_id = leased
    first = service.queue_command(run.id, "start", {"prompt": "Build"}, "start-turn-1")
    duplicate = service.queue_command(run.id, "start", {"prompt": "Ignored"}, "start-turn-1")
    assert duplicate.id == first.id

    claimed = service.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert [item.id for item in claimed] == [first.id]
    assert service.claim_commands(run.id, remote.id, lease_id, run.epoch) == []

    claimed[0].claim_expires_at = utc_now() - timedelta(seconds=1)
    service.store.save_command(claimed[0])
    retried = service.claim_commands(run.id, remote.id, lease_id, run.epoch)
    assert retried[0].id == first.id
    assert retried[0].delivery_count == 2

    acknowledged, first_ack = service.acknowledge_command(run.id, first.id, remote.id, lease_id, run.epoch)
    assert acknowledged.acked_at is not None
    assert first_ack
    assert service.acknowledge_command(run.id, first.id, remote.id, lease_id, run.epoch) == (acknowledged, False)
    assert service.claim_commands(run.id, remote.id, lease_id, run.epoch) == []


def test_run_status_is_only_assigned_by_the_transition_table() -> None:
    import cowork.coding

    root = Path(cowork.coding.__file__).parent
    status_write = re.compile(
        r"\b(?P<assigned>\w+)\.status\s*=(?!=)"
        r"|setattr\((?P<setattr>\w+),\s*[\"']status[\"']"
        r"|\b(?P<copied>\w+)\.model_copy\(update=\{[^}]*[\"']status[\"']",
    )
    session_and_resource_writes = {
        "control_service.py": {"computer", "existing", "record", "workspace"},
        "remote_execution.py": {"current"},
        "run_recovery.py": {"workspace"},
        "service.py": {"session"},
        "service_turns.py": {"current"},
        "store.py": {"session"},
        "turns.py": {"session"},
    }
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        if path.name != "run_state.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in status_write.finditer(line)
        if (match.group("assigned") or match.group("setattr") or match.group("copied"))
        not in session_and_resource_writes.get(path.name, set())
    ]
    assert offenders == []
