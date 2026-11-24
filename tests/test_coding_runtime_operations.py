from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cowork.coding.contracts import PermissionMode, TaskWorkspace, WorkspaceKind
from cowork.coding.control_models import CodeTask, RunStatus, RuntimeCommand, TaskRun
from cowork.coding.project_models import (
    CodeProject,
    LocalFolderResource,
    RepositoryResource,
)
from cowork.coding.project_workspaces import (
    PreparedProjectWorkspace,
    ProjectWorkspaceManager,
)
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.runtime_operations import RuntimeWorkspaceOperations
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease
from cowork.coding.workspace import WorkspaceManager


def lease(project: CodeProject | None = None) -> RuntimeLease:
    return RuntimeLease(
        task=CodeTask(id="task-ops", title="Ops task", prompt="Build"),
        run=TaskRun(id="run-ops", task_id="task-ops", computer_id="remote", status=RunStatus.running, lease_id="lease-ops"),
        lease_id="lease-ops",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=project or CodeProject(
            id="ops-project",
            name="Ops project",
            resources=[
                RepositoryResource(id="repo", name="Repo", source_url="https://example.invalid/repo.git"),
                RepositoryResource(id="docs", name="Docs", source_url="https://example.invalid/docs.git"),
            ],
        ),
        execution=RuntimeExecutionConfig(engine_id="fake", model="fake-model", permission_mode=PermissionMode.workspace),
    )


def operation(index: int, payload: dict[str, object]) -> RuntimeCommand:
    return RuntimeCommand(id=f"operation-{index}", run_id="run-ops", epoch=1, kind="operation", payload=payload)


def test_remote_validation_honors_a_read_only_change_during_a_lease(tmp_path: Path) -> None:
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"))
    seen = []
    manager.run_commands = lambda *args, policy=None, **kwargs: seen.append(policy) or []
    prepared = SimpleNamespace(workspaces=[], ports={})
    operations = RuntimeWorkspaceOperations(lease(), manager, prepared, object())

    result, error = operations.execute(operation(1, {"operation": "validate_v2", "permission_mode": "read_only", "network_access": False}))

    assert error is None
    assert result == {"items": []}
    assert seen[0].read_only


def test_refresh_project_hands_validate_the_new_commands_without_touching_workspaces() -> None:
    seen: list[CodeProject] = []
    manager = SimpleNamespace(
        run_commands=lambda project, workspaces, phase, ports, policy=None: seen.append(project) or [],
        check_policy=lambda *_args: None,
    )
    prepared = SimpleNamespace(workspaces=[SimpleNamespace(folder_id="repo")], ports={})
    operations = RuntimeWorkspaceOperations(lease(), manager, prepared, object())
    check = {"id": "check", "label": "Check", "argv": ["npm", "test"], "phase": "validate"}

    assert operations.execute(operation(0, {"operation": "validate_v2", "permission_mode": "workspace", "network_access": False})) == ({"items": []}, None)
    assert seen[0].resources[0].commands == []

    result, error = operations.execute(operation(1, {"operation": "refresh_project", "commands": {"repo": [check], "gone": [check]}}))

    assert (result, error) == ({"resources": 1}, None)
    operations.execute(operation(2, {"operation": "validate_v2", "permission_mode": "workspace", "network_access": False}))
    assert [command.id for command in seen[1].resources[0].commands] == ["check"]
    assert seen[1].resources[1].commands == []
    assert [resource.id for resource in seen[1].resources] == ["repo", "docs"]
    assert operations.prepared is prepared


def test_refresh_project_then_validate_runs_the_new_commands_through_the_real_runner(tmp_path: Path) -> None:
    folder = tmp_path / "app"
    folder.mkdir()
    project = CodeProject(
        id="ops-project",
        name="Ops project",
        resources=[LocalFolderResource(id="app", name="App", path=str(folder), computer_id="remote")],
    )
    workspace = TaskWorkspace(
        folder_id="app",
        folder_name="App",
        source_path=str(folder),
        workspace_path=str(folder),
        workspace_kind=WorkspaceKind.local_copy,
        source_dirty=False,
    )
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"))
    prepared = PreparedProjectWorkspace(primary=workspace, workspaces=(workspace,), ports={})
    operations = RuntimeWorkspaceOperations(lease(project), manager, prepared, object())
    check = {"id": "check", "label": "Check", "argv": [sys.executable, "-c", "print('late')"], "phase": "validate"}

    assert operations.execute(operation(0, {"operation": "validate_v2", "permission_mode": "workspace", "network_access": False})) == ({"items": []}, None)
    assert operations.execute(operation(1, {"operation": "refresh_project", "commands": {"app": [check]}})) == ({"resources": 1}, None)

    result, error = operations.execute(operation(2, {"operation": "validate_v2", "permission_mode": "workspace", "network_access": False}))

    assert error is None
    # On Linux the sandbox launcher adds a warning when its home is under a
    # temporary folder, as it is in tests; the command's own output comes first.
    assert [(item["label"], item["return_code"], item["output"].splitlines()[0]) for item in result["items"]] == [("Check", 0, "late")]


@pytest.mark.parametrize("payload", [{}, {"commands": ["not", "a", "map"]}])
def test_refresh_project_rejects_a_malformed_payload(payload: dict[str, object]) -> None:
    operations = RuntimeWorkspaceOperations(lease(), SimpleNamespace(), SimpleNamespace(workspaces=[]), object())

    result, error = operations.execute(operation(0, {"operation": "refresh_project", **payload}))

    assert result is None
    assert error == "A project command refresh needs a commands map"


@pytest.mark.parametrize("name", ["commit", "validate"])
def test_legacy_control_plane_cannot_start_a_check(name: str) -> None:
    manager = Mock()
    ops = RuntimeWorkspaceOperations(lease(), manager, SimpleNamespace(workspaces=[], ports={}), object())

    result, error = ops.execute(operation(0, {"operation": name, "message": "Commit"}))

    assert result is None and "Update the Cowork server" in error
    assert not manager.mock_calls


@pytest.mark.parametrize("name", ["commit_v2", "validate_v2"])
@pytest.mark.parametrize("policy", [{"permission_mode": "workspace"}, {"network_access": False}])
def test_versioned_checks_require_the_full_current_task_policy(name: str, policy: dict) -> None:
    manager = Mock()
    ops = RuntimeWorkspaceOperations(lease(), manager, SimpleNamespace(workspaces=[], ports={}), object())

    result, error = ops.execute(operation(0, {"operation": name, "message": "Commit", **policy}))

    assert result is None and "current task permissions" in error
    assert not manager.mock_calls


@pytest.mark.parametrize("name", ["commit", "validate"])
def test_control_plane_refuses_to_fall_back_to_an_older_worker(name: str) -> None:
    control = Mock()
    control.store.get_run.return_value.status = RunStatus.ready
    control.wait_for_command.return_value.error = "The selected computer does not support that operation"
    remote = RemoteExecutionCoordinator(control, Mock(), Mock(), Mock())
    session = SimpleNamespace(run_id="run-ops", effective_permission_mode=PermissionMode.read_only, network_access=False)

    with pytest.raises(RuntimeError, match="Update Cowork on the selected execution computer"):
        remote.operation(session, name, {"message": "Commit"})

    control.queue_command.assert_called_once()
    assert control.queue_command.call_args.args[2] == {
        "operation": f"{name}_v2", "message": "Commit", "permission_mode": "read_only", "network_access": False,
    }
