from __future__ import annotations

from types import SimpleNamespace

import pytest

from cowork.coding.contracts import PermissionMode
from cowork.coding.control_models import CodeTask, RunStatus, RuntimeCommand, TaskRun
from cowork.coding.project_models import CodeProject, RepositoryResource
from cowork.coding.runtime_operations import RuntimeWorkspaceOperations
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease


def lease() -> RuntimeLease:
    return RuntimeLease(
        task=CodeTask(id="task-ops", title="Ops task", prompt="Build"),
        run=TaskRun(id="run-ops", task_id="task-ops", computer_id="remote", status=RunStatus.running, lease_id="lease-ops"),
        lease_id="lease-ops",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=CodeProject(
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


def test_refresh_project_hands_validate_the_new_commands_without_touching_workspaces() -> None:
    seen: list[CodeProject] = []
    manager = SimpleNamespace(run_commands=lambda project, workspaces, phase, ports: seen.append(project) or [])
    prepared = SimpleNamespace(workspaces=[SimpleNamespace(folder_id="repo")], ports={})
    operations = RuntimeWorkspaceOperations(lease(), manager, prepared, object())
    check = {"id": "check", "label": "Check", "argv": ["npm", "test"], "phase": "validate"}

    assert operations.execute(operation(0, {"operation": "validate"})) == ({"items": []}, None)
    assert seen[0].resources[0].commands == []

    result, error = operations.execute(operation(1, {"operation": "refresh_project", "commands": {"repo": [check], "gone": [check]}}))

    assert (result, error) == ({"resources": 1}, None)
    operations.execute(operation(2, {"operation": "validate"}))
    assert [command.id for command in seen[1].resources[0].commands] == ["check"]
    assert seen[1].resources[1].commands == []
    assert [resource.id for resource in seen[1].resources] == ["repo", "docs"]
    assert operations.prepared is prepared


@pytest.mark.parametrize("payload", [{}, {"commands": ["not", "a", "map"]}])
def test_refresh_project_rejects_a_malformed_payload(payload: dict[str, object]) -> None:
    operations = RuntimeWorkspaceOperations(lease(), SimpleNamespace(), SimpleNamespace(workspaces=[]), object())

    result, error = operations.execute(operation(0, {"operation": "refresh_project", **payload}))

    assert result is None
    assert error == "A project command refresh needs a commands map"
