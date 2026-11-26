from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tests.support.coding.service_fakes import git, repository

from cowork.coding.contracts import PermissionMode, TaskWorkspace, WorkspaceKind
from cowork.coding.control_models import CodeTask, RunStatus, RuntimeCommand, TaskRun
from cowork.coding.project_models import (
    CodeProject,
    DeliveryPlanItem,
    DraftPullRequestRequest,
    DraftPullRequestSpec,
    LocalFolderResource,
    RepositoryResource,
)
from cowork.coding.project_workspaces import (
    PreparedProjectWorkspace,
    ProjectWorkspaceManager,
)
from cowork.coding.runtime_operations import RuntimeWorkspaceOperations
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease
from cowork.coding.task_delivery import TaskDeliveryService
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

    assert operations.execute(operation(0, {"operation": "validate"})) == ({"items": []}, None)
    assert operations.execute(operation(1, {"operation": "refresh_project", "commands": {"app": [check]}})) == ({"resources": 1}, None)

    result, error = operations.execute(operation(2, {"operation": "validate"}))

    assert error is None
    assert [(item["label"], item["return_code"], item["output"]) for item in result["items"]] == [("Check", 0, "late")]


@pytest.mark.parametrize("payload", [{}, {"commands": ["not", "a", "map"]}])
def test_refresh_project_rejects_a_malformed_payload(payload: dict[str, object]) -> None:
    operations = RuntimeWorkspaceOperations(lease(), SimpleNamespace(), SimpleNamespace(workspaces=[]), object())

    result, error = operations.execute(operation(0, {"operation": "refresh_project", **payload}))

    assert result is None
    assert error == "A project command refresh needs a commands map"


def test_draft_request_distinguishes_an_empty_selection_from_an_omitted_selection() -> None:
    assert DraftPullRequestRequest(title="Publish", confirmed=True).drafts == []
    with pytest.raises(ValidationError):
        DraftPullRequestRequest(title="Publish", confirmed=True, drafts=[])


@pytest.fixture
def publish_operation(tmp_path: Path):
    repo = repository(tmp_path)
    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"))
    workspace = TaskWorkspace(
        folder_id="repo", folder_name="Repo", source_path=str(repo),
        workspace_path=str(repo), workspace_kind=WorkspaceKind.git_worktree,
    )
    prepared = PreparedProjectWorkspace(primary=workspace, workspaces=(workspace,), ports={})
    operations = RuntimeWorkspaceOperations(lease(), manager, prepared, object())
    runner = manager.workspaces.git
    payload = {
        "operation": "push_confirmed_commit", "folder_id": "repo",
        "revision": runner.run(repo, "rev-parse", "HEAD").stdout.strip(),
        "task_branch": runner.run(repo, "branch", "--show-current").stdout.strip(),
        "remote_url": str(remote),
    }
    return operations, payload, repo, remote, runner


def test_remote_push_publishes_the_confirmed_commit_to_the_confirmed_repository(publish_operation, tmp_path: Path) -> None:
    operations, payload, repo, remote, runner = publish_operation
    secondary = tmp_path / "secondary.git"
    git(tmp_path, "init", "--bare", str(secondary))
    git(repo, "remote", "set-url", "--push", "origin", str(secondary))

    item = DeliveryPlanItem(
        folder_id="repo", folder_name="Repo", workspace_path=str(repo), base_branch="main",
        task_branch=payload["task_branch"], head_revision=payload["revision"], remote_url=str(remote), status="ready",
    )
    session = SimpleNamespace(id="task", task_id=None, deliveries=[])
    remote_service = Mock()
    remote_service.is_remote.return_value = True

    def remote_operation(_session, name, body, **_kwargs):
        if name == "delivery_plan":
            return {"items": [item.model_dump(mode="json")]}
        result, error = operations.execute(operation(0, {"operation": name, **body}))
        if error:
            raise RuntimeError(error)
        return result

    remote_service.operation.side_effect = remote_operation
    store = Mock()
    store.load_session.return_value = session
    delivery = TaskDeliveryService(store, Mock(), lambda _: operations.project, Mock(), remote_service, lambda _: session)
    integrations = Mock()
    integrations.create_draft_pull_request.return_value = "https://github.example/pull/1"
    request = DraftPullRequestRequest(
        title="Publish",
        confirmed=True,
        drafts=[DraftPullRequestSpec(
            folder_id="repo", title="Publish", task_branch=item.task_branch,
            remote_url=item.remote_url, head_revision=item.head_revision,
        )],
    )
    [record] = delivery.create_drafts(session.id, request, integrations)

    assert record.status == "published", record.detail
    integrations.create_draft_pull_request.assert_called_once()
    assert integrations.create_draft_pull_request.call_args.kwargs["repository_url"] == str(remote)
    assert runner.run(remote, "rev-parse", payload["task_branch"]).stdout.strip() == payload["revision"]
    assert runner.run(secondary, "for-each-ref").stdout == ""


@pytest.mark.parametrize("change", ["branch", "remote", "revision"])
def test_remote_push_refuses_changes_after_confirmation(publish_operation, tmp_path: Path, change: str) -> None:
    operations, payload, repo, remote, runner = publish_operation
    if change == "branch":
        git(repo, "switch", "-c", "another-branch")
    elif change == "remote":
        git(repo, "remote", "set-url", "origin", str(tmp_path / "other.git"))
    else:
        git(repo, "commit", "--allow-empty", "-m", "Later commit")

    result, error = operations.execute(operation(0, payload))

    assert result is None
    assert "confirm again" in error
    assert runner.run(remote, "for-each-ref").stdout == ""


def test_remote_push_refuses_lfs_before_publishing(publish_operation) -> None:
    operations, payload, repo, remote, runner = publish_operation
    (repo / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    git(repo, "add", ".gitattributes")
    git(repo, "commit", "-m", "Track LFS")
    payload["revision"] = runner.run(repo, "rev-parse", "HEAD").stdout.strip()

    result, error = operations.execute(operation(0, payload))

    assert result is None
    assert "Git LFS" in error
    assert runner.run(remote, "for-each-ref").stdout == ""
