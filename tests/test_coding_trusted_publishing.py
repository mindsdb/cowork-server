from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.coding.service_fakes import CREDS, FakeEngine, git, repository, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus, WorkspaceMode
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    DraftPullRequestSpec,
    ProjectCreateRequest,
    ProjectFolder,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the hooks under test are POSIX shell scripts")

DUMMY_CREDENTIAL = "cowork-dummy-credential"


def output(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def hook(path: Path, record: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nenv > '{record}'\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Integrations:
    def __init__(self) -> None:
        self.drafts: list[dict] = []
        self.pushes: list[str] = []

    def create_draft_pull_request(self, _project, **kwargs):
        self.drafts.append(kwargs)
        return "https://github.example/draft/1"

    def git_push_credentials(self, _project, repository_url, _connection_name):
        self.pushes.append(repository_url)
        return SimpleNamespace(
            remote_url=repository_url,
            environment={"COWORK_TEST_CREDENTIAL": DUMMY_CREDENTIAL},
        )


@pytest.fixture
def ready_task(tmp_path: Path):
    repo = repository(tmp_path)
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Publishing",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    task = service.create_session(
        SessionCreateRequest(workspace_mode=WorkspaceMode.worktree, project_id=project.id, prompt="Publish"),
        CREDS,
        "fake",
        "fake-model",
    )
    task = wait_for_status(service, task.id, SessionStatus.completed)
    workspace = Path(task.workspace_path)
    (workspace / "README.md").write_text("delivery\n", encoding="utf-8")
    service.commit(task.id, "Prepare delivery")
    return SimpleNamespace(service=service, task=task, repo=repo, remote=remote, workspace=workspace)


def confirm(item, **changes) -> DraftPullRequestSpec:
    values = {
        "folder_id": item.folder_id,
        "title": "Delivery",
        "head_revision": item.head_revision,
        "task_branch": item.task_branch,
        "remote_url": item.remote_url,
    }
    values.update(changes)
    return DraftPullRequestSpec(**values)


def test_publishing_runs_no_task_hook_and_pushes_the_confirmed_commit(ready_task, tmp_path: Path) -> None:
    record = tmp_path / "hook-env.txt"
    # Hook managers such as husky keep hooks in the task's own files.
    hook(ready_task.workspace / ".husky" / "pre-push", record)
    ready_task.service.commit(ready_task.task.id, "Add hooks")
    git(ready_task.repo, "config", "core.hooksPath", ".husky")
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    hook(ready_task.repo / ".git" / "hooks" / "pre-push", record)
    # Repository configuration must not redirect where the credential goes.
    # pushInsteadOf leaves the remote the person reviewed looking unchanged.
    decoy = tmp_path / "decoy.git"
    git(tmp_path, "init", "--bare", str(decoy))
    git(ready_task.repo, "config", f"url.{decoy}.pushInsteadOf", str(ready_task.remote))

    integrations = Integrations()
    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(title="Delivery", drafts=[confirm(item)], confirmed=True),
        integrations,  # type: ignore[arg-type]
    )

    assert records[0].status == "published", records[0].detail
    assert not record.exists()
    assert output(ready_task.remote, "rev-parse", f"refs/heads/{item.task_branch}") == item.head_revision
    assert output(decoy, "for-each-ref") == ""


def test_publishing_refuses_a_commit_made_after_confirmation(ready_task) -> None:
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    (ready_task.workspace / "README.md").write_text("after confirmation\n", encoding="utf-8")
    ready_task.service.commit(ready_task.task.id, "Later change")

    integrations = Integrations()
    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(title="Delivery", drafts=[confirm(item)], confirmed=True),
        integrations,  # type: ignore[arg-type]
    )

    assert records[0].status == "failed"
    assert "confirm again" in records[0].detail
    assert integrations.pushes == []
    assert output(ready_task.remote, "for-each-ref") == ""


def test_publishing_refuses_a_changed_destination(ready_task) -> None:
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    integrations = Integrations()
    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(
            title="Delivery",
            drafts=[confirm(item, task_branch="someone-elses-branch")],
            confirmed=True,
        ),
        integrations,  # type: ignore[arg-type]
    )

    assert records[0].status == "failed"
    assert integrations.pushes == []


def test_publishing_refuses_git_lfs_rather_than_pushing_pointer_files(ready_task) -> None:
    (ready_task.workspace / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    ready_task.service.commit(ready_task.task.id, "Track binaries with LFS")
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]

    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(title="Delivery", drafts=[confirm(item)], confirmed=True),
        Integrations(),  # type: ignore[arg-type]
    )

    assert records[0].status == "failed"
    assert "Git LFS" in records[0].detail
    assert output(ready_task.remote, "for-each-ref") == ""


def test_publishing_refuses_a_detached_confirmed_commit(ready_task) -> None:
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    git(ready_task.workspace, "checkout", "--detach", item.head_revision)
    plan = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    assert plan.status == "unavailable"
    assert plan.task_branch is None

    integrations = Integrations()
    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(title="Delivery", drafts=[confirm(item)], confirmed=True),
        integrations,  # type: ignore[arg-type]
    )
    assert records[0].status == "failed"
    assert integrations.pushes == []
    assert output(ready_task.remote, "for-each-ref") == ""


@pytest.mark.parametrize("already_published", [False, True])
def test_publishing_checks_lfs_pointers_in_outgoing_history(ready_task, already_published: bool) -> None:
    workspace = ready_task.workspace
    pointer = workspace / "asset.bin"
    attributes = workspace / ".gitattributes"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 64\n")
    attributes.write_text("*.bin filter=lfs -text\n")
    ready_task.service.commit(ready_task.task.id, "Track asset")
    pointer.unlink()
    attributes.unlink()
    ready_task.service.commit(ready_task.task.id, "Remove asset and tracking rule")
    if already_published:
        git(workspace, "push", "--no-verify", "origin", "HEAD:refs/heads/base")
    (workspace / "README.md").write_text("ready for delivery\n")
    ready_task.service.commit(ready_task.task.id, "Finish task")
    item = ready_task.service.delivery_plan(ready_task.task.id).items[0]
    integrations = Integrations()
    records = ready_task.service.create_draft_pull_requests(
        ready_task.task.id,
        DraftPullRequestRequest(title="Delivery", drafts=[confirm(item)], confirmed=True),
        integrations,  # type: ignore[arg-type]
    )
    if already_published:
        assert records[0].status == "published", records[0].detail
        assert output(ready_task.remote, "rev-parse", item.task_branch) == item.head_revision
    else:
        assert records[0].status == "failed"
        assert "Git LFS" in records[0].detail
        assert integrations.drafts == []
        assert output(ready_task.remote, "for-each-ref") == ""
