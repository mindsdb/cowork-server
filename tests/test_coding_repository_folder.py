"""A task on a folder that holds several Git repositories, without being one."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cowork.coding import source_changes
from cowork.coding.contracts import WorkspaceKind
from cowork.coding.workspace import GitUnavailableError, WorkspaceError, WorkspaceManager


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def make_repository(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    git(path, "init")
    git(path, "config", "user.email", "cowork@example.invalid")
    git(path, "config", "user.name", "Cowork Test")
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content, encoding="utf-8")
    (path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-m", "base")
    return path


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """monorepo-style folder: repos at depth 1 and 2, loose files, dependency folders."""
    root = tmp_path / "monorepo"
    make_repository(root / "app", {"src/main.py": "print('app')\n"})
    make_repository(root / "services" / "api", {"server.py": "serve()\n"})
    (root / "services" / "README.md").write_text("services\n", encoding="utf-8")
    (root / "plan.md").write_text("plan\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "left-pad" / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    (root / "app" / "node_modules").mkdir()
    (root / "app" / "node_modules" / "dep.js").write_text("ignored\n", encoding="utf-8")
    # Uncommitted work in a repository carries over into the task.
    (root / "app" / "src" / "main.py").write_text("print('dirty')\n", encoding="utf-8")
    (root / "app" / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    return root


def worktrees(repository: Path) -> list[str]:
    return [line for line in git(repository, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")]


def test_inner_repositories_become_worktrees_inside_the_folder_copy(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")

    prepared = manager.prepare("task", str(folder), allow_direct_folder=True)
    workspace = prepared.workspace_path

    assert prepared.kind == WorkspaceKind.local_copy
    assert (workspace / "app" / ".git").is_file()
    assert (workspace / "services" / "api" / ".git").is_file()
    assert (workspace / "app" / "src" / "main.py").read_text(encoding="utf-8") == "print('dirty')\n"
    assert (workspace / "app" / "untracked.txt").read_text(encoding="utf-8") == "untracked\n"
    assert (workspace / "plan.md").is_file()
    assert (workspace / "services" / "README.md").is_file()
    assert (workspace / "docs" / "guide.md").is_file()
    # Dependencies are neither copied at the folder level nor checked out from a repo.
    assert not (workspace / "node_modules").exists()
    assert not (workspace / "app" / "node_modules").exists()
    assert manager.diff(str(workspace), None) == []
    baseline = manager.local_copies._baseline_for(workspace)
    assert not (baseline / "app" / ".git").exists()


def test_changes_in_repositories_and_loose_files_review_and_apply_back(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    (workspace / "services" / "api" / "server.py").write_text("serve(port=8080)\n", encoding="utf-8")
    (workspace / "plan.md").write_text("plan v2\n", encoding="utf-8")

    changed = {item.path: item.status for item in manager.diff(str(workspace), None)}
    manager.apply_to_source("task", str(folder), str(workspace), None)

    assert changed == {"services/api/server.py": "M", "plan.md": "M"}
    assert (folder / "services" / "api" / "server.py").read_text(encoding="utf-8") == "serve(port=8080)\n"
    assert (folder / "plan.md").read_text(encoding="utf-8") == "plan v2\n"
    # The user's own uncommitted edit in another repository is untouched.
    assert (folder / "app" / "src" / "main.py").read_text(encoding="utf-8") == "print('dirty')\n"


def test_cleanup_and_release_forget_only_their_own_worktrees(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    # Another tool's worktree whose folder was deleted: Cowork must leave its
    # registration alone rather than prune the user's repository.
    stale = tmp_path / "someone-elses-worktree"
    git(folder / "app", "worktree", "add", "--detach", str(stale))
    subprocess.run(["rm", "-rf", str(stale)], check=True)
    first = manager.prepare("first", str(folder), allow_direct_folder=True).workspace_path
    second = manager.prepare("second", str(folder), allow_direct_folder=True).workspace_path
    assert len(worktrees(folder / "app")) == 4

    manager.cleanup("first", str(folder), str(first), WorkspaceKind.local_copy, None)
    assert manager.release("second", str(folder), str(second), WorkspaceKind.local_copy) is True

    assert worktrees(folder / "app") == [f"worktree {(folder / 'app').resolve()}", f"worktree {stale.resolve()}"]
    assert len(worktrees(folder / "services" / "api")) == 1


def test_a_released_repository_folder_restores_as_worktrees_with_its_changes(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    (workspace / "app" / "src" / "main.py").write_text("print('task')\n", encoding="utf-8")
    (workspace / "docs" / "new.md").write_text("new\n", encoding="utf-8")
    before = {(item.path, item.status) for item in manager.diff(str(workspace), None)}

    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is True
    manager.restore("task", str(folder), str(workspace), WorkspaceKind.local_copy)

    assert (workspace / "app" / ".git").is_file()
    assert {(item.path, item.status) for item in manager.diff(str(workspace), None)} == before
    assert before == {("app/src/main.py", "M"), ("docs/new.md", "A")}


def test_commits_in_an_inner_worktree_survive_release_and_restore(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    api = workspace / "services" / "api"
    (api / "server.py").write_text("serve(port=8080)\n", encoding="utf-8")
    git(api, "commit", "-qam", "task commit")
    head = git(api, "rev-parse", "HEAD")
    ref = "refs/cowork/released-inner/task/services/api"

    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is True
    assert git(folder / "services" / "api", "rev-parse", ref) == head

    manager.restore("task", str(folder), str(workspace), WorkspaceKind.local_copy)

    assert git(api, "rev-parse", "HEAD") == head
    assert git(api, "status", "--porcelain") == ""
    assert {item.path for item in manager.diff(str(workspace), None)} == {"services/api/server.py"}
    assert git(folder / "services" / "api", "for-each-ref", "refs/cowork") == ""


def test_deleting_a_released_repository_folder_forgets_its_pinned_commits(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is True
    assert git(folder / "app", "for-each-ref", "refs/cowork")

    manager.discard_release("task", str(folder))

    assert git(folder / "app", "for-each-ref", "refs/cowork") == ""
    assert git(folder / "services" / "api", "for-each-ref", "refs/cowork") == ""


def test_a_repository_folder_is_kept_when_a_checkout_holds_a_nested_repository(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    git(workspace / "app" / "src", "init", "-q")

    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is False
    assert (workspace / "app" / "src" / ".git").is_dir()
    assert git(folder / "app", "for-each-ref", "refs/cowork") == ""


def test_a_fork_gets_its_own_worktrees_and_inherits_the_parent_changes(tmp_path: Path, folder: Path) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    parent = manager.prepare("parent", str(folder), allow_direct_folder=True).workspace_path
    (parent / "services" / "api" / "server.py").write_text("forked()\n", encoding="utf-8")

    child = manager.fork("child", str(folder), str(parent), WorkspaceKind.local_copy, None).workspace_path

    assert git(child / "services" / "api", "rev-parse", "--show-toplevel") == str(
        (child / "services" / "api").resolve()
    )
    assert {item.path for item in manager.diff(str(child), None)} == {"services/api/server.py"}
    (child / "plan.md").write_text("child only\n", encoding="utf-8")
    assert (parent / "plan.md").read_text(encoding="utf-8") == "plan\n"


def test_a_repository_whose_changes_cannot_be_carried_is_copied_instead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folder: Path
) -> None:
    def refuse(*_args, **_kwargs) -> None:
        raise WorkspaceError("Local changes are too large to copy safely")

    monkeypatch.setattr(source_changes, "copy_source_changes", refuse)
    manager = WorkspaceManager(tmp_path / "coding")

    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path

    assert (workspace / "app" / ".git").is_dir()
    assert (workspace / "app" / "src" / "main.py").read_text(encoding="utf-8") == "print('dirty')\n"
    assert (workspace / "services" / "api" / ".git").is_file()
    assert len(worktrees(folder / "app")) == 1


def test_without_git_a_repository_folder_is_copied_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folder: Path
) -> None:
    manager = WorkspaceManager(tmp_path / "coding")

    def no_git(*_args, **_kwargs):
        raise GitUnavailableError("Git is not installed or is not available on PATH")

    monkeypatch.setattr(manager.git, "run", no_git)

    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path

    assert (workspace / "app" / ".git").is_dir()
    assert (workspace / "node_modules").is_dir()


def test_a_repository_folder_with_a_copied_repository_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folder: Path
) -> None:
    def refuse(*_args, **_kwargs) -> None:
        raise WorkspaceError("Local changes are too large to copy safely")

    monkeypatch.setattr(source_changes, "copy_source_changes", refuse)
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    assert (workspace / "app" / ".git").is_dir()

    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is False
    assert workspace.is_dir()
