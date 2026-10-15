"""A released workspace comes back exactly, or stays put and says why."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest
from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus, WorkspaceKind
from cowork.coding.workspace import WorkspaceError, WorkspaceManager
from cowork.coding.workspace_retention import RetentionPolicy
from cowork.common.settings.app_settings import CodingSettings


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def released_worktree(tmp_path: Path, key: str = "task") -> tuple[WorkspaceManager, Path, Path]:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    worktree = manager.prepare(key, str(repo), allow_direct_folder=False).workspace_path
    return manager, repo, worktree


def round_trip(manager: WorkspaceManager, repo: Path, worktree: Path, key: str = "task") -> None:
    assert manager.release(key, str(repo), str(worktree), WorkspaceKind.git_worktree) is True, manager.release_refusal(key)
    assert not worktree.exists()
    manager.restore(key, str(repo), str(worktree), WorkspaceKind.git_worktree)


def test_a_staged_only_change_survives_a_round_trip(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    (worktree / "README.md").write_text("staged\n", encoding="utf-8")
    git(worktree, "add", "README.md")

    round_trip(manager, repo, worktree)

    assert git(worktree, "diff", "--cached", "--name-only") == "README.md"
    assert git(worktree, "diff", "--name-only") == ""
    assert git(worktree, "show", ":README.md") == "staged"


def test_staged_and_unstaged_versions_of_a_file_stay_distinct(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    (worktree / "README.md").write_text("staged\n", encoding="utf-8")
    (worktree / "added.txt").write_text("added\n", encoding="utf-8")
    git(worktree, "add", "README.md", "added.txt")
    (worktree / "README.md").write_text("unstaged\n", encoding="utf-8")
    (worktree / "added.txt").unlink()
    (worktree / "loose.txt").write_text("untracked\n", encoding="utf-8")
    status_before = git(worktree, "status", "--porcelain")

    round_trip(manager, repo, worktree)

    assert git(worktree, "show", ":README.md") == "staged"
    assert (worktree / "README.md").read_text(encoding="utf-8") == "unstaged\n"
    assert not (worktree / "added.txt").exists()
    assert (worktree / "loose.txt").read_text(encoding="utf-8") == "untracked\n"
    assert git(worktree, "status", "--porcelain") == status_before


def test_a_task_on_a_named_branch_comes_back_on_it_and_new_commits_advance_it(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    git(worktree, "switch", "-c", "cowork/fix")
    (worktree / "fix.txt").write_text("fix\n", encoding="utf-8")
    git(worktree, "add", "fix.txt")
    git(worktree, "commit", "-m", "fix")
    reviewed = git(worktree, "rev-parse", "HEAD")

    round_trip(manager, repo, worktree)

    assert git(worktree, "symbolic-ref", "--short", "HEAD") == "cowork/fix"
    assert git(worktree, "rev-parse", "HEAD") == reviewed
    (worktree / "more.txt").write_text("more\n", encoding="utf-8")
    git(worktree, "add", "more.txt")
    git(worktree, "commit", "-m", "more")
    assert git(repo, "rev-parse", "cowork/fix") == git(worktree, "rev-parse", "HEAD")
    assert git(repo, "rev-parse", "cowork/fix~1") == reviewed


def test_a_branch_that_moved_while_released_is_left_alone_and_restore_can_retry(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    git(worktree, "switch", "-c", "cowork/fix")
    (worktree / "README.md").write_text("task\n", encoding="utf-8")
    assert manager.release("task", str(repo), str(worktree), WorkspaceKind.git_worktree) is True
    task_head = git(repo, "rev-parse", "cowork/fix")
    git(repo, "commit", "--allow-empty", "-m", "elsewhere")
    git(repo, "branch", "-f", "cowork/fix", "HEAD")
    moved = git(repo, "rev-parse", "cowork/fix")
    assert moved != task_head

    with pytest.raises(WorkspaceError, match="has moved"):
        manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)

    assert git(repo, "rev-parse", "cowork/fix") == moved
    assert manager.is_released("task", WorkspaceKind.git_worktree)
    git(repo, "branch", "-f", "cowork/fix", task_head)
    manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)
    assert (worktree / "README.md").read_text(encoding="utf-8") == "task\n"


def test_a_branch_checked_out_elsewhere_is_not_touched(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    git(worktree, "switch", "-c", "cowork/fix")
    assert manager.release("task", str(repo), str(worktree), WorkspaceKind.git_worktree) is True
    git(repo, "switch", "cowork/fix")
    (repo / "README.md").write_text("user's own edit\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="checked out in another folder"):
        manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)

    assert (repo / "README.md").read_text(encoding="utf-8") == "user's own edit\n"
    assert git(repo, "symbolic-ref", "--short", "HEAD") == "cowork/fix"
    assert manager.is_released("task", WorkspaceKind.git_worktree)


def test_a_deleted_task_branch_is_recreated_where_the_task_left_it(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    git(worktree, "switch", "-c", "cowork/fix")
    head = git(worktree, "rev-parse", "HEAD")
    assert manager.release("task", str(repo), str(worktree), WorkspaceKind.git_worktree) is True
    git(repo, "branch", "-D", "cowork/fix")

    manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)

    assert git(worktree, "symbolic-ref", "--short", "HEAD") == "cowork/fix"
    assert git(worktree, "rev-parse", "HEAD") == head


def test_a_missing_source_keeps_the_saved_state_for_later(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    (worktree / "README.md").write_text("task\n", encoding="utf-8")
    assert manager.release("task", str(repo), str(worktree), WorkspaceKind.git_worktree) is True
    moved = repo.with_name("moved")
    repo.rename(moved)

    with pytest.raises(WorkspaceError, match="unavailable"):
        manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)

    moved.rename(repo)
    manager.restore("task", str(repo), str(worktree), WorkspaceKind.git_worktree)
    assert (worktree / "README.md").read_text(encoding="utf-8") == "task\n"


def test_ignored_files_keep_the_workspace_unless_they_are_caches(tmp_path: Path) -> None:
    manager, repo, worktree = released_worktree(tmp_path)
    (worktree / ".gitignore").write_text(".env\n__pycache__/\n", encoding="utf-8")
    (worktree / "__pycache__").mkdir()
    (worktree / "__pycache__" / "a.pyc").write_bytes(b"cache")
    (worktree / ".env").write_text("TOKEN=1\n", encoding="utf-8")

    assert manager.release("task", str(repo), str(worktree), WorkspaceKind.git_worktree) is False
    assert ".env" in (manager.release_refusal("task") or "")
    assert (worktree / ".env").is_file()

    (worktree / ".env").unlink()
    round_trip(manager, repo, worktree)
    assert (worktree / ".gitignore").is_file()
    assert manager.release_refusal("task") is None


def test_a_folder_copy_whose_inner_checkout_has_staged_changes_is_kept(tmp_path: Path) -> None:
    folder = tmp_path / "repos"
    inner = folder / "app"
    inner.mkdir(parents=True)
    git(inner, "init")
    git(inner, "config", "user.email", "cowork@example.invalid")
    git(inner, "config", "user.name", "Cowork Test")
    (inner / "a.txt").write_text("a\n", encoding="utf-8")
    git(inner, "add", ".")
    git(inner, "commit", "-m", "a")
    manager = WorkspaceManager(tmp_path / "coding")
    workspace = manager.prepare("task", str(folder), allow_direct_folder=True).workspace_path
    (workspace / "app" / "a.txt").write_text("staged\n", encoding="utf-8")
    git(workspace / "app", "add", "a.txt")

    assert manager.release("task", str(folder), str(workspace), WorkspaceKind.local_copy) is False
    assert "staged" in (manager.release_refusal("task") or "")
    assert (workspace / "app" / "a.txt").read_text(encoding="utf-8") == "staged\n"


def test_releasing_idle_tasks_is_off_unless_turned_on(tmp_path: Path) -> None:
    assert CodingSettings().workspace_retention_enabled is False
    assert RetentionPolicy().enabled is False
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt="Go"), CREDS, "fake", "fake-model")
    wait_for_status(service, created.id, SessionStatus.completed)
    service.store.update_session(
        created.id,
        lambda current: setattr(current, "updated_at", current.updated_at.replace(year=2000)),
        touch_updated_at=False,
    )

    assert service.retention.run_policy(RetentionPolicy(keep_count=0)) == []
    assert service.run_workspace_retention() == []
    assert Path(service.get_session(created.id).workspace_path).is_dir()


def test_a_kept_workspace_says_why_and_a_failed_rebuild_is_reported(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt="Go"), CREDS, "fake", "fake-model")
    session = wait_for_status(service, created.id, SessionStatus.completed)
    workspace = Path(session.workspace_path)
    (workspace / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (workspace / "secret.txt").write_text("keep me\n", encoding="utf-8")

    archived = service.set_archived(session.id, True)

    assert archived.workspace_released_at is None
    assert "secret.txt" in (archived.workspace_kept_reason or "")
    assert (workspace / "secret.txt").is_file()

    (workspace / "secret.txt").unlink()
    service.set_archived(session.id, False)
    assert service.retention.release(session.id) is True
    assert service.get_session(session.id).workspace_kept_reason is None
    moved = repo.with_name("moved")
    repo.rename(moved)
    with pytest.raises(WorkspaceError):
        service.ensure_workspace(session.id)
    failed = service.get_session(session.id)
    assert failed.workspace_released_at is not None
    assert "unavailable" in (failed.workspace_restore_error or "")

    moved.rename(repo)
    service.ensure_workspace(session.id)
    assert service.get_session(session.id).workspace_restore_error is None


def test_concurrent_restores_after_a_failure_never_report_success(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt="Go"), CREDS, "fake", "fake-model")
    session = wait_for_status(service, created.id, SessionStatus.completed)
    assert service.retention.release(session.id) is True
    moved = repo.with_name("moved")
    repo.rename(moved)
    outcomes: list[str] = []

    def use() -> None:
        try:
            service.ensure_workspace(session.id)
            outcomes.append("ok")
        except Exception:  # noqa: BLE001 - recorded for the assertion below.
            outcomes.append("failed")

    threads = [threading.Thread(target=use) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert outcomes == ["failed"] * 3
    assert service.get_session(session.id).workspace_released_at is not None
