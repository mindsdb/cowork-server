"""Deleting a task returns at once; its managed folders are removed in the background."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from tests.support.coding.service_fakes import CREDS, FakeEngine, copy_folders_without_git, repository, service_with, wait_for_status

from cowork.coding import pending_removals as pending_module
from cowork.coding.contracts import SessionCreateRequest, SessionStatus, WorkspaceKind, WorkspaceMode
from cowork.coding.pending_removals import PendingRemovals
from cowork.coding.workspace import WorkspaceManager


def create(service, path: Path, mode: WorkspaceMode = WorkspaceMode.worktree):
    session_id = service.create_session(
        SessionCreateRequest(path=str(path), prompt="Go", workspace_mode=mode), CREDS, "fake", "fake-model",
    ).id
    return wait_for_status(service, session_id, SessionStatus.completed)


def worktree_list(repo: Path) -> str:
    return subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def blocked_remover(monkeypatch: pytest.MonkeyPatch):
    """Hold background removal until the test releases it."""
    release = threading.Event()
    started = threading.Event()
    remove = pending_module.force_remove

    def blocked(path: Path) -> None:
        started.set()
        release.wait(10)
        remove(path)

    monkeypatch.setattr(pending_module, "force_remove", blocked)
    yield started, release
    release.set()


def test_deleting_a_folder_copy_returns_before_its_files_are_removed(tmp_path: Path, blocked_remover) -> None:
    started, release = blocked_remover
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "big.txt").write_text("data\n", encoding="utf-8")
    service = service_with(tmp_path, FakeEngine())
    copy_folders_without_git(service.workspaces)
    session = create(service, folder)
    assert session.workspace_kind == WorkspaceKind.local_copy
    (Path(session.workspace_path) / "edit.txt").write_text("unsaved\n", encoding="utf-8")
    pending = service.workspaces.pending_removals

    service.delete_session(session.id)

    assert started.wait(30)
    assert session.id not in {item.id for item in service.store.list_sessions()}
    assert not Path(session.workspace_path).exists()
    assert any(pending.root.iterdir())
    release.set()
    assert pending.wait(30)
    assert not any(pending.root.iterdir())
    assert sorted(path.name for path in folder.iterdir()) == ["big.txt"]


def test_deleting_a_worktree_task_drops_its_git_registration(tmp_path: Path, blocked_remover) -> None:
    _started, release = blocked_remover
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session = create(service, repo)
    assert session.workspace_kind == WorkspaceKind.git_worktree
    assert session.workspace_path in worktree_list(repo)

    service.delete_session(session.id)

    # The registration is gone while the moved folder is still being removed.
    assert session.workspace_path not in worktree_list(repo)
    assert not Path(session.workspace_path).exists()
    release.set()
    assert service.workspaces.pending_removals.wait(30)
    assert not any(service.workspaces.pending_removals.root.iterdir())
    assert (repo / "README.md").read_text(encoding="utf-8") == "base\n"


def test_deleting_an_in_place_task_moves_nothing(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session = create(service, repo, WorkspaceMode.in_place)
    (repo / "keep.txt").write_text("agent edit\n", encoding="utf-8")

    service.delete_session(session.id)

    assert not any(service.workspaces.pending_removals.root.iterdir())
    assert (repo / "keep.txt").read_text(encoding="utf-8") == "agent edit\n"
    assert (repo / ".git").is_dir()


def test_leftovers_are_removed_when_the_server_starts(tmp_path: Path) -> None:
    leftover = tmp_path / "coding" / "pending-removal" / "abc"
    (leftover / "nested").mkdir(parents=True)
    (leftover / "nested" / "file.txt").write_text("x", encoding="utf-8")
    (leftover / "nested").chmod(0o500)

    manager = WorkspaceManager(tmp_path / "coding")

    assert manager.pending_removals.wait(30)
    assert not leftover.exists()


def test_a_failed_removal_does_not_block_later_deletes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remove = pending_module.force_remove
    stuck: list[Path] = []

    def fail_first(path: Path) -> None:
        if not stuck or stuck[0] == path:
            stuck[:] = [path]
            raise PermissionError("busy")
        remove(path)

    monkeypatch.setattr(pending_module, "force_remove", fail_first)
    pending = PendingRemovals(tmp_path / "pending")
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        (folder / "data").mkdir(parents=True)

    assert pending.discard(first)
    assert pending.wait(30)
    assert pending.discard(second)
    assert pending.wait(30)

    assert not first.exists() and not second.exists()
    # Only the folder that failed is left, for the next pass to retry.
    assert list(pending.root.iterdir()) == stuck
    assert pending.status().failed == 1
    # A failed restart retry retains the warning. A user-triggered retry can
    # clear it without restarting or deleting another task.
    restarted = PendingRemovals(pending.root)
    assert restarted.wait(30)
    assert restarted.status().failed == 1
    monkeypatch.setattr(pending_module, "force_remove", remove)
    restarted.retry()
    assert restarted.wait(30)
    assert restarted.status().failed == 0
    assert not any(pending.root.iterdir())


def test_a_suppressed_filesystem_error_is_still_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.setattr(pending_module, "force_remove", lambda _path: None)
    pending = PendingRemovals(tmp_path / "pending")
    assert pending.discard(folder)
    assert pending.wait(30)
    assert pending.status().failed == 1


def test_cleanup_status_survives_a_missing_markers_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pending = PendingRemovals(tmp_path / "pending")
    shutil.rmtree(pending.errors_root)

    assert pending.status().failed == 0

    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.setattr(pending_module, "force_remove", lambda _path: None)
    assert pending.discard(folder)
    assert pending.wait(30)
    # The worker recreates the folder, so the failure still survives a restart.
    assert PendingRemovals(tmp_path / "pending").status().failed == 1


def test_a_failure_whose_folder_is_gone_is_forgotten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.setattr(pending_module, "force_remove", lambda _path: None)
    pending = PendingRemovals(tmp_path / "pending")
    assert pending.discard(folder)
    assert pending.wait(30)
    assert pending.status().failed == 1

    for leftover in pending.root.iterdir():
        shutil.rmtree(leftover)

    assert pending.status().failed == 0
    assert list(pending.errors_root.iterdir()) == []


def test_a_folder_that_cannot_be_moved_is_removed_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WorkspaceManager(tmp_path / "coding")
    task_dir = manager.snapshots_root / "task-1"
    (task_dir / "local-copy").mkdir(parents=True)

    def cross_device(*_args) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(pending_module.os, "rename", cross_device)
    manager.discard_recovery("task-1")

    assert not task_dir.exists()
