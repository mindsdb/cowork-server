from __future__ import annotations

import subprocess
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status
from cowork.coding.contracts import (
    DeliveryAutomationPolicy,
    SessionCreateRequest,
    SessionStatus,
    WorkspaceKind,
    utc_now,
)
from cowork.coding.workspace import WorkspaceManager
from cowork.coding.workspace_retention import RetentionPolicy


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def ref_exists(repo: Path, ref: str) -> bool:
    return subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=repo).returncode == 0


# --- WorkspaceManager primitives -------------------------------------------


def test_released_worktree_restores_commits_uncommitted_and_untracked_changes(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("task-1", str(repo), allow_direct_folder=False)
    worktree = prepared.workspace_path
    (worktree / "committed.txt").write_text("committed\n", encoding="utf-8")
    git(worktree, "add", "committed.txt")
    git(worktree, "commit", "-m", "task commit")
    head = git(worktree, "rev-parse", "HEAD")
    (worktree / "README.md").write_text("edited\n", encoding="utf-8")
    (worktree / "untracked.txt").write_text("new\n", encoding="utf-8")

    assert manager.release("task-1", str(repo), str(worktree), WorkspaceKind.git_worktree) is True

    assert not worktree.exists()
    assert ref_exists(repo, "refs/cowork/released/task-1")

    manager.restore("task-1", str(repo), str(worktree), WorkspaceKind.git_worktree)

    assert git(worktree, "rev-parse", "HEAD") == head
    assert (worktree / "committed.txt").read_text(encoding="utf-8") == "committed\n"
    assert (worktree / "README.md").read_text(encoding="utf-8") == "edited\n"
    assert (worktree / "untracked.txt").read_text(encoding="utf-8") == "new\n"
    assert not ref_exists(repo, "refs/cowork/released/task-1")


def test_a_worktree_is_kept_when_its_source_repository_is_gone(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("task-2", str(repo), allow_direct_folder=False)

    released = manager.release(
        "task-2", str(tmp_path / "moved-away"), str(prepared.workspace_path), WorkspaceKind.git_worktree
    )

    assert released is False
    assert prepared.workspace_path.is_dir()


def test_released_local_copy_restores_its_changes_onto_the_current_source(tmp_path: Path) -> None:
    source = tmp_path / "folder"
    source.mkdir()
    (source / "keep.txt").write_text("keep\n", encoding="utf-8")
    (source / "edit.txt").write_text("before\n", encoding="utf-8")
    (source / "remove.txt").write_text("remove\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("task-3", str(source), allow_direct_folder=True)
    copy = prepared.workspace_path
    (copy / "edit.txt").write_text("after\n", encoding="utf-8")
    (copy / "remove.txt").unlink()
    (copy / "added").mkdir()
    (copy / "added" / "new.txt").write_text("new\n", encoding="utf-8")
    before = {(item.path, item.status) for item in manager.diff(str(copy), None)}

    assert manager.release("task-3", str(source), str(copy), WorkspaceKind.local_copy) is True
    assert not copy.exists()
    # Work done elsewhere while the task was released is picked up, and is
    # not mistaken for a task change.
    (source / "keep.txt").write_text("changed outside the task\n", encoding="utf-8")

    manager.restore("task-3", str(source), str(copy), WorkspaceKind.local_copy)

    assert {(item.path, item.status) for item in manager.diff(str(copy), None)} == before
    assert before == {("edit.txt", "M"), ("remove.txt", "D"), ("added/new.txt", "A")}
    assert (copy / "keep.txt").read_text(encoding="utf-8") == "changed outside the task\n"
    assert (copy / "edit.txt").read_text(encoding="utf-8") == "after\n"
    assert not (copy / "remove.txt").exists()


def test_restoring_without_a_saved_release_fails_without_touching_disk(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    target = manager.worktrees_root / "never-released"

    with pytest.raises(Exception, match="no saved copy"):
        manager.restore("never-released", str(repo), str(target), WorkspaceKind.git_worktree)
    assert not target.exists()


# --- Service policy and restore-on-use -------------------------------------


def completed_task(service, repo: Path, prompt: str = "Work") -> str:
    created = service.create_session(SessionCreateRequest(path=str(repo), prompt=prompt), CREDS, "fake", "fake-model")
    wait_for_status(service, created.id, SessionStatus.completed)
    return created.id


def age(service, session_id: str, hours: float) -> None:
    service.store.update_session(
        session_id,
        lambda current: setattr(current, "updated_at", utc_now() - timedelta(hours=hours)),
        touch_updated_at=False,
    )


def test_policy_keeps_recent_tasks_and_releases_the_idle_rest(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    old, recent = completed_task(service, repo, "old"), completed_task(service, repo, "recent")
    age(service, old, 48)
    age(service, recent, 30)

    released = service.retention.run_policy(RetentionPolicy(keep_count=1, min_idle=timedelta(hours=24)))

    assert released == [old]
    assert service.get_session(old).workspace_released_at is not None
    assert not Path(service.get_session(old).workspace_path).exists()
    assert Path(service.get_session(recent).workspace_path).exists()


def test_policy_never_releases_pinned_recently_active_or_automated_tasks(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    pinned, fresh, automated = (completed_task(service, repo, name) for name in ("pinned", "fresh", "automated"))
    for session_id in (pinned, automated):
        age(service, session_id, 48)
    service.set_pinned(pinned, True)
    service.store.update_session(
        automated,
        lambda current: setattr(current, "delivery_policy", DeliveryAutomationPolicy(merge_when_approved=True)),
        touch_updated_at=False,
    )

    assert service.retention.run_policy(RetentionPolicy(keep_count=0, min_idle=timedelta(hours=24))) == []


def test_archiving_releases_and_using_the_task_restores_its_changes(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session_id = completed_task(service, repo)
    workspace = Path(service.get_session(session_id).workspace_path)
    (workspace / "README.md").write_text("task change\n", encoding="utf-8")

    archived = service.set_archived(session_id, True)

    assert archived.workspace_released_at is not None
    assert not workspace.exists()

    files = service.diff(session_id)

    assert [item.path for item in files] == ["README.md"]
    assert (workspace / "README.md").read_text(encoding="utf-8") == "task change\n"
    assert service.get_session(session_id).workspace_released_at is None


def test_a_new_turn_on_a_released_task_runs_in_its_rebuilt_workspace(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    session_id = completed_task(service, repo)
    assert service.retention.release(session_id) is True

    service.submit_turn(session_id, "Continue", CREDS)
    wait_for_status(service, session_id, SessionStatus.completed)

    assert Path(service.get_session(session_id).workspace_path).is_dir()
    assert engine.prompts[-1] == "Continue"


def test_deleting_a_released_task_forgets_its_saved_state(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session_id = completed_task(service, repo)
    assert service.retention.release(session_id) is True
    assert git(repo, "for-each-ref", "refs/cowork/released")

    service.delete_session(session_id)

    assert git(repo, "for-each-ref", "refs/cowork/released") == ""
    assert not list(service.workspaces.snapshots_root.rglob("release.json"))


def test_concurrent_views_of_a_released_task_share_one_restore(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session_id = completed_task(service, repo)
    assert service.retention.release(session_id) is True
    errors: list[BaseException] = []

    def view(call) -> None:
        try:
            call(session_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below.
            errors.append(exc)

    threads = [threading.Thread(target=view, args=(call,)) for call in (service.diff, service.git_state, service.diff)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert service.get_session(session_id).workspace_released_at is None
