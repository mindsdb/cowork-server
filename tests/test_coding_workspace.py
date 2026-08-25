from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cowork.coding import workspace as workspace_module
from cowork.coding.contracts import WorkspaceKind
from cowork.coding.workspace import GitRunner, WorkspaceError, WorkspaceManager
from cowork.common.settings.app_settings import get_app_settings


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "cowork@example.invalid")
    git(repo, "config", "user.name", "Cowork Test")
    (repo / "keep.txt").write_text("base\n", encoding="utf-8")
    (repo / "remove.txt").write_text("remove\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def test_prepare_uses_detached_worktree_and_preserves_dirty_source(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    (repo / "source-only.txt").write_text("uncommitted\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "coding")

    prepared = manager.prepare("session-1", str(repo), False)

    assert prepared.kind == WorkspaceKind.git_worktree
    assert prepared.workspace_path != repo
    assert prepared.source_dirty is True
    assert prepared.warning and "not copied" in prepared.warning
    assert not (prepared.workspace_path / "source-only.txt").exists()
    assert manager.git_state(str(repo), str(prepared.workspace_path)).detached is True


def test_diff_includes_uncommitted_untracked_deleted_and_committed_changes(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-2", str(repo), False)
    worktree = prepared.workspace_path
    (worktree / "keep.txt").write_text("changed\n", encoding="utf-8")
    (worktree / "new.txt").write_text("new\n", encoding="utf-8")
    (worktree / "remove.txt").unlink()

    first = {item.path: item for item in manager.diff(str(worktree), prepared.base_revision)}
    assert set(first) == {"keep.txt", "new.txt", "remove.txt"}
    assert first["new.txt"].status == "??"
    assert first["keep.txt"].additions == 1
    assert first["keep.txt"].deletions == 1

    git(worktree, "add", "--all")
    git(worktree, "commit", "-m", "task changes")
    committed = {item.path: item for item in manager.diff(str(worktree), prepared.base_revision)}
    assert set(committed) == {"keep.txt", "new.txt", "remove.txt"}
    assert committed["new.txt"].status == "A"


def test_diff_bounds_large_tracked_file_content(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-large", str(repo), False)
    large = prepared.workspace_path / "large.txt"
    large.write_text("x" * (2 * 1024 * 1024 + 1), encoding="utf-8")
    git(prepared.workspace_path, "add", "large.txt")

    rendered = {item.path: item for item in manager.diff(str(prepared.workspace_path), prepared.base_revision)}

    assert rendered["large.txt"].binary is True
    assert rendered["large.txt"].patch.startswith("Large file changed")


def test_untracked_symlink_diff_does_not_follow_external_content(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-symlink", str(repo), False)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("must-not-appear-in-review\n", encoding="utf-8")
    link = prepared.workspace_path / "outside-link"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    rendered = {item.path: item for item in manager.diff(str(prepared.workspace_path), prepared.base_revision)}

    assert "must-not-appear-in-review" not in rendered["outside-link"].patch
    assert rendered["outside-link"].patch == f"Symbolic link → {secret}"
    assert rendered["outside-link"].binary is True


def test_diff_bounds_the_combined_inline_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-budget", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("changed\n", encoding="utf-8")
    (prepared.workspace_path / "remove.txt").write_text("also changed\n", encoding="utf-8")
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_DIFF_BYTES", 1)

    rendered = manager.diff(str(prepared.workspace_path), prepared.base_revision)

    assert rendered[0].patch
    assert rendered[1].patch.startswith("Inline diff omitted")


def test_apply_preflights_and_saves_recovery_patch(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-3", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("from task\n", encoding="utf-8")
    (prepared.workspace_path / "new.txt").write_text("new\n", encoding="utf-8")

    snapshot = manager.apply_to_source(
        "session-3", str(repo), str(prepared.workspace_path), prepared.base_revision
    )

    assert snapshot is not None and snapshot.is_file()
    assert "keep.txt" in snapshot.read_text(encoding="utf-8")
    assert (repo / "keep.txt").read_text(encoding="utf-8") == "from task\n"
    assert (repo / "new.txt").read_text(encoding="utf-8") == "new\n"


def test_apply_conflict_does_not_partially_change_source(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-4", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("from task\n", encoding="utf-8")
    (repo / "keep.txt").write_text("from user\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="stopped before changing"):
        manager.apply_to_source("session-4", str(repo), str(prepared.workspace_path), prepared.base_revision)

    assert (repo / "keep.txt").read_text(encoding="utf-8") == "from user\n"


def test_non_git_folder_requires_isolation_and_uses_a_managed_copy(tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    manager = WorkspaceManager(tmp_path / "coding")
    with pytest.raises(WorkspaceError, match="Local folder isolation was not enabled"):
        manager.prepare("direct-1", str(folder), False)

    prepared = manager.prepare("direct-2", str(folder), True)
    assert prepared.kind == WorkspaceKind.local_copy
    assert prepared.workspace_path != folder.resolve()
    assert prepared.workspace_path.is_dir()


def test_repository_without_commits_uses_an_isolated_copy(tmp_path: Path) -> None:
    repo = tmp_path / "empty-repository"
    repo.mkdir()
    git(repo, "init")
    (repo / "untracked.txt").write_text("source\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "coding")

    inspection = manager.inspect(str(repo))
    prepared = manager.prepare("unborn-1", str(repo), False)

    assert inspection.is_git is True
    assert inspection.revision is None
    assert inspection.warning is None
    assert prepared.kind == WorkspaceKind.local_copy
    assert prepared.workspace_path != repo.resolve()
    assert (prepared.workspace_path / "untracked.txt").read_text(encoding="utf-8") == "source\n"
    assert manager.git_state(str(repo), str(prepared.workspace_path)).revision is None


def test_unborn_repository_copy_never_reviews_or_hands_off_git_metadata(tmp_path: Path) -> None:
    source = tmp_path / "repository-container"
    nested = source / "nested"
    nested.mkdir(parents=True)
    git(source, "init")
    git(nested, "init")
    (nested / "file.txt").write_text("base\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("container-1", str(source), False)

    assert prepared.kind == WorkspaceKind.local_copy
    git(prepared.workspace_path / "nested", "add", "file.txt")
    (prepared.workspace_path / "nested" / "file.txt").write_text("changed\n", encoding="utf-8")
    changed = manager.diff(str(prepared.workspace_path), prepared.base_revision)

    assert [item.path for item in changed] == ["nested/file.txt"]
    manager.apply_to_source(
        "container-1",
        str(source),
        str(prepared.workspace_path),
        prepared.base_revision,
    )
    assert (nested / "file.txt").read_text(encoding="utf-8") == "changed\n"
    assert git(nested, "status", "--short") == "?? file.txt"


def test_cleanup_only_removes_the_managed_task_worktree(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-5", str(repo), False)
    (prepared.workspace_path / "uncommitted.txt").write_text("discarded\n", encoding="utf-8")

    manager.cleanup(
        "session-5", str(repo), str(prepared.workspace_path), prepared.kind, prepared.base_revision
    )

    assert repo.is_dir()
    assert not prepared.workspace_path.exists()
    assert "session-5" not in git(repo, "worktree", "list")
    snapshot = tmp_path / "coding" / "snapshots" / "session-5" / "cleanup.patch"
    assert snapshot.is_file()
    assert "uncommitted.txt" in snapshot.read_text(encoding="utf-8")


def test_cleanup_refuses_unmanaged_paths(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    with pytest.raises(WorkspaceError, match="unmanaged worktree"):
        manager.cleanup("session-6", str(repo), str(repo), WorkspaceKind.git_worktree, "abc")


def test_git_runner_refuses_before_spawning_in_org_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Git must not spawn"))

    with pytest.raises(WorkspaceError, match="not available"):
        GitRunner().run(tmp_path, "status")

    get_app_settings.cache_clear()
