from __future__ import annotations

import random
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
    assert first["new.txt"].unstaged is True
    assert first["new.txt"].staged is False
    assert first["keep.txt"].additions == 1
    assert first["keep.txt"].deletions == 1

    git(worktree, "add", "--all")
    git(worktree, "commit", "-m", "task changes")
    committed = {item.path: item for item in manager.diff(str(worktree), prepared.base_revision)}
    assert set(committed) == {"keep.txt", "new.txt", "remove.txt"}
    assert committed["new.txt"].status == "A"
    assert committed["new.txt"].staged is False
    assert committed["new.txt"].unstaged is False


def test_review_file_actions_stage_unstage_and_discard(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-review", str(repo), False)
    worktree = prepared.workspace_path
    (worktree / "keep.txt").write_text("changed\n", encoding="utf-8")
    (worktree / "new.txt").write_text("new\n", encoding="utf-8")

    manager.review_file_action(str(worktree), "keep.txt", "stage")
    staged = {item.path: item for item in manager.diff(str(worktree), prepared.base_revision)}
    assert staged["keep.txt"].staged is True
    assert staged["keep.txt"].unstaged is False

    manager.review_file_action(str(worktree), "keep.txt", "unstage")
    unstaged = {item.path: item for item in manager.diff(str(worktree), prepared.base_revision)}
    assert unstaged["keep.txt"].staged is False
    assert unstaged["keep.txt"].unstaged is True

    manager.review_file_action(str(worktree), "keep.txt", "discard")
    manager.review_file_action(str(worktree), "new.txt", "discard")
    assert manager.diff(str(worktree), prepared.base_revision) == []
    assert (worktree / "keep.txt").read_text(encoding="utf-8") == "base\n"
    assert not (worktree / "new.txt").exists()


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt", "nested/../../outside.txt"])
def test_review_file_actions_reject_paths_outside_workspace(tmp_path: Path, path: str) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare(f"session-review-{len(path)}", str(repo), False)

    with pytest.raises(WorkspaceError, match="inside this task workspace"):
        manager.review_file_action(str(prepared.workspace_path), path, "discard")


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


def test_cleanup_snapshot_is_truncated_at_the_combined_patch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-snapshot-budget", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("changed\n", encoding="utf-8")
    (prepared.workspace_path / "large.bin").write_bytes(random.Random(0).randbytes(50_000))
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_DIFF_BYTES", 4_096)

    manager.cleanup(
        "session-snapshot-budget", str(repo), str(prepared.workspace_path), prepared.kind, prepared.base_revision
    )

    snapshot_dir = tmp_path / "coding" / "snapshots" / "session-snapshot-budget"
    data = (snapshot_dir / "cleanup.patch").read_bytes()
    assert data.startswith(b"diff --git a/keep.txt b/keep.txt")
    assert b"large.bin" not in data
    assert data.endswith(b"\n") and len(data) <= 4_096
    assert (snapshot_dir / "cleanup.patch.truncated").read_text(encoding="utf-8") == (
        workspace_module.SNAPSHOT_TRUNCATED_MARKER
    )
    git(repo, "apply", "--check", str(snapshot_dir / "cleanup.patch"))


def test_handoff_and_fork_move_a_patch_over_the_cleanup_budget_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-handoff-budget", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("from task\n", encoding="utf-8")
    (prepared.workspace_path / "new.bin").write_bytes(bytes(range(256)) * 20)
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_DIFF_BYTES", 1)

    forked = manager.fork(
        "session-fork-budget", str(repo), str(prepared.workspace_path), prepared.kind, prepared.base_revision
    )
    manager.apply_to_source("session-handoff-budget", str(repo), str(prepared.workspace_path), prepared.base_revision)

    for root in (forked.workspace_path, repo):
        assert (root / "keep.txt").read_text(encoding="utf-8") == "from task\n"
        assert (root / "new.bin").read_bytes() == bytes(range(256)) * 20
    assert not list((tmp_path / "coding" / "snapshots").rglob("*.truncated"))


def test_failed_snapshot_leaves_no_partial_patch_behind(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    manager = WorkspaceManager(tmp_path / "coding")
    prepared = manager.prepare("session-snapshot-failure", str(repo), False)
    (prepared.workspace_path / "keep.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        manager.cleanup("session-snapshot-failure", str(repo), str(prepared.workspace_path), prepared.kind, "no-such-rev")

    assert list((tmp_path / "coding" / "snapshots" / "session-snapshot-failure").iterdir()) == []


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


def test_git_runner_rejects_an_unavailable_working_directory_before_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Git must not spawn"))

    with pytest.raises(WorkspaceError, match="available local folder"):
        GitRunner().run(tmp_path / "missing", "status")


@pytest.mark.parametrize(
    ("method", "command_output"),
    [
        ("changes", "M\0../../outside.txt\0"),
        ("status", "?? ../../outside.txt\0"),
    ],
)
def test_git_reported_paths_are_revalidated_before_file_operations(
    tmp_path: Path, method: str, command_output: str
) -> None:
    class UntrustedGitOutput:
        def run(self, _cwd: Path, *args: str, **_kwargs) -> subprocess.CompletedProcess[str]:
            output = command_output if (method == "changes" and args[0] == "diff") or args[0] == "status" else ""
            return subprocess.CompletedProcess(["git", *args], 0, stdout=output, stderr="")

    manager = WorkspaceManager(tmp_path / "coding", git=UntrustedGitOutput())

    with pytest.raises(WorkspaceError, match="inside this task workspace"):
        if method == "changes":
            manager._changes_since(tmp_path, "base")
        else:
            manager._status_entries(tmp_path)
