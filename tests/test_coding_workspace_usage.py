"""Each task's workspace size is measured in the background and served from memory."""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from pathlib import Path

from coding_service_fakes import CREDS, FakeEngine, git, repository, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus, WorkspaceKind
from cowork.coding.project_models import ProjectCreateRequest, ProjectFolder
from cowork.coding.workspace import UsageRoots
from cowork.coding.workspace_usage import WorkspaceUsage, tree_bytes


def write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_a_tree_is_measured_without_following_links_or_counting_hard_links_twice(tmp_path: Path) -> None:
    root = tmp_path / "task"
    write(root / "a.bin", 64 * 1024)
    os.link(root / "a.bin", root / "same.bin")
    outside = tmp_path / "outside"
    write(outside / "big.bin", 1024 * 1024)
    (root / "link").symlink_to(outside, target_is_directory=True)

    measured = tree_bytes(root)

    assert 64 * 1024 <= measured < 1024 * 1024


def roots(*workspace: Path, recovery: tuple[Path, ...] = ()) -> UsageRoots:
    return UsageRoots(workspace=tuple(workspace), recovery=recovery)


def test_each_task_is_measured_across_every_managed_folder(tmp_path: Path) -> None:
    workspaces, baselines, snapshots = tmp_path / "workspaces", tmp_path / "baselines", tmp_path / "snapshots"
    write(workspaces / "task-1" / "app" / "file", 100_000)
    write(baselines / "task-1" / "file", 50_000)
    write(snapshots / "task-1" / "local-release" / "after" / "file", 30_000)
    write(workspaces / "task-2" / "file", 10_000)
    usage = WorkspaceUsage(
        lambda: roots(workspaces, baselines, recovery=(snapshots,)), lambda: ["task-1", "task-2", "not a valid key"],
    )

    page = usage.measure()

    items = {item.session_id: item for item in page.items}
    assert items["task-1"].bytes >= 180_000
    assert 30_000 <= items["task-1"].recovery_bytes < 100_000
    assert 10_000 <= items["task-2"].bytes < 150_000 and items["task-2"].recovery_bytes == 0
    assert items["not a valid key"].bytes == 0
    assert page.removed_task_bytes == 0
    assert page.total_bytes == sum(item.bytes for item in page.items)
    assert page.measured_at is not None


def test_storage_left_by_removed_tasks_is_counted_apart(tmp_path: Path) -> None:
    workspaces, snapshots = tmp_path / "workspaces", tmp_path / "snapshots"
    write(workspaces / "task-1" / "file", 10_000)
    write(snapshots / "gone" / "local-copy" / "file", 40_000)
    usage = WorkspaceUsage(lambda: roots(workspaces, recovery=(snapshots,)), lambda: ["task-1"])

    page = usage.measure()

    assert [item.session_id for item in page.items] == ["task-1"]
    assert page.removed_task_bytes >= 40_000
    assert page.total_bytes == page.items[0].bytes + page.removed_task_bytes


def test_asking_for_usage_answers_at_once_and_measures_in_the_background(tmp_path: Path) -> None:
    started, release = threading.Event(), threading.Event()

    def slow_ids():
        started.set()
        assert release.wait(timeout=3)
        return ["task-1"]

    usage = WorkspaceUsage(lambda: roots(tmp_path), slow_ids)

    first = usage.snapshot()
    assert first.measured_at is None and first.measuring
    assert started.wait(timeout=3)
    assert usage.snapshot().measuring  # one measurement at a time
    release.set()
    thread = usage._thread
    if thread is not None:
        thread.join(timeout=3)

    settled = usage.snapshot()
    assert settled.measured_at is not None and not settled.measuring
    assert [item.session_id for item in settled.items] == ["task-1"]


def test_a_fresh_measurement_is_reused(tmp_path: Path) -> None:
    calls = []
    usage = WorkspaceUsage(lambda: roots(tmp_path), lambda: calls.append(1) or [], max_age=timedelta(hours=1))
    usage.measure()

    page = usage.snapshot()

    assert not page.measuring
    assert len(calls) == 1


def test_the_service_reports_a_worktree_task_and_the_total(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    write(repo / "data.bin", 200_000)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "data")
    service = service_with(tmp_path, FakeEngine())
    session = wait_for_status(
        service,
        service.create_session(SessionCreateRequest(path=str(repo), prompt="Go"), CREDS, "fake", "fake-model").id,
        SessionStatus.completed,
    )

    page = service.workspace_usage.measure()

    sizes = {item.session_id: item.bytes for item in page.items}
    assert sizes[session.id] >= 200_000
    assert page.total_bytes >= sizes[session.id]


def test_a_deleted_folder_copy_keeps_counting_its_recovery_copy(tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    write(folder / "big.bin", 300_000)
    service = service_with(tmp_path, FakeEngine())
    session = wait_for_status(
        service,
        service.create_session(
            SessionCreateRequest(path=str(folder), prompt="Go", allow_direct_folder=True), CREDS, "fake", "fake-model",
        ).id,
        SessionStatus.completed,
    )
    assert session.workspace_kind == WorkspaceKind.local_copy
    (Path(session.workspace_path) / "edit.txt").write_text("unsaved work\n", encoding="utf-8")

    service.delete_session(session.id)
    page = service.workspace_usage.measure()

    # The task is gone, but the copy saved for recovery still takes space.
    assert session.id not in {item.session_id for item in page.items}
    assert page.removed_task_bytes >= 300_000


def test_a_project_task_is_measured_across_its_folders(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first, second = repository(tmp_path / "one"), repository(tmp_path / "two")
    for repo, size in ((first, 120_000), (second, 80_000)):
        write(repo / "data.bin", size)
        git(repo, "add", ".")
        git(repo, "commit", "-m", "data")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Two",
        folders=[ProjectFolder(id="one", name="One", path=str(first)), ProjectFolder(id="two", name="Two", path=str(second))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    session = wait_for_status(
        service,
        service.create_session(SessionCreateRequest(project_id=project.id, prompt="Go"), CREDS, "fake", "fake-model").id,
        SessionStatus.completed,
    )

    page = service.workspace_usage.measure()

    assert {item.session_id: item.bytes for item in page.items}[session.id] >= 200_000
