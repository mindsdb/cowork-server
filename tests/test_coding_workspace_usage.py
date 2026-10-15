"""Each task's workspace size is measured in the background and served from memory."""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from pathlib import Path

from coding_service_fakes import CREDS, FakeEngine, git, repository, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus
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


def test_each_task_is_measured_across_every_managed_folder(tmp_path: Path) -> None:
    roots = [tmp_path / "workspaces", tmp_path / "baselines"]
    write(roots[0] / "task-1" / "app" / "file", 100_000)
    write(roots[1] / "task-1" / "file", 50_000)
    write(roots[0] / "task-2" / "file", 10_000)
    usage = WorkspaceUsage(lambda: roots, lambda: ["task-1", "task-2", "not a valid key"])

    page = usage.measure()

    sizes = {item.session_id: item.bytes for item in page.items}
    assert sizes["task-1"] >= 150_000
    assert 10_000 <= sizes["task-2"] < 150_000
    assert sizes["not a valid key"] == 0
    assert page.total_bytes == sum(sizes.values())
    assert page.measured_at is not None


def test_asking_for_usage_answers_at_once_and_measures_in_the_background(tmp_path: Path) -> None:
    started, release = threading.Event(), threading.Event()

    def slow_ids():
        started.set()
        assert release.wait(timeout=3)
        return ["task-1"]

    usage = WorkspaceUsage(lambda: [tmp_path], slow_ids)

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
    usage = WorkspaceUsage(lambda: [tmp_path], lambda: calls.append(1) or [], max_age=timedelta(hours=1))
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
