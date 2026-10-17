"""Measure the storage Cowork manages for each task, in the background.

Walking large trees is slow, so a measurement runs on its own thread and the
last result is served from memory. Asking for usage starts a new measurement
once the last one is older than ``max_age``.

What is counted: each ``<task>`` folder under the workspace roots (worktrees,
folder copies and their baselines) and the recovery roots (snapshots, saved
copies, released-workspace state). Symbolic links are never followed. A file
with several hard links is counted once per task, so a file hard-linked into
two tasks counts in both. Folders under a root that belong to no current task
are reported together as ``removed_task_bytes``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from cowork.coding.contracts import WorkspaceUsageItem, WorkspaceUsagePage, utc_now
from cowork.coding.workspace_key import managed_key

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE = timedelta(minutes=10)


class UsageRootsLike(Protocol):
    workspace: tuple[Path, ...]
    recovery: tuple[Path, ...]


class WorkspaceUsage:
    def __init__(
        self,
        roots: Callable[[], UsageRootsLike],
        task_ids: Callable[[], Iterable[str]],
        max_age: timedelta = DEFAULT_MAX_AGE,
    ) -> None:
        self._roots = roots
        self._task_ids = task_ids
        self._max_age = max_age
        self._lock = threading.Lock()
        self._items: list[WorkspaceUsageItem] = []
        self._removed_task_bytes = 0
        self._measured_at: datetime | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self, session_id: str | None = None) -> WorkspaceUsagePage:
        """The last measurement, starting a new one in the background if it is stale.

        A measurement that does not include ``session_id``, such as one taken
        before that task was created, counts as stale.
        """
        with self._lock:
            stale = self._measured_at is None or utc_now() - self._measured_at > self._max_age
            if session_id is not None and all(item.session_id != session_id for item in self._items):
                stale = True
            if stale and self._thread is None:
                self._thread = threading.Thread(target=self._measure_in_background, name="coding-workspace-usage", daemon=True)
                self._thread.start()
            return self._page()

    def measure(self) -> WorkspaceUsagePage:
        """Measure now, on the calling thread."""
        roots = self._roots()
        items: list[WorkspaceUsageItem] = []
        owned: set[str] = set()
        for task_id in self._task_ids():
            try:
                relative = managed_key(task_id)
            except ValueError:
                items.append(WorkspaceUsageItem(session_id=task_id, bytes=0))
                continue
            owned.add(relative.parts[0])
            seen: set[tuple[int, int]] = set()
            workspace = sum(tree_bytes(root / relative, seen) for root in roots.workspace)
            recovery = sum(tree_bytes(root / relative, seen) for root in roots.recovery)
            items.append(WorkspaceUsageItem(session_id=task_id, bytes=workspace + recovery, recovery_bytes=recovery))
        removed = sum(
            _unowned_bytes(root, owned) for root in dict.fromkeys((*roots.workspace, *roots.recovery))
        )
        with self._lock:
            self._items = items
            self._removed_task_bytes = removed
            self._measured_at = utc_now()
            return self._page()

    def _measure_in_background(self) -> None:
        try:
            self.measure()
        except Exception:
            logger.exception("Measuring task storage failed")
        finally:
            with self._lock:
                self._thread = None

    def _page(self) -> WorkspaceUsagePage:
        return WorkspaceUsagePage(
            items=list(self._items),
            removed_task_bytes=self._removed_task_bytes,
            total_bytes=sum(item.bytes for item in self._items) + self._removed_task_bytes,
            measured_at=self._measured_at,
            measuring=self._thread is not None,
        )


def _unowned_bytes(root: Path, owned: set[str]) -> int:
    """Storage in ``root`` under folders that belong to no current task."""
    try:
        entries = list(os.scandir(root))
    except OSError:
        return 0
    total = 0
    for entry in entries:
        if entry.name in owned or not entry.is_dir(follow_symlinks=False):
            continue
        total += tree_bytes(Path(entry.path))
    return total


def tree_bytes(root: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """Disk allocated under ``root``, without following links or counting a hard link twice."""
    seen = set() if seen is None else seen
    total = 0
    pending = [str(root)]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(entry.path)
            if info.st_nlink > 1 and not entry.is_dir(follow_symlinks=False):
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
            # Allocated blocks where the platform reports them; Windows does not.
            blocks = getattr(info, "st_blocks", None)
            total += blocks * 512 if blocks is not None else info.st_size
    return total
