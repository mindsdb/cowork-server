"""Measure how much disk each task's workspace uses, in the background.

Walking large trees is slow, so a measurement runs on its own thread and the
last result is served from memory. Asking for usage starts a new measurement
once the last one is older than ``max_age``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

from cowork.coding.contracts import WorkspaceUsageItem, WorkspaceUsagePage, utc_now
from cowork.coding.workspace_key import managed_key

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE = timedelta(minutes=10)


class WorkspaceUsage:
    def __init__(
        self,
        task_roots: Callable[[], Iterable[Path]],
        task_ids: Callable[[], Iterable[str]],
        max_age: timedelta = DEFAULT_MAX_AGE,
    ) -> None:
        self._task_roots = task_roots
        self._task_ids = task_ids
        self._max_age = max_age
        self._lock = threading.Lock()
        self._items: list[WorkspaceUsageItem] = []
        self._measured_at: datetime | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self) -> WorkspaceUsagePage:
        """The last measurement, starting a new one in the background if it is stale."""
        with self._lock:
            stale = self._measured_at is None or utc_now() - self._measured_at > self._max_age
            if stale and self._thread is None:
                self._thread = threading.Thread(target=self._measure_in_background, name="coding-workspace-usage", daemon=True)
                self._thread.start()
            return self._page()

    def measure(self) -> WorkspaceUsagePage:
        """Measure now, on the calling thread."""
        items = [
            WorkspaceUsageItem(session_id=task_id, bytes=self._task_bytes(task_id))
            for task_id in self._task_ids()
        ]
        with self._lock:
            self._items = items
            self._measured_at = utc_now()
            return self._page()

    def _measure_in_background(self) -> None:
        try:
            self.measure()
        except Exception:
            logger.exception("Measuring task workspace disk usage failed")
        finally:
            with self._lock:
                self._thread = None

    def _page(self) -> WorkspaceUsagePage:
        return WorkspaceUsagePage(
            items=list(self._items),
            total_bytes=sum(item.bytes for item in self._items),
            measured_at=self._measured_at,
            measuring=self._thread is not None,
        )

    def _task_bytes(self, task_id: str) -> int:
        try:
            relative = managed_key(task_id)
        except ValueError:
            return 0
        seen: set[tuple[int, int]] = set()
        return sum(tree_bytes(root / relative, seen) for root in self._task_roots())


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
