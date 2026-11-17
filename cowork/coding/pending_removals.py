"""Remove deleted tasks' managed folders off the request path.

Removing a multi-gigabyte workspace can take minutes. A folder is first moved
into this area with a rename, which is quick at any size because it stays on
the same filesystem, and a background thread then removes it. Anything still
here when the server stops is removed on the next start.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path

from cowork.coding.local_copy import force_remove

logger = logging.getLogger(__name__)


class PendingRemovals:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._requested = False
        self.root.mkdir(parents=True, exist_ok=True)
        if self._entries():
            self._start()

    def discard(self, path: Path) -> bool:
        """Move ``path`` here and remove it in the background.

        Returns False, leaving ``path`` in place, when it cannot be moved,
        such as when it sits on another filesystem; the caller removes it then.
        """
        if not path.exists() and not path.is_symlink():
            return True
        # Unique names keep one folder that fails to remove from blocking the next.
        target = self.root / uuid.uuid4().hex
        try:
            os.rename(path, target)
        except OSError as exc:
            logger.warning("Could not move %s aside for removal: %s", path, exc)
            return False
        self._start()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current removal pass. True once nothing is running."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _entries(self) -> list[Path]:
        try:
            return [Path(entry.path) for entry in os.scandir(self.root)]
        except OSError:
            return []

    def _start(self) -> None:
        with self._lock:
            self._requested = True
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._drain, name="coding-pending-removals", daemon=True)
            self._thread.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._requested:
                    self._thread = None
                    return
                self._requested = False
            for entry in self._entries():
                try:
                    if entry.is_symlink() or not entry.is_dir():
                        entry.unlink(missing_ok=True)
                    else:
                        force_remove(entry)
                except OSError as exc:
                    # Left for the next pass: the next delete or server start.
                    logger.warning("Could not remove %s: %s", entry, exc)
