"""Cross-platform, process-safe lock for one artifact's mutable journals."""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def artifact_lock(folder: Path, *, timeout: float = 5.0):
    """Serialize source, revision, repair, and local-comment mutations.

    The lock is an exclusive-create file rather than ``fcntl``/``msvcrt`` so
    the exact same protocol works in packaged macOS, Windows, and Linux apps.
    Artifact operations are short atomic-file swaps; a 30-second-old lock is
    therefore a crashed process remnant and can be recovered safely.
    """
    lock = folder / ".revisions" / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    token = f"{os.getpid()}:{uuid.uuid4().hex}\n"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, token.encode("ascii"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Artifact is busy; try again")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            # A long-paused process must not remove a successor's lock after
            # another process legitimately recovered its stale file.
            if lock.read_text(encoding="ascii") == token:
                lock.unlink(missing_ok=True)
        except OSError:
            pass
