"""Cross-platform, process-safe lock for one artifact's mutable journals."""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_lstat,
    dir_open,
    dir_unlink,
    read_bounded,
)
from cowork.services.artifact_journal import opened_journal

_LOCK_NAME = ".lock"


def _holds(journal: PinnedDir, token: bytes) -> bool:
    fd = dir_open(journal, _LOCK_NAME, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    try:
        return b"".join(read_bounded(fd, len(token) + 1)) == token
    finally:
        os.close(fd)


@contextmanager
def artifact_lock(folder: Path, *, timeout: float = 5.0):
    """Serialize source, revision, repair, and local-comment mutations.

    The lock is an exclusive-create file rather than ``fcntl``/``msvcrt`` so
    the exact same protocol works in packaged macOS, Windows, and Linux apps.
    Artifact operations are short atomic-file swaps; a 30-second-old lock is
    therefore a crashed process remnant and can be recovered safely.

    The lock lives in the journal, opened without following links, so a
    `.revisions` that is a link is replaced by a real folder here rather than
    locking another artifact's journal.
    """
    deadline = time.monotonic() + timeout
    token = f"{os.getpid()}:{uuid.uuid4().hex}\n".encode("ascii")
    with opened_journal(folder, create=True) as journal:
        while True:
            try:
                fd = dir_open(
                    journal,
                    _LOCK_NAME,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | O_NOFOLLOW,
                    0o600,
                )
                try:
                    os.write(fd, token)
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - dir_lstat(journal, _LOCK_NAME).st_mtime > 30:
                        dir_unlink(journal, _LOCK_NAME)
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
                if _holds(journal, token):
                    dir_unlink(journal, _LOCK_NAME)
            except OSError:
                pass
