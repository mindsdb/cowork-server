"""An artifact's journal folder, `.revisions`, opened without following links.

The journal holds the server's own records of one artifact: its revisions, a
pending source write, agent repairs, local comments, the draft-review grant and
the lock. It sits inside the artifact folder, which the agent's code can write.
A cell locked by the code sandbox can rewrite any file there and hard-link any
file it can read, and a link left from before the sandbox, or planted by a
remote worker, can stand where any folder should be. So every read and write
of the journal goes through descriptors opened here: no part of the path is
followed through a link, a file the server reads must be a regular file with
one name, and no read runs past its cap.
"""
from __future__ import annotations

import logging
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from anton.core.artifacts.internal_files import REVISIONS_DIRNAME

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_fsync,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_replace,
    dir_scandir,
    dir_unlink,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)

logger = logging.getLogger(__name__)

JOURNAL_DIRNAME = REVISIONS_DIRNAME
#: The largest journal JSON file the server reads. A comment journal can
#: legitimately reach a few megabytes; the cap stops a sparse file, which a
#: cell can make with its `truncate` right, from filling the server's memory.
MAX_JOURNAL_FILE_BYTES = 32 * 1024 * 1024


class JournalFileRefused(OSError):
    """A journal name holds something the server never writes there: a file
    that isn't regular, one with a second name, or one past its cap."""


def _open_folder(parent: PinnedDir, name: str, *, create: bool) -> PinnedDir | None:
    """Open the folder ``name`` below ``parent`` without following a link.

    Without ``create``, a missing name, a link, or anything else that isn't a
    real folder answers None: that part of the journal doesn't exist. With
    ``create``, a missing folder is made, and anything else at the name is
    removed first. Only the name goes: removing a link or a second name never
    touches the file it points to.
    """
    try:
        found = dir_lstat(parent, name)
    except FileNotFoundError:
        found = None
    if found is not None and stat.S_ISDIR(found.st_mode):
        try:
            return open_pinned_child(parent, name)
        except OSError:
            # Replaced by something else since the lstat.
            if create:
                raise
            return None
    if not create:
        return None
    if found is not None:
        try:
            dir_unlink(parent, name)
        except FileNotFoundError:
            pass
        logger.warning(
            "Replaced an artifact journal entry that is not a folder",
            extra={
                "event": "artifact_journal_entry_replaced",
                "reason": "link" if stat.S_ISLNK(found.st_mode) else "not_a_folder",
            },
        )
    try:
        dir_mkdir(parent, name)
    except FileExistsError:
        # Another request made it first.
        pass
    return open_pinned_child(parent, name)


def _open_chain(
    folder: Path, parts: tuple[str, ...], *, create: bool, opened: ExitStack
) -> PinnedDir | None:
    try:
        current = opened.enter_context(pinned_dir(folder, nofollow_base=True))
    except OSError:
        if create:
            raise
        return None
    for name in (JOURNAL_DIRNAME, *parts):
        current = _open_folder(current, name, create=create)
        if current is None:
            return None
        opened.callback(current.close)
    return current


@contextmanager
def opened_journal(
    folder: Path, *parts: str, create: bool = False
) -> Iterator[PinnedDir | None]:
    """The journal of the artifact at ``folder``, or its folder ``parts``.

    ``folder`` itself is opened without following a link. Without ``create``,
    this yields None when the artifact, the journal or a folder on the way is
    missing or isn't a real folder, which callers read as an empty journal.
    With ``create``, the journal and each folder on the way are made, and
    anything else standing at one of their names is replaced.
    """
    with ExitStack() as opened:
        yield _open_chain(folder, parts, create=create, opened=opened)


def read_journal_file(
    folder: Path, *parts: str, max_bytes: int = MAX_JOURNAL_FILE_BYTES
) -> bytes | None:
    """The bytes of ``.revisions/<parts>``, or None when it doesn't exist.

    Raises JournalFileRefused when the name holds anything the server doesn't
    write there, and OSError (ELOOP) for a link, so a caller can tell a damaged
    journal from an empty one.
    """
    *folders, name = parts
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return None
        try:
            fd = dir_open(journal, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        except FileNotFoundError:
            return None
        try:
            found = os.fstat(fd)
            if not stat.S_ISREG(found.st_mode):
                raise JournalFileRefused("Journal file is not a regular file")
            if found.st_nlink != 1:
                raise JournalFileRefused("Journal file has a second name")
            if found.st_size > max_bytes:
                raise JournalFileRefused("Journal file is too large")
            return b"".join(read_bounded(fd, found.st_size))
        finally:
            os.close(fd)


def write_journal_file(folder: Path, *parts: str, content: bytes) -> None:
    """Replace ``.revisions/<parts>`` with ``content`` in one durable step.

    The journal and each folder on the way are made when missing. The new file
    is created beside the old one and renamed over it, so a link or a second
    name already at the name is replaced, never written through.
    """
    *folders, name = parts
    with opened_journal(folder, *folders, create=True) as journal:
        tmp = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            fd = dir_open(
                journal, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            dir_replace(journal, tmp, name)
            try:
                dir_fsync(journal)
            except OSError:
                # The file is durable and the rename atomic; a filesystem that
                # can't sync a folder only loses the rename's durability.
                pass
        finally:
            try:
                dir_unlink(journal, tmp)
            except OSError:
                pass


def remove_journal_file(folder: Path, *parts: str) -> bool:
    """Remove the name ``.revisions/<parts>``. True when one was there."""
    *folders, name = parts
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return False
        try:
            dir_unlink(journal, name)
        except FileNotFoundError:
            return False
        return True


def journal_file_names(folder: Path, *parts: str) -> list[str]:
    """The names in the journal folder ``parts``, sorted. Empty when it doesn't
    exist."""
    with opened_journal(folder, *parts) as journal:
        if journal is None:
            return []
        with dir_scandir(journal) as entries:
            return sorted(entry.name for entry in entries)


def prune_journal_folder(folder: Path, *parts: str, keep: set[str]) -> None:
    """Remove each name in the journal folder ``parts`` that isn't in ``keep``.

    Folders are left alone, and the names are listed and removed through one
    descriptor, so nothing outside the journal can be removed.
    """
    with opened_journal(folder, *parts) as journal:
        if journal is None:
            return
        with dir_scandir(journal) as entries:
            stale = [
                entry.name
                for entry in entries
                if entry.name not in keep and not entry.is_dir(follow_symlinks=False)
            ]
        for name in stale:
            try:
                dir_unlink(journal, name)
            except FileNotFoundError:
                pass


def journal_file_mtime_ns(folder: Path, *parts: str) -> int:
    """When ``.revisions/<parts>`` last changed, or 0 when it doesn't exist.
    The name isn't followed."""
    *folders, name = parts
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return 0
        try:
            return dir_lstat(journal, name).st_mtime_ns
        except FileNotFoundError:
            return 0
