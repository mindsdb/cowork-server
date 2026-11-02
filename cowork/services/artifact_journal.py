"""An artifact's journal folder, `.revisions`, opened without following links.

The journal holds the server's own records of one artifact: its revisions, a
pending source write, agent repairs, local comments, the draft-review grant and
the lock. It sits inside the artifact folder, which the agent's code can write.
A cell locked by the code sandbox can rewrite any file there and hard-link any
file it can read, and a link left from before the sandbox, or planted by a
remote worker, can stand where any folder should be. So every read and write
of the journal goes through descriptors opened here: the artifact folder and
every folder below it open without following a link, a file the server reads
must be a regular file with one name, and no read runs past its cap. The
folders above the artifact folder were opened the same way when the artifact
was found (artifact_identity._opened_from_trusted_root).
"""
from __future__ import annotations

import errno
import json
import logging
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from anton.core.artifacts.internal_files import REVISIONS_DIRNAME

from cowork.common.logger import log_context
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
    is_child_name,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)

logger = logging.getLogger(__name__)

JOURNAL_DIRNAME = REVISIONS_DIRNAME
#: The largest journal JSON file the server reads. A comment journal or an
#: agent repair, which carries its whole comment thread, can legitimately reach
#: tens of megabytes; the cap stops a sparse file, which a cell can make with
#: its `truncate` right, from filling the server's memory.
MAX_JOURNAL_FILE_BYTES = 32 * 1024 * 1024
#: The cap for a journal record the server only ever writes small: a pending
#: source write or the draft-review grant.
MAX_JOURNAL_RECORD_BYTES = 64 * 1024

# An open that fails with one of these found no folder at the name, which reads
# as an empty journal. Any other failure, such as EMFILE or EIO, says the open
# couldn't be made, and is raised: read as empty, it would let the next write
# replace the whole journal.
_NO_FOLDER = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP})


class JournalFileRefused(OSError):
    """A journal name holds something the server never writes there: a link,
    a file that isn't regular, one with a second name, or one past its cap."""


def _refuse(folder: Path, reason: str) -> JournalFileRefused:
    logger.warning(
        "Refused an artifact journal file",
        extra={
            "event": "artifact_journal_file_refused",
            "reason": reason,
            **log_context(artifact_slug=folder.name),
        },
    )
    return JournalFileRefused(f"Journal file refused: {reason}")


def _check_names(names: tuple[str, ...]) -> None:
    if not all(is_child_name(name) for name in names):
        raise ValueError("Journal names must be direct-child names")


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
        except OSError as exc:
            # Replaced by something else since the lstat.
            if create or exc.errno not in _NO_FOLDER:
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
    _check_names(parts)
    try:
        current = opened.enter_context(pinned_dir(folder, nofollow_base=True))
    except OSError as exc:
        if create or exc.errno not in _NO_FOLDER:
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
    anything else standing at one of their names is replaced. Any other
    failure to open a folder is raised.
    """
    with ExitStack() as opened:
        yield _open_chain(folder, parts, create=create, opened=opened)


def _read_child(folder: Path, journal: PinnedDir, name: str, max_bytes: int) -> bytes | None:
    """The bytes of the file ``name`` in ``journal``, the journal of the
    artifact at ``folder``, or None when it doesn't exist. Raises
    JournalFileRefused when the name holds anything the server doesn't write
    there."""
    try:
        fd = dir_open(journal, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _refuse(folder, "link") from exc
        raise
    try:
        found = os.fstat(fd)
        if not stat.S_ISREG(found.st_mode):
            raise _refuse(folder, "not_regular")
        if found.st_nlink != 1:
            raise _refuse(folder, "second_name")
        if found.st_size > max_bytes:
            raise _refuse(folder, "too_large")
        return b"".join(read_bounded(fd, found.st_size))
    finally:
        os.close(fd)


def _parse(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except RecursionError as exc:
        # Deep nesting raises RecursionError, which isn't a ValueError; every
        # caller already treats a ValueError as a damaged record.
        raise ValueError("Journal file is nested too deeply") from exc


def read_journal_file(
    folder: Path, *parts: str, max_bytes: int = MAX_JOURNAL_FILE_BYTES
) -> bytes | None:
    """The bytes of ``.revisions/<parts>``, or None when it doesn't exist.

    Raises JournalFileRefused when the name holds anything the server doesn't
    write there, so a caller can tell a damaged journal from an empty one, and
    any other OSError when the read couldn't be made.
    """
    *folders, name = parts
    _check_names((name,))
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return None
        return _read_child(folder, journal, name, max_bytes)


def read_journal_json(
    folder: Path, *parts: str, max_bytes: int = MAX_JOURNAL_FILE_BYTES
) -> object | None:
    """``.revisions/<parts>`` parsed as JSON, or None when it doesn't exist.

    Raises as read_journal_file does, and ValueError for a file that isn't
    JSON the parser can hold.
    """
    raw = read_journal_file(folder, *parts, max_bytes=max_bytes)
    return None if raw is None else _parse(raw)


def iter_journal_json(
    folder: Path,
    *parts: str,
    suffix: str,
    accept: Callable[[str], bool],
    max_bytes: int = MAX_JOURNAL_FILE_BYTES,
) -> Iterator[tuple[str, object]]:
    """Each file in the journal folder ``parts`` named ``<stem><suffix>`` with
    ``accept(stem)`` true, as its stem and its parsed JSON.

    The folder is opened once and every file is read through it, one at a
    time, so a caller that stops early or keeps only some records holds no
    more than one. A file that is refused or isn't JSON is skipped; a read
    that couldn't be made is raised.
    """
    with opened_journal(folder, *parts) as journal:
        if journal is None:
            return
        with dir_scandir(journal) as entries:
            stems = sorted(
                entry.name.removesuffix(suffix)
                for entry in entries
                if entry.name.endswith(suffix)
            )
        for stem in stems:
            if not accept(stem):
                continue
            try:
                raw = _read_child(folder, journal, f"{stem}{suffix}", max_bytes)
                if raw is None:
                    continue
                parsed = _parse(raw)
            except (JournalFileRefused, ValueError):
                continue
            yield stem, parsed


def write_journal_file(folder: Path, *parts: str, content: bytes) -> None:
    """Replace ``.revisions/<parts>`` with ``content`` in one durable step.

    The journal and each folder on the way are made when missing. The new file
    is created beside the old one and renamed over it, so a link or a second
    name already at the name is replaced, never written through.
    """
    *folders, name = parts
    _check_names((name,))
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
    _check_names((name,))
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return False
        try:
            dir_unlink(journal, name)
        except FileNotFoundError:
            return False
        return True


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
    _check_names((name,))
    with opened_journal(folder, *folders) as journal:
        if journal is None:
            return 0
        try:
            return dir_lstat(journal, name).st_mtime_ns
        except FileNotFoundError:
            return 0
