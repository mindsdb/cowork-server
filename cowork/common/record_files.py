"""One bounded read of a small file in a pinned folder.

A saved connection's record, and the managed folder's registry that the
operator's commands read, sit in folders that agent code could once write to,
so a reader can't trust what is at a name: a link, a file with a second name,
a FIFO or something very large. read_record_bytes opens the name through its
folder's descriptor, following no link, and reads only a regular file with one
name, no larger than its limit.
"""

from __future__ import annotations

import errno
import os
import stat

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_open,
    read_bounded,
)

# The largest saved connection record file a reader opens.
MAX_RECORD_BYTES = 1 << 20


class RecordFileError(OSError):
    """A file read_record_bytes refuses. ``errno`` is EMLINK for anything but a
    regular file with one name, and EFBIG for one larger than the limit."""


def read_record_bytes(
    *, directory: PinnedDir, name: str, limit: int = MAX_RECORD_BYTES
) -> bytes:
    """The bytes of ``name`` in ``directory``, opened without following a link.

    The open's, fstat's and read's own errors raise as they come, each an
    OSError with its errno: FileNotFoundError when nothing is there, ELOOP for
    a link. A file the reader refuses raises RecordFileError.
    """
    descriptor = dir_open(directory, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RecordFileError(errno.EMLINK, "not a regular file with one name")
        if info.st_size > limit:
            raise RecordFileError(errno.EFBIG, "larger than the read limit")
        return b"".join(read_bounded(descriptor, info.st_size))
    finally:
        os.close(descriptor)
