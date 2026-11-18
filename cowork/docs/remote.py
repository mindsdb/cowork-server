"""The doc tools of a remote (pod) turn, run on this server.

The pod forwards each call (cowork.turnqueue.host_tools); this module binds
those calls to the doc executor (cowork.docs.tools) with what a remote turn
needs on top of a desktop one: the pod's paths are untrusted, since this
server sees every organization's files.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from typing import TYPE_CHECKING

from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, dir_open, opened_subdir_nofollow, read_bounded
from cowork.docs.images import MAX_IMAGE_BYTES

if TYPE_CHECKING:
    from cowork.harnesses.anton_harness.agent_files import AgentFiles


def pod_image_reader(files: AgentFiles | None) -> Callable[[str], bytes]:
    """Read an image the pod names by its own path: placed in the project by
    the pod's mounts (AgentFiles.pod_project_parts), then opened from the
    project folder one component at a time with O_NOFOLLOW, so a link the
    agent's code plants anywhere on the way is refused rather than followed.
    Raises OSError for anything it will not read."""

    def read(src: str) -> bytes:
        parts = files.pod_project_parts(src) if files is not None else None
        if not parts:
            raise OSError("not a path in this turn's project")
        *dirs, name = parts
        try:
            with opened_subdir_nofollow(files.project_root, *dirs) as pinned:
                fd = dir_open(pinned, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
                try:
                    found = os.fstat(fd)
                    if not stat.S_ISREG(found.st_mode) or found.st_size > MAX_IMAGE_BYTES:
                        raise OSError("not an image file within the size limit")
                    return b"".join(read_bounded(fd, found.st_size))
                finally:
                    os.close(fd)
        except ValueError as exc:
            # An embedded NUL or similar: the path is unreadable, not a bug.
            raise OSError(str(exc)) from exc

    return read
