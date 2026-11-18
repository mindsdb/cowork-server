"""Which entries of a project folder are its ordinary files: the files a person
or the agent put there to be read, as opposed to the server's own folders,
hidden entries, caches and the files an operating system leaves behind.

One rule for every reader that asks: cloud routing (whether a project holds
files, ``cowork.services.files.project_holds_ordinary_files``), the copy of a
project's files into a cloud turn's workspace
(``cowork.services.project_file_copy.stage_project_files``) and a recipient's
listing of a shared project
(``cowork.services.shared_projects.SharedProjectFiles``), which hides more
names on top of it. So a project that routes a turn to the agent because it
holds files also stages those files.

The walk pins every folder by descriptor and opens each child without following
a link. A link, a hard-linked file (a second name for a file elsewhere, such as
``.anton/.env``), a pipe or a device is never an ordinary file.
"""
from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

from cowork.common.paths import PinnedDir, dir_lstat, dir_scandir, open_pinned_child
from cowork.services.artifact_roots import SERVER_PROJECT_FOLDERS

#: Folders deeper than this are not walked.
MAX_DEPTH = 32
#: Entries one walk examines, ordinary or not, before it stops.
MAX_LIST_ENTRIES = 10_000
#: Package caches, never ordinary at any depth, compared case-folded. The
#: server's own project folders (SERVER_PROJECT_FOLDERS) are not ordinary
#: either, at the project root only (ordinary_component), the one place the
#: server keeps them.
CACHE_NAMES = frozenset({"__pycache__", "node_modules"})
#: Files the operating system leaves in a folder, which are not the user's.
OS_FILES = frozenset({"desktop.ini", "thumbs.db", "icon\r"})
#: What a walk reads as "not there" by default: a missing entry, a parent that
#: stopped being a folder, and a link refused by ``O_NOFOLLOW``. Any other
#: error raises, so a reader never answers "no files" for "could not read".
ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP})
#: What a route that adds a file to a project's root says when the file's
#: name is not ordinary there, so no cloud turn would ever read the file.
NOT_A_PROJECT_FILE_NAME = (
    "This file name can't be a project file: Cowork's agent doesn't read files with this name."
)


def safe_component(name: str) -> bool:
    """A non-empty name that is not hidden and holds no separator. A ``:`` or
    ``\\`` is a valid name on the shared store, and the upload route takes it,
    so it stays ordinary here; a reader that needs more (a recipient's view of
    a shared project) checks that itself."""
    return bool(name) and not name.startswith(".") and not any(c in name for c in ("/", "\0"))


def ordinary_component(name: str, *, at_root: bool) -> bool:
    """A name an ordinary file or the folders above it may carry, ``at_root``
    when the name is in the project folder itself. Compared case-folded. The
    server keeps its own folders (SERVER_PROJECT_FOLDERS) at the project root
    only, so a user's folder deeper down with one of their names, such as
    ``docs/skills/``, holds ordinary files."""
    folded = name.casefold()
    if not safe_component(name) or folded in CACHE_NAMES or folded in OS_FILES:
        return False
    return not (at_root and folded in SERVER_PROJECT_FOLDERS)


def ordinary_path(parts: tuple[str, ...]) -> bool:
    """Whether a path below the project folder, by its components, can name an
    ordinary file."""
    return 0 < len(parts) <= MAX_DEPTH and all(
        ordinary_component(part, at_root=depth == 0) for depth, part in enumerate(parts)
    )


def utf8_path(parts: tuple[str, ...]) -> bool:
    """Whether every component of a path encodes as UTF-8. A name on disk
    that does not decodes with lone surrogates, which no JSON a reader writes
    can hold: the copy counts such a file without naming it, and a listing
    leaves it out."""
    try:
        for part in parts:
            part.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class OrdinaryFile(NamedTuple):
    """One ordinary file a walk found."""

    #: The pinned folder that holds the file, open while the walk yields it.
    directory: PinnedDir
    #: The file's own name in ``directory``.
    name: str
    #: Its path below the project folder, by component.
    parts: tuple[str, ...]
    #: Its ``lstat``: a regular file with one name.
    info: os.stat_result


@dataclass
class WalkState:
    """How far a walk got."""

    visited: int = 0
    #: The walk stopped at MAX_DEPTH or at its entry bound, so files past
    #: that point were not seen.
    truncated: bool = False


def walk_ordinary_files(
    root: PinnedDir,
    *,
    state: WalkState,
    max_entries: int | None = None,
    unreadable: frozenset[int] = ABSENT_ERRNOS,
    component: Callable[[str], bool] | None = None,
) -> Iterator[OrdinaryFile]:
    """Yield every ordinary file below the pinned project folder ``root``,
    depth first, until ``max_entries`` (MAX_LIST_ENTRIES when None) entries
    were examined.

    ``component``, for a reader whose view is narrower than the ordinary
    files (a recipient's view of a shared project), is a rule every name on
    the way to a file must pass too, at any depth: a folder whose name fails
    it is never walked into, so it spends no more of ``max_entries`` than its
    own entry.

    An entry whose ``lstat`` or folder open fails with an errno in
    ``unreadable`` is skipped; any other error raises.
    """
    bound = MAX_LIST_ENTRIES if max_entries is None else max_entries
    yield from _walk(root, (), state=state, max_entries=bound, unreadable=unreadable, component=component)


def _walk(
    directory: PinnedDir,
    prefix: tuple[str, ...],
    *,
    state: WalkState,
    max_entries: int,
    unreadable: frozenset[int],
    component: Callable[[str], bool] | None,
) -> Iterator[OrdinaryFile]:
    with dir_scandir(directory) as entries:
        for entry in entries:
            state.visited += 1
            if state.visited > max_entries:
                state.truncated = True
                return
            if not ordinary_component(entry.name, at_root=not prefix):
                continue
            if component is not None and not component(entry.name):
                continue
            parts = (*prefix, entry.name)
            try:
                info = dir_lstat(directory, entry.name)
                if stat.S_ISDIR(info.st_mode):
                    if len(parts) >= MAX_DEPTH:
                        state.truncated = True
                        continue
                    child = open_pinned_child(directory, entry.name)
                    try:
                        yield from _walk(
                            child, parts, state=state, max_entries=max_entries, unreadable=unreadable,
                            component=component,
                        )
                    finally:
                        child.close()
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    yield OrdinaryFile(directory=directory, name=entry.name, parts=parts, info=info)
            except OSError as error:
                # Moved, removed or swapped for a link mid-walk.
                if error.errno in unreadable:
                    continue
                raise
            if state.visited > max_entries:
                return
