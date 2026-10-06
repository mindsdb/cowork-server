"""Bounded listing of the files under a folder.

Shared by the project files route and a chat's working folders. Callers read
the caps through this module, so a test that lowers one affects every caller.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A project directory Cowork allocated holds the agent's own output, so its
#: size is bounded by what the agent wrote. A folder the user chose can be a
#: repository or a home directory, and one request used to materialise every
#: path beneath it with a stat and a resolve each before returning anything.
MAX_LISTED_FILES = 2000

#: Entries examined, as opposed to returned. The cap above counts files the
#: caller may actually see, so on its own an unreadable subtree could still be
#: walked without limit before any of them were found.
MAX_EXAMINED_ENTRIES = 50_000


@dataclass
class WalkBudget:
    """Whether the entry ceiling, rather than the file cap, stopped the walk."""

    exhausted: bool = False


def iter_folder_files(base: Path, budget: WalkBudget) -> Iterator[Path]:
    """Candidate files under `base`, breadth-first, bounded by entries seen.

    Breadth-first so a truncated listing shows the user's own top-level files
    instead of whatever a depth-first walk reached inside the first large
    subdirectory it happened to enter.

    A symlinked directory is neither descended into nor listed, which is what
    `Path.rglob` did: it yielded the link itself and the caller skipped it as a
    directory. Every desktop project has `skills/<slug>` directory symlinks
    from `reconcile_project`, so descending would spend the budget on trees
    whose entries `file_meta` then discards for resolving outside `base`.

    Sets `budget.exhausted` when it stops at `MAX_EXAMINED_ENTRIES`.
    """
    queue: deque[Path] = deque([base])
    examined = 0
    while queue:
        try:
            entries = sorted(queue.popleft().iterdir())
        except OSError:
            continue
        for entry in entries:
            examined += 1
            if examined > MAX_EXAMINED_ENTRIES:
                budget.exhausted = True
                return
            try:
                if entry.is_symlink():
                    if entry.is_dir():
                        continue
                elif entry.is_dir():
                    queue.append(entry)
                    continue
            except OSError:
                continue
            yield entry


def file_meta(p: Path, base: Path) -> dict[str, Any] | None:
    """Listing row for `p` relative to `base`, or None if it is gone or
    resolves outside `base`."""
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    try:
        resolved = p.resolve()
        rel = resolved.relative_to(base.resolve())
    except ValueError:
        return None
    return {
        "path": str(rel).replace("\\", "/"),
        "name": p.name,
        "size": st.st_size,
        "modified": st.st_mtime,
        "is_dir": p.is_dir(),
    }
