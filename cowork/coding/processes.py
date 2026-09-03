from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import PurePath

import psutil


def _end(processes: Iterable[psutil.Process], timeout: float) -> None:
    """Terminate, then kill what is still alive after ``timeout`` seconds."""
    pending = list(processes)
    for process in pending:
        with contextlib.suppress(psutil.Error, OSError):
            process.terminate()
    _, alive = psutil.wait_procs(pending, timeout=timeout)
    for process in alive:
        with contextlib.suppress(psutil.Error, OSError):
            process.kill()


def terminate_descendants(parent_pid: int | None, timeout: float = 1.5) -> None:
    """Best-effort, cross-platform teardown of a coding runtime's children."""
    if parent_pid is None:
        return
    try:
        parent = psutil.Process(parent_pid)
        descendants = parent.children(recursive=True)
    except (psutil.Error, OSError):
        return
    _end(reversed(descendants), timeout)


def terminate_command_trees(
    parent_pid: int | None,
    *,
    protected: Callable[[psutil.Process], bool],
    timeout: float = 1.5,
) -> int:
    """End every process tree under ``parent_pid`` except the ones ``protected`` claims.

    Each direct child of the parent is either a helper the runtime keeps
    between turns (``protected`` returns True) or the root of a command tree a
    turn left running, which goes together with all its descendants. A child
    that cannot be inspected is left alone. Returns how many processes ended.
    """
    if parent_pid is None:
        return 0
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children()
    except (psutil.Error, OSError):
        return 0
    victims: list[psutil.Process] = []
    for child in children:
        try:
            if protected(child):
                continue
            subtree = child.children(recursive=True)
        except (psutil.Error, OSError):
            continue
        # Children before their parent, so a shell cannot react to a dying
        # child by restarting it.
        victims.extend(reversed(subtree))
        victims.append(child)
    if victims:
        _end(victims, timeout)
    return len(victims)


def executable_name(path: str) -> str:
    """The bare program name of a command or argv[0]: no directory, no ``.exe``."""
    name = PurePath(path.replace("\\", "/")).name.lower()
    return name.removesuffix(".exe")


def runs_command(cmdline: Sequence[str], command: str, args: Sequence[str]) -> bool:
    """Whether ``cmdline`` is ``command args...`` (the program compared by bare name)."""
    if not cmdline or executable_name(cmdline[0]) != executable_name(command):
        return False
    return list(cmdline[1 : 1 + len(args)]) == list(args)
