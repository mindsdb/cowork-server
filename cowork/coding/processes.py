from __future__ import annotations

import contextlib

import psutil


def terminate_descendants(parent_pid: int | None, timeout: float = 1.5) -> None:
    """Best-effort, cross-platform teardown of a coding runtime's children."""
    if parent_pid is None:
        return
    try:
        parent = psutil.Process(parent_pid)
        descendants = parent.children(recursive=True)
    except (psutil.Error, OSError):
        return
    for process in reversed(descendants):
        with contextlib.suppress(psutil.Error, OSError):
            process.terminate()
    _, alive = psutil.wait_procs(descendants, timeout=timeout)
    for process in alive:
        with contextlib.suppress(psutil.Error, OSError):
            process.kill()
