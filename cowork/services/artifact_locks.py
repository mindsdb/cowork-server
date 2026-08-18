"""Per-slug publish locks for the autopublish reconciler.

Placed at `<artifacts_base>/.locks/<slug>.lock`, deliberately OUTSIDE the artifact
folders: `services.artifacts._user_files` walks a folder recursively and only
filters housekeeping names at the top level, so a lock file inside an artifact
would count as user content — moving `content_mtime`, making an empty artifact
look non-empty, and (under `static/`) riding along into a fullstack bundle. The
`.locks` directory has no `metadata.json`, so the artifact enumerator never sees
it either.

The lock is released only after a SUCCESSFUL publish, or after a publish that
failed synchronously. It is deliberately NOT released when a publish wait times
out: `asyncio.to_thread` is not cancellable, so that upload is still running, and
releasing would let a second publisher into the same slug. A TTL is what prevents
a permanently stuck lock in that case.

A file lock rather than an asyncio.Lock because contention is genuinely
cross-process: `deployment/cowork-server/values.yaml` now runs `replicaCount: 2`
(the turn record buffer and the in-flight turn registry moved to Redis, so the
old single-replica pin is gone), and every replica mounts the same artifacts
tree. An in-process lock would only serialize one pod against itself.

On a shared network filesystem (EFS / RWX PVC) `O_CREAT|O_EXCL` is best-effort,
so two replicas can still both enter the same slug. That is tolerated: the real
defence against a duplicate published artifact is reusing `report_id` from
`.published.json`, which makes a second publish overwrite the same report rather
than mint a second URL.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LOCKS_DIRNAME = ".locks"


def _lock_path(artifacts_base: Path, slug: str) -> Path:
    return Path(artifacts_base) / LOCKS_DIRNAME / f"{slug}.lock"


def _create_exclusive(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    except OSError:
        logger.warning("Could not create publish lock %s", path, exc_info=True)
        return False
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        pass
    finally:
        os.close(fd)
    return True


def acquire(artifacts_base: Path, slug: str, *, ttl_s: float) -> bool:
    """Try to take the publish lock for one slug. True when acquired.

    A lock older than `ttl_s` is treated as abandoned and stolen. Any filesystem
    error is reported as "not acquired": skipping a publish this turn is safe (the
    next turn retries), publishing twice is not.
    """
    path = _lock_path(artifacts_base, slug)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Could not create locks dir %s", path.parent, exc_info=True)
        return False

    if _create_exclusive(path):
        return True

    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if age < ttl_s:
        return False
    try:
        os.unlink(path)
    except OSError:
        return False
    return _create_exclusive(path)


def release(artifacts_base: Path, slug: str) -> None:
    """Drop the lock. Call this after a publish succeeded or failed outright —
    never in a timeout path (see the module docstring). Silent when already gone.
    """
    try:
        _lock_path(artifacts_base, slug).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not release publish lock for %s", slug, exc_info=True)
