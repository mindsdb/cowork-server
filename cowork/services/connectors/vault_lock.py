"""Shared per-(engine, name) lock registry guarding read-modify-write
access to a single connection's vault record.

Every write path that does read-record -> mutate-in-memory -> save-record
on the SAME connection must serialize against every other one, or whichever
save() lands last silently reverts the other's change (e.g. a Google Picker
merge racing an OAuth token refresh on the same connection). This lives
outside ConnectionsService so google.py's OAuth callback and persist.py's
persist_connection() — neither of which goes through ConnectionsService —
can share the exact same lock instance for a given (engine, name), not just
the ones two ConnectionsService methods happen to call.
"""
from __future__ import annotations

import threading

_locks: dict[tuple[str, str], threading.Lock] = {}
_locks_guard = threading.Lock()


def lock_for(engine: str, name: str) -> threading.Lock:
    key = (engine, name)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def discard_lock(engine: str, name: str) -> None:
    """Drop the lock for a deleted connection so `_locks` doesn't grow
    unbounded over the life of the process. Safe to call even while another
    thread holds the lock — it only removes the registry's reference, not
    the Lock object itself, so anyone still holding it keeps working."""
    with _locks_guard:
        _locks.pop((engine, name), None)
