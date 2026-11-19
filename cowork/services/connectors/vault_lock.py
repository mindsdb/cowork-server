"""Shared per-record lock registry guarding read-modify-write access to a
single connection's vault record.

Every write path that does read-record -> mutate-in-memory -> save-record
on the SAME connection must serialize against every other one, or whichever
save() lands last silently reverts the other's change (e.g. a Google Picker
merge racing an OAuth token refresh on the same connection). This lives
outside ConnectionsService so google.py's OAuth callback and persist.py's
persist_connection() — neither of which goes through ConnectionsService —
can share the exact same lock instance for a given record, not just the
ones two ConnectionsService methods happen to call.

A record is its vault's root plus (engine, name): each person keeps their
own vault where connections are personal, and two people's same-named
connections are two records. Keyed by (engine, name) alone, one person's
delete dropped the lock another person's in-flight write held.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import NamedTuple


class _RecordKey(NamedTuple):
    vault_root: Path
    engine: str
    name: str


_locks: dict[_RecordKey, threading.Lock] = {}
_locks_guard = threading.Lock()


def lock_for(*, vault_root: Path, engine: str, name: str) -> threading.Lock:
    key = _RecordKey(vault_root, engine, name)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def discard_lock(*, vault_root: Path, engine: str, name: str) -> None:
    """Drop the lock for a deleted connection so `_locks` doesn't grow
    unbounded over the life of the process. Safe to call even while another
    thread holds the lock — it only removes the registry's reference, not
    the Lock object itself, so anyone still holding it keeps working."""
    with _locks_guard:
        _locks.pop(_RecordKey(vault_root, engine, name), None)
