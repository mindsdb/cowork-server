"""Capability tokens: bearer credentials that open one narrow door for one session.

The server mints a token for one ``purpose`` and the session that owns it,
hands the raw value once to the process that needs it, and keeps only the
token's SHA-256 hash. A route that serves one purpose checks a presented token
with ``verify``, and the owner revokes its tokens when the session ends.

The store lives in this process's memory, so every token dies with the server.

Purposes:

- ``inference``: a Code mode engine session's credential for the loopback
  inference proxy (``/api/v1/coding/inference/...``). The coding runtime
  manager (cowork.coding.runtime.RuntimeManager) mints one per engine session
  and revokes it when that session closes.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass
from typing import Literal

CapabilityPurpose = Literal["inference"]


@dataclass(frozen=True)
class CapabilityRecord:
    """What an issued token may do, and which session owns it."""

    purpose: CapabilityPurpose
    session_id: str


_lock = threading.Lock()
# SHA-256 of each live token, never the token itself.
_records: dict[bytes, CapabilityRecord] = {}


def _digest(*, token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def mint(*, purpose: CapabilityPurpose, session_id: str) -> str:
    """Issue a token for ``purpose``, owned by ``session_id``, and return it.
    The raw value is returned once; the store keeps only its hash."""
    token = secrets.token_urlsafe(32)
    with _lock:
        _records[_digest(token=token)] = CapabilityRecord(purpose=purpose, session_id=session_id)
    return token


def verify(*, token: str | None, purpose: CapabilityPurpose) -> CapabilityRecord | None:
    """The record of a live token issued for ``purpose``, or None.

    Every stored hash is compared in constant time, so the answer's timing
    says nothing about how close a guess came. A token issued for another
    purpose answers None.
    """
    if not token:
        return None
    digest = _digest(token=token)
    found: CapabilityRecord | None = None
    with _lock:
        for stored, record in _records.items():
            if hmac.compare_digest(stored, digest):
                found = record
    if found is None or found.purpose != purpose:
        return None
    return found


def revoke(*, token: str) -> bool:
    """Revoke one token. Returns whether it was live."""
    with _lock:
        return _records.pop(_digest(token=token), None) is not None


def revoke_session(*, session_id: str) -> int:
    """Revoke every token ``session_id`` owns. Returns how many were live."""
    with _lock:
        owned = [stored for stored, record in _records.items() if record.session_id == session_id]
        for stored in owned:
            del _records[stored]
    return len(owned)
