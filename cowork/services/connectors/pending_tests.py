"""Custom connector tests waiting on the member: to allow a private host, or to
confirm what the API returned.

An entry holds the member's values until they answer, in this process's
memory only. Its id is the only handle on it: it goes to the member's form in
one unrecorded stream event, never into the conversation, a log or a listing,
so the agent cannot answer for the member. Ids are single use, expire, and
answer only to the scope that created them.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from cowork.db.scoped import TenantScope

PendingKind = Literal["approval", "confirm"]

_TTL_SECONDS = 10 * 60
#: Per scope; a new entry past this drops the scope's oldest.
_MAX_PER_SCOPE = 5


@dataclass
class PendingTest:
    """What a test needs to continue or to save once the member answers."""

    kind: PendingKind
    scope: TenantScope
    connector_id: str
    method: str | None
    name: str
    conversation_id: str | None
    form_id: str
    form_spec: dict[str, Any]
    test_request: dict[str, Any]
    #: The values the test runs with: the submission's, plus saved fields on a follow-up step.
    test_values: dict[str, Any]
    #: The values saved on confirm: the submission's own.
    credentials: dict[str, Any]
    secret_names: list[str]
    label: str = ""
    user_label: str = ""
    extends_name: str | None = None
    #: Set when an OAuth grant was saved before its test; confirm only lists it.
    grant_slug: str | None = None
    #: True when the form is handcrafted, so a pass also stores its definition.
    handcrafted: bool = False
    #: Private host approvals so far, by ``PrivateHost.key``.
    approved: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: On an approval entry: the key and addresses the member is asked about.
    asking: tuple[str, tuple[str, ...]] | None = None
    created_at: float = field(default_factory=time.monotonic)


class PendingTests:
    """The in-memory store of tests waiting on the member."""

    def __init__(self) -> None:
        self._entries: dict[str, PendingTest] = {}

    def put(self, entry: PendingTest) -> str:
        """Keep ``entry`` and return its new random id."""
        self._purge()
        same_scope = sorted(
            (e.created_at, key) for key, e in self._entries.items() if e.scope == entry.scope
        )
        while len(same_scope) >= _MAX_PER_SCOPE:
            _, oldest = same_scope.pop(0)
            self._entries.pop(oldest, None)
        entry_id = secrets.token_urlsafe(24)
        self._entries[entry_id] = entry
        return entry_id

    def pop(self, entry_id: str, kind: PendingKind, scope: TenantScope) -> PendingTest | None:
        """Remove and return the entry, when it is ``kind`` and ``scope``'s own.

        An entry another scope asks for, or of another kind, is left in place,
        so a guessed or replayed id cannot cancel someone else's test.
        """
        self._purge()
        entry = self._entries.get(entry_id)
        if entry is None or entry.kind != kind or entry.scope != scope:
            return None
        return self._entries.pop(entry_id)

    def _purge(self) -> None:
        cutoff = time.monotonic() - _TTL_SECONDS
        for key in [k for k, e in self._entries.items() if e.created_at < cutoff]:
            self._entries.pop(key, None)


pending_tests = PendingTests()
