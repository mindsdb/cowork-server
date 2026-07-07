"""Shared types for the turn-streaming buffer.

A turn (one agent run for one user message) produces an ordered sequence
of records. Readers replay from a `seq` offset and then live-tail; the
terminal record tells them when to stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

# Terminal reasons a turn can end with. The buffer always writes exactly
# one terminal record as its last entry.
TerminalReason = Literal["completed", "cancelled", "error", "interrupted", "restart"]

_TERMINAL_TYPES = frozenset({"Done", "Cancelled", "Error", "Interrupted"})

# reason → terminal record type
REASON_TO_TYPE: dict[str, str] = {
    "completed": "Done",
    "cancelled": "Cancelled",
    "error": "Error",
    "interrupted": "Interrupted",
    "restart": "Interrupted",
}


@dataclass(frozen=True)
class TurnRecord:
    """One event in a turn's stream.

    `data` is the SSE-shape dict the client adapter consumes, so a
    replaying reader feeds it through the exact same reducer as the live
    stream (no parsing back from the wire).
    """
    seq: int
    ts: str
    type: str
    data: dict

    @property
    def is_terminal(self) -> bool:
        return self.type in _TERMINAL_TYPES


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
