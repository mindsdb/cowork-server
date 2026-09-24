"""Langfuse attribution for Codex requests passing through the inference proxy.

Anton stamps its own ``Langfuse-*`` headers; Codex does not know that
convention. It does send its own identity on every Responses call — a
``thread-id`` header and an ``x-codex-turn-metadata`` JSON header carrying the
turn — so the proxy translates those into the headers MindsHub Inference reads.
Without them a code-mode trace has no Request label and no session.

Codex's headers are internals, not a contract, so every piece here is optional:
a request missing or mangling them proxies exactly as before, just unlabelled.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict

from fastapi import Request

from cowork.build_info import surface

HARNESS = "codex"
# Mirrors anton's ``surface:`` tag prefix so one filter covers both harnesses.
SURFACE_TAG_PREFIX = "surface:"

HEADER_THREAD_ID = "thread-id"
HEADER_SESSION_ID = "session-id"
HEADER_TURN_METADATA = "x-codex-turn-metadata"

# Codex ids are UUIDs; anything else is treated as absent rather than forwarded.
_CODEX_ID = re.compile(r"^[A-Za-z0-9-]{1,128}$")
MAX_TURN_METADATA_BYTES = 4096


class TurnOrdinals:
    """Number each Codex turn within its thread: 1, 2, 3, ...

    The Request label reads ``turn-{n}``, but Codex identifies turns by UUID.
    Every request in one turn (retries, compaction) carries the same UUID, so
    each new UUID seen on a thread takes the next number.

    In memory only: after a server restart a resumed thread counts from 1
    again. The trace session still groups the whole thread.
    """

    def __init__(self, max_threads: int = 1024, turns_per_thread: int = 16) -> None:
        self._max_threads = max_threads
        self._turns_per_thread = turns_per_thread
        self._threads: OrderedDict[str, tuple[int, OrderedDict[str, int]]] = OrderedDict()
        self._lock = threading.Lock()

    def ordinal(self, thread_id: str, turn_id: str) -> int:
        with self._lock:
            count, turns = self._threads.pop(thread_id, (0, OrderedDict()))
            if turn_id not in turns:
                count += 1
                turns[turn_id] = count
                while len(turns) > self._turns_per_thread:
                    turns.popitem(last=False)
            self._threads[thread_id] = (count, turns)
            while len(self._threads) > self._max_threads:
                self._threads.popitem(last=False)
            return turns[turn_id]


_TURNS = TurnOrdinals()


def _codex_id(value: object) -> str | None:
    return value if isinstance(value, str) and _CODEX_ID.match(value) else None


def _turn_id(request: Request) -> str | None:
    raw = request.headers.get(HEADER_TURN_METADATA)
    if not raw or len(raw) > MAX_TURN_METADATA_BYTES:
        return None
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return _codex_id(metadata.get("turn_id")) if isinstance(metadata, dict) else None


def trace_headers(request: Request, turns: TurnOrdinals = _TURNS) -> dict[str, str]:
    """Build the ``Langfuse-*`` headers for one Codex request, or ``{}``."""
    thread_id = _codex_id(request.headers.get(HEADER_THREAD_ID)) or _codex_id(
        request.headers.get(HEADER_SESSION_ID)
    )
    if thread_id is None:
        return {}
    resolved_surface = surface()
    metadata: dict[str, object] = {"harness": HARNESS}
    tags = [HARNESS]
    if resolved_surface:
        metadata["surface"] = resolved_surface
        tags.append(f"{SURFACE_TAG_PREFIX}{resolved_surface}")
    if turn_id := _turn_id(request):
        metadata["turn_id"] = turns.ordinal(thread_id, turn_id)
    return {
        "Langfuse-Session-Id": thread_id,
        "Langfuse-Tags": ",".join(tags),
        "Langfuse-Metadata": json.dumps(metadata, separators=(",", ":")),
    }
