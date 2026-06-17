"""Turn-stream buffers — storage-agnostic interface + backends.

A `StreamBuffer` decouples the agent run (one producer) from any number
of readers (the live SSE response, a reconnecting client, a dev running
`tail -f`). Two properties this gives us:

  1. The work is decoupled from any single consumer — the producer runs
     as a detached task; a reader disconnecting never reaches it.
  2. State survives reconnects (and the page-close/reopen loop) — a
     returning client passes ``from_seq`` and `tail()` replays from that
     offset then continues into the live tail.

Backends:
  - ``FileStreamBuffer`` — JSONL file per turn. Used for desktop and the
    current single-instance cloud container. Ported from the proven
    bundled-server implementation (mindsdb/cowork `turn_buffer.py`).
  - ``RedisStreamBuffer`` — WIP. Backed by a Redis Stream so any web
    replica can tail a run executed by a separate worker. Wired when we
    move to multi-instance cloud (see class docstring). The interface is
    identical so the responses handler / endpoints never change.

Select the backend with ``COWORK_STREAM_BACKEND`` (see backend.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Iterator

from cowork.streaming.records import (
    REASON_TO_TYPE,
    TerminalReason,
    TurnRecord,
    now_iso,
)

logger = logging.getLogger(__name__)


# ── interface ────────────────────────────────────────────────────────


class StreamBuffer(ABC):
    """One-producer / many-readers ordered event buffer for a turn.

    Producer: ``await append(type, data)`` per event, then ``await
    close(reason)`` exactly once. Readers: ``async for rec in
    tail(from_seq)``.
    """

    @abstractmethod
    async def append(self, type_: str, data: dict) -> int:
        """Write one record, return its seq."""

    @abstractmethod
    async def close(self, reason: TerminalReason, extra: dict | None = None) -> None:
        """Write the terminal record exactly once (idempotent)."""

    @abstractmethod
    def tail(self, from_seq: int = 0) -> AsyncIterator[TurnRecord]:
        """Yield records with ``seq >= from_seq``, then live-tail to the
        terminal record. Never raises on consumer cancellation."""

    @property
    @abstractmethod
    def latest_seq(self) -> int:
        """Sequence of the NEXT record (== count written so far)."""

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        ...


# ── file backend (desktop + single-instance cloud) ───────────────────

_BAD_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(name: str) -> str:
    cleaned = _BAD_NAME_CHARS.sub("_", name or "").strip("_") or "_"
    return cleaned[:128]


def turn_buffer_path(streams_dir: Path, conversation_id: str, turn_id: int) -> Path:
    return streams_dir / _safe_segment(conversation_id) / f"turn_{int(turn_id):06d}.jsonl"


def read_records(path: Path, from_seq: int = 0) -> Iterator[TurnRecord]:
    """Read JSONL records with ``seq >= from_seq``. Tolerates a partial
    last line (producer crash mid-write) — skipped, not raised."""
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written record from a crash
                seq = int(obj.get("seq", -1))
                if seq < from_seq:
                    continue
                yield TurnRecord(
                    seq=seq,
                    ts=str(obj.get("ts", "")),
                    type=str(obj.get("type", "")),
                    data=dict(obj.get("data") or {}),
                )
    except OSError as exc:
        logger.warning("Could not read turn buffer at %s: %s", path, exc)


class FileStreamBuffer(StreamBuffer):
    """JSONL file buffer with a renewable-event live tail.

    The "many readers, one writer" signal: each append swaps in a fresh
    ``asyncio.Event`` and fires the old one, so a current waiter wakes
    without racing a future waiter. Disk write + flush per record, no
    fsync — losing the last few KB on a hard crash is acceptable for a UI
    replay log; the boot-time orphan sweep (recovery.py) handles the
    missing terminal.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._path.open("a", encoding="utf-8")
        self._seq = 0
        self._new_data = asyncio.Event()
        self._done = asyncio.Event()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def latest_seq(self) -> int:
        return self._seq

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def append(self, type_: str, data: dict) -> int:
        if self._closed:
            logger.warning("Append to closed buffer %s ignored", self._path)
            return self._seq
        record = {"seq": self._seq, "ts": now_iso(), "type": type_, "data": data}
        try:
            self._writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._writer.flush()
        except Exception:
            logger.exception("Failed to write turn record (path=%s)", self._path)
            return self._seq
        seq = self._seq
        self._seq += 1
        old, self._new_data = self._new_data, asyncio.Event()
        old.set()
        return seq

    async def close(self, reason: TerminalReason, extra: dict | None = None) -> None:
        if self._closed:
            return
        await self.append(REASON_TO_TYPE.get(reason, "Done"), {"reason": reason, **(extra or {})})
        try:
            self._writer.close()
        except Exception:
            pass
        self._closed = True
        self._done.set()
        old, self._new_data = self._new_data, asyncio.Event()
        old.set()

    async def tail(self, from_seq: int = 0) -> AsyncIterator[TurnRecord]:
        seen = from_seq - 1
        while True:
            # Snapshot the event BEFORE reading so an append between the
            # read and the wait can't be lost (it either shows on re-read
            # or fires the snapshot we're about to await).
            waiter = self._new_data
            emitted_terminal = False
            for rec in read_records(self._path, from_seq=seen + 1):
                seen = rec.seq
                yield rec
                if rec.is_terminal:
                    emitted_terminal = True
            if emitted_terminal or self._closed:
                return
            done_waiter = asyncio.create_task(self._done.wait())
            data_waiter = asyncio.create_task(waiter.wait())
            try:
                await asyncio.wait({done_waiter, data_waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                done_waiter.cancel()
                data_waiter.cancel()


# ── redis backend (cloud / multi-instance) — WIP ─────────────────────


class RedisStreamBuffer(StreamBuffer):
    """WIP — Redis Streams backend for multi-instance cloud.

    NOT YET IMPLEMENTED. Wired when we move off the single container so a
    run executed by a separate worker can be tailed from any web replica.
    The interface is identical to FileStreamBuffer, so the responses
    handler and HTTP endpoints don't change — only backend.py's factory.

    Design (Redis Streams map 1:1 onto this interface):
      key  = f"cowork:stream:{conversation_id}:{turn_id}"
      append(type, data) -> XADD key '*' seq <n> type <t> data <json>
                            (the stream entry id doubles as the seq;
                             use an explicit field too for portability)
      tail(from_seq)     -> XRANGE key from_seq '+'   (replay) then
                            XREAD BLOCK 0 STREAMS key <last-id>  (live)
      close(reason)      -> XADD terminal record + XTRIM MAXLEN / EXPIRE
      latest_seq         -> XLEN key
      is_closed          -> last entry type in _TERMINAL_TYPES
    Cancellation + in-flight status move to a shared run-status key
    (HSET) and a cancel channel (PUBLISH), since the producer lives in a
    worker, not the web process. See registry.py for the dispatch side.
    """

    _WIP = "RedisStreamBuffer is WIP — enabled when cowork moves to multi-instance cloud."

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(self._WIP)

    async def append(self, type_: str, data: dict) -> int:
        raise NotImplementedError(self._WIP)

    async def close(self, reason: TerminalReason, extra: dict | None = None) -> None:
        raise NotImplementedError(self._WIP)

    def tail(self, from_seq: int = 0) -> AsyncIterator[TurnRecord]:
        raise NotImplementedError(self._WIP)

    @property
    def latest_seq(self) -> int:
        raise NotImplementedError(self._WIP)

    @property
    def is_closed(self) -> bool:
        raise NotImplementedError(self._WIP)
