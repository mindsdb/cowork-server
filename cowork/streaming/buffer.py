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
  - ``RedisStreamBuffer`` — a Redis Stream per turn, so any web replica
    can replay and tail a turn another replica is streaming. Used for
    multi-instance cloud. The interface is identical so the responses
    handler / endpoints never change.

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

from cowork.turnqueue.redis_client import get_redis
from cowork.streaming.records import (
    REASON_TO_TYPE,
    TerminalReason,
    TurnRecord,
    now_iso,
)

logger = logging.getLogger(__name__)


def _is_final_client_frame(type_: str, data: dict) -> bool:
    if type_ in REASON_TO_TYPE.values():
        return True
    if type_ != "sse":
        return False
    return any(
        line in {"event: response.completed", "event: response.failed", "event: response.cancelled"}
        for line in (data.get("sse") or "").splitlines()
    )


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
    def is_finishing(self) -> bool:
        """A final client frame is being written; the next turn may wait for
        the internal terminal record instead of receiving a refusal."""
        return getattr(self, "_finishing", False)

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


def conversation_dir(streams_dir: Path, conversation_id: str) -> Path:
    """Directory holding all of a conversation's turn buffers."""
    return streams_dir / _safe_segment(conversation_id)


def turn_buffer_path(streams_dir: Path, conversation_id: str, turn_id: int) -> Path:
    return conversation_dir(streams_dir, conversation_id) / f"turn_{int(turn_id):06d}.jsonl"


def _parse_record(line: str | bytes) -> TurnRecord | None:
    """One JSONL line as a record, or None for a blank or half-written line."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None  # half-written record from a crash
    return TurnRecord(
        seq=int(obj.get("seq", -1)),
        ts=str(obj.get("ts", "")),
        type=str(obj.get("type", "")),
        data=dict(obj.get("data") or {}),
    )


def read_records(path: Path, from_seq: int = 0) -> Iterator[TurnRecord]:
    """Read JSONL records with ``seq >= from_seq``. Tolerates a partial
    last line (producer crash mid-write) — skipped, not raised."""
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = _parse_record(line)
                if record is not None and record.seq >= from_seq:
                    yield record
    except OSError as exc:
        logger.warning("Could not read turn buffer at %s: %s", path, exc)


class _NewRecords:
    """Reads a turn file a little at a time: each call parses only the
    complete lines written since the last one.

    A reader that re-read the whole file on every wake would parse a turn of
    N records about N * N / 2 times, on the event loop. This one keeps the
    byte offset it has read up to, and holds back a last line that has no
    newline yet until the rest of it is written.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._partial = b""

    def read(self) -> list[TurnRecord]:
        try:
            with self._path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning("Could not read turn buffer at %s: %s", self._path, exc)
            return []
        self._offset += len(chunk)
        *lines, self._partial = (self._partial + chunk).split(b"\n")
        return [record for record in map(_parse_record, lines) if record is not None]


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
        # A new turn starts with an empty file. The path is named by the turn
        # number, which is the conversation's message count, so a turn that
        # saved no message leaves its file where the next turn writes; its
        # records would be replayed to the next turn's readers, which stop at
        # its terminal record. Only this process writes these files, and the
        # registry opens one only once no turn of the conversation is
        # answering, so whatever the file held had already ended. Opened to
        # append, so every record still lands at the file's end.
        self._writer = self._path.open("a", encoding="utf-8")
        self._writer.truncate(0)
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
        if _is_final_client_frame(type_, data):
            self._finishing = True
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
        new_records = _NewRecords(self._path)
        while True:
            # Snapshot the event BEFORE reading so an append between the
            # read and the wait can't be lost (it either shows on re-read
            # or fires the snapshot we're about to await).
            waiter = self._new_data
            emitted_terminal = False
            for rec in new_records.read():
                if rec.seq < from_seq:
                    continue
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


# Buffers exist for reconnect and replay, not as a record of the turn. The
# transcript is in Postgres, so an hour past the last write is generous.
REDIS_BUFFER_TTL_SECONDS = 3600

# How long a live tail waits with no new record before it decides the turn is
# not coming back. Generous: a long tool call legitimately produces nothing for
# minutes, and the cost of being wrong is ending a turn the user was watching.
REDIS_TAIL_IDLE_TIMEOUT_SECONDS = 300


def stream_key(conversation_id: str, turn_id: int) -> str:
    return f"cowork:stream:{conversation_id}:{int(turn_id)}"


class RedisStreamBuffer(StreamBuffer):
    """Redis Streams backend for multi-instance cloud.

    One Redis Stream per turn, so a turn streamed into Redis by one replica
    can be replayed and followed by any other. The interface matches
    FileStreamBuffer exactly, so the responses handler and the HTTP
    endpoints don't change, only backend.py's factory.

    ``seq`` is an explicit field rather than the Redis entry id: the id is a
    timestamp, and a count-based seq would jump after XTRIM.
    """

    def __init__(self, conversation_id: str, turn_id: int) -> None:
        self.key = stream_key(conversation_id, turn_id)
        self._conversation_id = conversation_id
        self._turn_id = int(turn_id)
        self._next_seq = 0
        self._closed = False
        self._write_lock = asyncio.Lock()

    async def append(self, type_: str, data: dict) -> int:
        # Serialised: seq is assigned in this process, so two concurrent
        # appends could otherwise claim the same number.
        async with self._write_lock:
            if _is_final_client_frame(type_, data):
                self._finishing = True
            seq = self._next_seq
            self._next_seq += 1
            r = get_redis()
            await r.xadd(self.key, {
                "seq": str(seq),
                "ts": now_iso(),
                "type": type_,
                "data": json.dumps(data or {}),
            })
            await r.expire(self.key, REDIS_BUFFER_TTL_SECONDS)
            return seq

    async def close(self, reason: TerminalReason, extra: dict | None = None) -> None:
        if self._closed:
            return
        # Flip _closed only AFTER the terminal record is appended: is_closed is
        # the seal backstop's (handlers/responses.py) proof a terminal exists, so
        # if the append fails the buffer must stay open for the backstop to
        # re-seal it. Matches FileStreamBuffer.close ordering.
        await self.append(REASON_TO_TYPE[reason], {"reason": reason, **(extra or {})})
        self._closed = True

    _BLOCK_MS = 5000

    @staticmethod
    def _record(fields: dict) -> TurnRecord:
        return TurnRecord(
            seq=int(fields.get("seq", -1)),
            ts=str(fields.get("ts", "")),
            type=str(fields.get("type", "")),
            data=json.loads(fields.get("data") or "{}"),
        )

    async def tail(self, from_seq: int = 0) -> AsyncIterator[TurnRecord]:
        r = get_redis()
        last_id = "0-0"
        # Replay. Filtered on the seq field rather than sliced by entry id:
        # ids are timestamps and carry no relation to from_seq.
        for entry_id, fields in await r.xrange(self.key):
            last_id = entry_id
            rec = self._record(fields)
            if rec.seq < from_seq:
                continue
            yield rec
            if rec.is_terminal:
                return
        # Live. Handed off by entry id, so an append between the XRANGE above
        # and the first XREAD is picked up rather than skipped.
        idle_ms = 0
        while True:
            resp = await r.xread({self.key: last_id}, block=self._BLOCK_MS)
            if not resp:
                idle_ms += self._BLOCK_MS
                if idle_ms >= REDIS_TAIL_IDLE_TIMEOUT_SECONDS * 1000:
                    # Nothing is writing this turn. The producer replica died
                    # mid-turn, so no terminal record is ever coming and every
                    # reader would wait here forever. Write one, so this reader
                    # and any other stop, and the client sees a failed turn
                    # rather than a spinner.
                    async for rec in self._terminate_as_orphan():
                        yield rec
                    return
                continue
            idle_ms = 0
            for _key, entries in resp:
                for entry_id, fields in entries:
                    last_id = entry_id
                    rec = self._record(fields)
                    if rec.seq < from_seq:
                        continue
                    yield rec
                    if rec.is_terminal:
                        return

    async def clear_if_ended(self) -> None:
        """Delete this turn's stream when the turn that wrote it has ended.

        A new turn writes here when the turn before it at the same number
        saved no message, and its readers would replay that turn's records and
        stop at its terminal record. A stream with no terminal record may
        still be live on another replica, so it is left alone.
        """
        r = get_redis()
        entries = await r.xrevrange(self.key, count=1)
        if entries and self._record(entries[0][1]).is_terminal:
            await r.delete(self.key)

    async def refresh(self) -> None:
        """Load latest_seq and is_closed from Redis.

        A replica that did not write this turn has no in-process state, so
        the /in-flight and /tail endpoints call this before answering.
        """
        entries = await get_redis().xrevrange(self.key, count=1)
        if not entries:
            # No stream: either it expired, or the conversation was truncated and
            # its buffers deleted. Reporting it as open would have /in-flight
            # answer "running, seq 0" for a turn nobody is writing.
            self._next_seq = 0
            self._closed = True
            return
        _entry_id, fields = entries[0]
        rec = self._record(fields)
        self._next_seq = rec.seq + 1
        self._closed = rec.is_terminal

    async def _terminate_as_orphan(self) -> AsyncIterator[TurnRecord]:
        """Close an abandoned turn, and yield the terminal record.

        Written by whichever reader notices, since the replica that owned the
        turn is the one that is gone. Idempotent by construction: the write is
        conditional on the stream still having no terminal record, so a second
        reader arriving at the same moment replays this one rather than adding
        another.
        """
        logger.warning(
            "Turn buffer %s went quiet for %ss with no terminal record; ending it",
            self.key, REDIS_TAIL_IDLE_TIMEOUT_SECONDS,
        )
        r = get_redis()
        entries = await r.xrevrange(self.key, count=1)
        if entries and self._record(entries[0][1]).is_terminal:
            yield self._record(entries[0][1])
            return
        self._next_seq = (self._record(entries[0][1]).seq + 1) if entries else 0
        self._closed = False   # so close() writes rather than returning early
        await self.close("interrupted", {"error": "the worker running this turn stopped responding"})
        latest = await r.xrevrange(self.key, count=1)
        if latest:
            yield self._record(latest[0][1])

    @property
    def latest_seq(self) -> int:
        return self._next_seq

    @property
    def is_closed(self) -> bool:
        return self._closed
