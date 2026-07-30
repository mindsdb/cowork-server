"""Heartbeat on a quiet stream.

A comment-only SSE frame is inert for the client: all three parsers in
api.js look for a `data:` line and `continue` when there is none.
"""

from __future__ import annotations

import asyncio

import pytest

from cowork.handlers.responses import sse_from_buffer


class _QuietBuffer:
    """Yields one record, then stays silent, then a terminal record."""

    def __init__(self, quiet_for: float) -> None:
        self._quiet_for = quiet_for

    async def tail(self, from_seq: int = 0):
        yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
        await asyncio.sleep(self._quiet_for)
        yield _Rec(is_terminal=True, sse=None)


class _Rec:
    def __init__(self, is_terminal: bool, sse: str | None) -> None:
        self.is_terminal = is_terminal
        self.data = {"sse": sse} if sse else {}


async def test_quiet_stream_emits_heartbeats(monkeypatch):
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 0.05)
    frames = [f async for f in sse_from_buffer(_QuietBuffer(quiet_for=0.3), 0)]
    assert any(f.startswith(": keepalive") for f in frames)
    assert sum(1 for f in frames if f.startswith(": keepalive")) >= 2
    # The real record still comes through, before the heartbeats.
    assert frames[0].startswith("event: response.created")


async def test_busy_stream_emits_no_heartbeat(monkeypatch):
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 5.0)
    frames = [f async for f in sse_from_buffer(_QuietBuffer(quiet_for=0.01), 0)]
    assert not any(f.startswith(": keepalive") for f in frames)


async def test_no_heartbeat_after_terminal_record(monkeypatch):
    """A heartbeat must never appear after the terminal record: once the
    buffer signals completion, the generator returns immediately instead
    of racing another keepalive tick."""
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 0.05)

    class _ImmediateTerminalBuffer:
        async def tail(self, from_seq: int = 0):
            yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
            yield _Rec(is_terminal=True, sse=None)

    frames = [f async for f in sse_from_buffer(_ImmediateTerminalBuffer(), 0)]
    assert frames == ["event: response.created\ndata: {}\n\n"]
    assert not any(f.startswith(": keepalive") for f in frames)


async def test_shield_prevents_dropping_the_record_racing_the_timeout(monkeypatch):
    """Regression test for the shield around the in-flight __anext__().

    `pending` outlives a single loop iteration: it is created once and
    re-awaited after every keepalive tick. Without asyncio.shield,
    wait_for cancels `pending` itself on timeout, so the next iteration
    awaits an already-cancelled future and raises CancelledError — the
    stream tears down loudly mid-turn instead of emitting a heartbeat and
    then delivering the record.

    Note the fix is the shield, NOT re-creating `pending` on the timeout
    branch: a fresh `ensure_future(records.__anext__())` there would
    abandon the in-flight one and silently lose the record it was about
    to resolve with.

    This test's buffer emits its real record after a delay just longer
    than one keepalive interval, so at least one timeout must race
    against the in-flight __anext__() before the record resolves.
    """
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 0.05)

    class _RacingBuffer:
        async def tail(self, from_seq: int = 0):
            await asyncio.sleep(0.12)
            yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
            yield _Rec(is_terminal=True, sse=None)

    frames = [f async for f in sse_from_buffer(_RacingBuffer(), 0)]
    real_frames = [f for f in frames if not f.startswith(": keepalive")]
    assert real_frames == ["event: response.created\ndata: {}\n\n"]


async def test_underlying_iterator_is_closed_when_consumer_abandons_early(monkeypatch):
    """The consumer can stop iterating before the terminal record (e.g. a
    client disconnect). The underlying async generator must still be
    closed so it does not leak or keep running in the background."""
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 5.0)

    closed = False

    class _TrackingBuffer:
        async def tail(self, from_seq: int = 0):
            nonlocal closed
            try:
                yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
                await asyncio.sleep(10)
                yield _Rec(is_terminal=True, sse=None)
            finally:
                closed = True

    gen = sse_from_buffer(_TrackingBuffer(), 0)
    first = await gen.__anext__()
    assert first.startswith("event: response.created")
    # A real scheduling gap, so the prefetched __anext__() task actually
    # starts and the underlying generator is mid-await when we close. Without
    # it the prefetch has never run, ag_running_async is false and aclose()
    # trivially succeeds — the bug this test names cannot fire.
    await asyncio.sleep(0.01)
    await gen.aclose()
    assert closed is True


async def test_cancelled_consumer_propagates_and_closes_the_iterator(monkeypatch):
    """The production disconnect shape: StreamingResponse cancels the task
    consuming this generator. CancelledError must propagate unchanged (anyio
    treats anything else as a hard error) and the underlying async generator
    must still end up closed."""
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 5.0)

    closed = False
    first_seen = asyncio.Event()

    class _TrackingBuffer:
        async def tail(self, from_seq: int = 0):
            nonlocal closed
            try:
                yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
                await asyncio.sleep(10)
                yield _Rec(is_terminal=True, sse=None)
            finally:
                closed = True

    async def consume():
        async for _ in sse_from_buffer(_TrackingBuffer(), 0):
            first_seen.set()

    task = asyncio.create_task(consume())
    await first_seen.wait()
    # Let the prefetched __anext__() get scheduled before cancelling.
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:  # pragma: no cover - only reached if cancellation was swallowed
        raise AssertionError("CancelledError did not propagate")
    assert task.cancelled(), "cancellation was replaced by another exception"
    assert closed is True


async def test_exception_from_tail_propagates_to_the_consumer(monkeypatch):
    """A failure inside buffer.tail() mid-stream must reach the client
    unreshaped, not be swallowed into a silent end-of-stream."""
    monkeypatch.setattr("cowork.handlers.responses.SSE_KEEPALIVE_SECONDS", 5.0)

    class _BoomBuffer:
        async def tail(self, from_seq: int = 0):
            yield _Rec(is_terminal=False, sse="event: response.created\ndata: {}\n\n")
            raise RuntimeError("buffer exploded")

    frames: list[str] = []
    with pytest.raises(RuntimeError, match="buffer exploded"):
        async for frame in sse_from_buffer(_BoomBuffer(), 0):
            frames.append(frame)

    assert frames == ["event: response.created\ndata: {}\n\n"]
