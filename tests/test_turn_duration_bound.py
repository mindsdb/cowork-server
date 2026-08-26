"""An in-flight turn must not run forever, but must not be reaped while it is
making progress or waiting on the user (ENG-1711 / ENG-1717).

A hung producer (wedged tool call, stalled model stream) keeps its handle
`is_running`, never writes a terminal record, and — on the in-process
FileStreamBuffer path — leaves a client tail blocking indefinitely, holding the
shared stream slot so sends wedge in every conversation. Boot recovery only
covers a process restart; a re-adopted crash-orphan sidecar keeps the same task
alive, so the bound is enforced at runtime in `RunRegistry`.

The bound is on IDLE time (no buffer record written), not total duration: a
turn that streams frames — or one blocked on an `ask_user` card and then
resuming — keeps advancing `buffer.latest_seq` and is never reaped, so a long
deliberate turn is not killed mid-conversation.
"""

from __future__ import annotations

import asyncio

import pytest

import sys

from cowork.streaming.registry import RunHandle, registry

# The submodule name `cowork.streaming.registry` is shadowed on the package by
# the re-exported `registry` instance, so reach the real module (to patch its
# idle-bound globals) through sys.modules rather than attribute access.
_REGISTRY_MODULE = sys.modules["cowork.streaming.registry"]


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    # Sample often so the tests' small idle windows are detected promptly.
    monkeypatch.setattr(_REGISTRY_MODULE, "_IDLE_POLL_SECONDS", 0.01)
    yield
    registry.reset()


class _FakeBuffer:
    """Faithful to the progress signal the watchdog reads: `latest_seq` counts
    records written, advancing on every `append` and staying flat while the
    producer is silent (a wedge, or an open `ask_user` card)."""

    def __init__(self) -> None:
        self.closed: str | None = None
        self._seq = 0

    @property
    def latest_seq(self) -> int:
        return self._seq

    async def append(self, type_, data):
        self._seq += 1
        return self._seq

    async def close(self, reason, extra=None):
        self.closed = reason


async def test_hung_turn_is_bounded_and_sealed(monkeypatch):
    """A producer that goes fully silent is cancelled at the idle bound, and its
    CancelledError handler seals the buffer — the same path a user Stop takes —
    so a tail ends and the client releases its shared stream slot."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_IDLE_SECONDS", 0.05)
    buffer = _FakeBuffer()
    started = asyncio.Event()

    async def _hung_producer():
        started.set()
        try:
            await asyncio.sleep(3600)  # wedged: writes no record on its own
        except asyncio.CancelledError:
            await buffer.close("cancelled")  # mirror every real producer

    handle = await registry.start(
        conversation_id="conv-hung", turn_id=0, buffer=buffer,
        producer_coro=_hung_producer(),
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    # The 0.05 s idle window elapses well within this wait.
    await asyncio.wait_for(
        asyncio.gather(handle.task, return_exceptions=True), timeout=5
    )

    assert handle.task.done()
    assert handle.is_running is False
    assert buffer.closed == "cancelled"


async def test_normal_turn_completes_without_the_bound_firing(monkeypatch):
    """A producer that finishes on its own is untouched by the bound."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_IDLE_SECONDS", 5)
    buffer = _FakeBuffer()

    async def _quick_producer():
        await buffer.close("completed")

    handle = await registry.start(
        conversation_id="conv-quick", turn_id=0, buffer=buffer,
        producer_coro=_quick_producer(),
    )
    await asyncio.wait_for(handle.task, timeout=5)
    assert buffer.closed == "completed"


async def test_a_progressing_turn_is_not_reaped_past_the_window(monkeypatch):
    """The regression the idle bound exists to prevent: a turn whose total
    wall-clock exceeds the window is NOT reaped as long as it keeps writing
    records. This is the `ask_user`/long-deliberation case — each frame (a
    question, an answer's continuation) resets the idle window, so human wait
    never accumulates toward the bound."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_IDLE_SECONDS", 0.05)
    buffer = _FakeBuffer()

    async def _progressing_producer():
        # ~0.18 s total, far past the 0.05 s window, but each silent gap
        # (0.03 s) stays under it — so the watchdog must never fire.
        for _ in range(6):
            await buffer.append("sse", {})
            await asyncio.sleep(0.03)
        await buffer.close("completed")

    handle = await registry.start(
        conversation_id="conv-progress", turn_id=0, buffer=buffer,
        producer_coro=_progressing_producer(),
    )
    await asyncio.wait_for(handle.task, timeout=5)
    assert buffer.closed == "completed"  # completed on its own, not "cancelled"


async def test_external_cancel_still_propagates_through_the_bound(monkeypatch):
    """RunHandle.cancel must still cancel the producer through the wrapper, so
    an ordinary Stop keeps working and persists its partial turn."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_IDLE_SECONDS", 3600)
    buffer = _FakeBuffer()
    started = asyncio.Event()

    async def _producer():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await buffer.close("cancelled")

    handle = await registry.start(
        conversation_id="conv-cancel", turn_id=0, buffer=buffer,
        producer_coro=_producer(),
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    assert await handle.cancel() is True
    assert buffer.closed == "cancelled"


async def test_cancel_reports_success_even_when_the_teardown_raises():
    """A cancel that stops the turn answers True even if unwinding then fails.

    Every producer catches CancelledError to persist its partial turn and close
    its buffer. If either step raises, the exception escapes the producer and
    lands on the `await self.task` inside cancel(), which used to translate it
    into False. That is the same answer as "there was nothing to cancel", for a
    turn that had just been stopped.

    Built on a bare RunHandle rather than through `registry.start`: the unit
    here is `RunHandle.cancel`, and routing through the idle-bound wrapper made
    which branch runs depend on task scheduling.

    The `logger.exception` the fix also adds is deliberately not asserted. The
    record never reaches a handler under the full suite, including caplog's own,
    while it does in isolation, so asserting it buys a test that fails for
    reasons unrelated to the behaviour. The return value is the contract; the
    log is diagnostics.
    """
    async def _producer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Stands in for persist() or buffer.close() failing on the
            # cancellation path: a DB write or a Redis call, either of which can
            # fail independently of the cancel itself succeeding.
            raise RuntimeError("persist failed while unwinding")

    task = asyncio.ensure_future(_producer())
    await asyncio.sleep(0)  # let it reach the sleep, so there is a cancel to make
    handle = RunHandle(
        conversation_id="conv-cancel-teardown-raises",
        turn_id=0,
        buffer=_FakeBuffer(),
        task=task,
    )

    assert await handle.cancel() is True
    assert handle.is_running is False


async def test_cancel_still_reports_false_for_a_turn_that_already_finished():
    """The one meaning False keeps: there was no running turn to stop.

    Pinned beside the case above so the two cannot collapse into each other —
    that is exactly what made a stopped turn indistinguishable from a finished
    one.
    """
    async def _producer():
        return None

    task = asyncio.ensure_future(_producer())
    await asyncio.wait_for(task, timeout=5)
    handle = RunHandle(
        conversation_id="conv-cancel-already-done",
        turn_id=0,
        buffer=_FakeBuffer(),
        task=task,
    )

    assert await handle.cancel() is False


# The idle bound is env-configurable (COWORK_MAX_TURN_IDLE_SECONDS) as a
# pressure-release valve for deployments with legitimately long silent tool
# calls; a bad value must never disable the bound (ENG-1717).
class TestIdleBoundConfig:
    def test_defaults_to_600_when_unset(self, monkeypatch):
        monkeypatch.delenv("COWORK_MAX_TURN_IDLE_SECONDS", raising=False)
        assert _REGISTRY_MODULE._idle_bound_seconds() == 600

    def test_honors_a_valid_override(self, monkeypatch):
        monkeypatch.setenv("COWORK_MAX_TURN_IDLE_SECONDS", "1800")
        assert _REGISTRY_MODULE._idle_bound_seconds() == 1800

    def test_falls_back_on_unparseable(self, monkeypatch):
        monkeypatch.setenv("COWORK_MAX_TURN_IDLE_SECONDS", "soon")
        assert _REGISTRY_MODULE._idle_bound_seconds() == 600

    def test_falls_back_on_non_positive(self, monkeypatch):
        monkeypatch.setenv("COWORK_MAX_TURN_IDLE_SECONDS", "0")
        assert _REGISTRY_MODULE._idle_bound_seconds() == 600
        monkeypatch.setenv("COWORK_MAX_TURN_IDLE_SECONDS", "-30")
        assert _REGISTRY_MODULE._idle_bound_seconds() == 600
