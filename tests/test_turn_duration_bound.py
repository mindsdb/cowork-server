"""An in-flight turn must not run forever (ENG-1711 / ENG-1717).

A hung producer (wedged tool call, stalled model stream) otherwise keeps its
handle `is_running`, never writes a terminal record, and — on the in-process
FileStreamBuffer path — leaves a client tail blocking indefinitely, holding the
shared stream slot so sends wedge in every conversation. Boot recovery only
covers a process restart; a re-adopted crash-orphan sidecar keeps the same task
alive, so the bound is enforced at runtime in `RunRegistry`.
"""

from __future__ import annotations

import asyncio

import pytest

import sys

from cowork.streaming.registry import registry

# The submodule name `cowork.streaming.registry` is shadowed on the package by
# the re-exported `registry` instance, so reach the real module (to patch its
# duration-bound global) through sys.modules rather than attribute access.
_REGISTRY_MODULE = sys.modules["cowork.streaming.registry"]


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    registry.reset()


class _FakeBuffer:
    def __init__(self) -> None:
        self.closed: str | None = None

    async def append(self, type_, data):
        return 1

    async def close(self, reason, extra=None):
        self.closed = reason


async def test_hung_turn_is_bounded_and_sealed(monkeypatch):
    """A producer that never finishes is cancelled at the duration bound, and
    its CancelledError handler seals the buffer — the same path a user Stop
    takes — so a tail ends and the client releases its shared stream slot."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_DURATION_SECONDS", 0.05)
    buffer = _FakeBuffer()
    started = asyncio.Event()

    async def _hung_producer():
        started.set()
        try:
            await asyncio.sleep(3600)  # wedged: writes no terminal on its own
        except asyncio.CancelledError:
            await buffer.close("cancelled")  # mirror every real producer

    handle = await registry.start(
        conversation_id="conv-hung", turn_id=0, buffer=buffer,
        producer_coro=_hung_producer(),
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    # The 0.05 s bound elapses well within this wait.
    await asyncio.wait_for(
        asyncio.gather(handle.task, return_exceptions=True), timeout=5
    )

    assert handle.task.done()
    assert handle.is_running is False
    assert buffer.closed == "cancelled"


async def test_normal_turn_completes_without_the_bound_firing(monkeypatch):
    """A producer that finishes on its own is untouched by the bound."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_DURATION_SECONDS", 5)
    buffer = _FakeBuffer()

    async def _quick_producer():
        await buffer.close("completed")

    handle = await registry.start(
        conversation_id="conv-quick", turn_id=0, buffer=buffer,
        producer_coro=_quick_producer(),
    )
    await asyncio.wait_for(handle.task, timeout=5)
    assert buffer.closed == "completed"


async def test_external_cancel_still_propagates_through_the_bound(monkeypatch):
    """RunHandle.cancel must still cancel the producer through the wrapper, so
    an ordinary Stop keeps working and persists its partial turn."""
    monkeypatch.setattr(_REGISTRY_MODULE, "_MAX_TURN_DURATION_SECONDS", 3600)
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
