"""Shared test doubles (importing conftest directly double-loads it)."""

import asyncio


class FakeRequest:
    """Minimal Request stub for calling route handlers directly. `headers` is
    empty unless a test sets it (only the org-mode bearer path reads it)."""

    def __init__(self, headers: dict | None = None) -> None:
        self.headers = headers or {}


class PausedHarness:
    """get_harness() stand-in whose answer waits on ``release`` mid-stream.

    ``answering`` is set once a turn has written its first frame and is
    waiting on the model; ``started`` counts the turns that reached the model.
    """

    id = "stub"

    def __init__(self) -> None:
        self.answering = asyncio.Event()
        self.release = asyncio.Event()
        self.started = 0

    def stream_response(self, **kwargs):
        self.started += 1
        return None

    async def formatter(self, stream, model, event_sink):
        from cowork.streaming import sse_frame

        yield sse_frame("response.created", {"type": "response.created"})
        self.answering.set()
        await self.release.wait()
        delta = {"type": "response.output_text.delta", "delta": "ok"}
        event_sink(delta["type"], delta)
        yield sse_frame(delta["type"], delta)
        yield sse_frame("response.completed", {"type": "response.completed", "response": {"output": []}})
