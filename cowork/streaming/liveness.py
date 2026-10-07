"""Keep a turn alive while anton waits on a silent model call.

A model can think for minutes before it sends its first token, and the
OpenAI SDK drops the provider's keepalive comments. Nothing reaches the
turn's buffer in that time, so the UI's 300 s idle cut and the server's idle
watchdog both read the turn as hung.

anton reports what it is waiting on through ``ChatSession.model_calls``. While
that reports a call waiting on the provider and the buffer has been quiet for
``MODEL_WAIT_TICK_SECONDS``, ``ModelWaitTicker`` appends one ``model_wait``
progress frame, which resets every idle bound. A hung tool, a cell, an open
question or a frozen process reports no waiting call, so it still hits every
bound.

The frame is a live signal only. It goes straight to the buffer, never
through the formatter's ``event_sink``, so a reload does not replay it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from cowork.schemas.responses import Role
from cowork.streaming.buffer import StreamBuffer

logger = logging.getLogger(__name__)

# The progress phase anton, the cloud pod and the cowork UI agree on.
MODEL_WAIT_PHASE = "model_wait"
MODEL_WAIT_LABEL = "Still working"

# How long the buffer must stay quiet before a frame goes out, and how often
# the ticker checks. Frames therefore arrive every 20 to 25 s, well inside the
# UI's 300 s cut. Module constants rather than settings so tests can shrink
# them; the loop reads them on every pass.
MODEL_WAIT_TICK_SECONDS = 20.0
MODEL_WAIT_POLL_SECONDS = 5.0


class ModelCallSnapshot(Protocol):
    """The fields this module reads from anton's ``ModelCallSnapshot``.

    Read by attribute rather than imported: the installed anton can predate
    the type, and the ticker then never sees one.
    """

    message: str
    open_for_s: float


def model_wait_sse(*, message: str, waited_s: float | None, sequence_number: int | None = None) -> str:
    """One ``model_wait`` progress frame as an SSE string.

    Same shape as the formatter's generic progress frame, so a UI that does
    not know the phase still counts it as activity. ``sequence_number`` is the
    formatter's counter when the formatter relays a pod's frame; the in-process
    ticker has no counter to draw from and leaves it out.
    """
    data: dict[str, Any] = {"type": "response.in_progress"}
    if sequence_number is not None:
        data["sequence_number"] = sequence_number
    data.update({
        "thought_role": Role.thought_progress.value,
        "content": f"{MODEL_WAIT_LABEL}: {message}" if message else MODEL_WAIT_LABEL,
        "phase": MODEL_WAIT_PHASE,
        "message": message,
        "eta_seconds": waited_s,
        "tool_use_id": "",
        "at_ms": int(time.time() * 1000),
    })
    return f"event: response.in_progress\ndata: {json.dumps(data)}\n\n"


class ModelWaitTicker:
    """Writes ``model_wait`` frames for one in-process turn.

    The harness calls ``attach`` once its anton session exists and ``detach``
    when the turn ends. The producer runs the loop with ``running(buffer=...)``
    around the formatter loop, so the loop never outlives the turn and never
    runs beside the buffer's terminal write.
    """

    def __init__(self) -> None:
        self._session: object | None = None

    def attach(self, *, session: object) -> None:
        self._session = session

    def detach(self) -> None:
        self._session = None

    def _snapshot(self) -> ModelCallSnapshot | None:
        # Read late and by attribute: an anton without `model_calls` returns
        # None here, so the turn behaves exactly as it did without a ticker.
        tracker = getattr(self._session, "model_calls", None)
        if tracker is None:
            return None
        return tracker.snapshot()

    async def _tick(self, *, buffer: StreamBuffer) -> None:
        loop = asyncio.get_running_loop()
        last_seq = buffer.latest_seq
        quiet_since = loop.time()
        while not buffer.is_closed:
            await asyncio.sleep(MODEL_WAIT_POLL_SECONDS)
            if buffer.is_closed:
                return
            seq = buffer.latest_seq
            if seq != last_seq:
                last_seq, quiet_since = seq, loop.time()
                continue
            if loop.time() - quiet_since < MODEL_WAIT_TICK_SECONDS:
                continue
            snapshot = self._snapshot()
            if snapshot is None:
                continue
            await buffer.append("sse", {"sse": model_wait_sse(
                message=snapshot.message, waited_s=snapshot.open_for_s,
            )})
            last_seq, quiet_since = buffer.latest_seq, loop.time()

    async def _tick_safely(self, *, buffer: StreamBuffer) -> None:
        # A liveness aid, never a source of failure: an error here stops the
        # ticks and leaves the turn to the idle bounds it had without them.
        try:
            await self._tick(buffer=buffer)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[liveness] model-wait ticker failed; no more keep-alive frames this turn")

    @asynccontextmanager
    async def running(self, *, buffer: StreamBuffer) -> AsyncIterator[None]:
        task = asyncio.ensure_future(self._tick_safely(buffer=buffer))
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
