"""Log when the event loop or a thread limiter stops making progress.

A blocking call on the event loop freezes every request on the process,
liveness included, and the kubelet then restarts the pod. Without this, the
first sign is the restart, and the culprit has to be guessed from gaps in the
logs. The watchdog is a daemon thread that pings the loop once a second. When
a ping goes unanswered for STALL_AFTER_S it logs ``event_loop_stalled`` with
the loop thread's stack and the requests in flight, and when the loop answers
again it logs ``event_loop_recovered`` with how long the stall lasted. It
also logs ``limiter_saturated`` when a limiter requests queue on stays full
for SATURATED_AFTER_S.

The thread reads only memory: no file store, no database, no settings. A
watchdog that could block on what it watches would go quiet exactly when it
is needed.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

import anyio

from cowork.common import request_logging

logger = logging.getLogger(__name__)

PING_EVERY_S = 1.0
STALL_AFTER_S = 2.0
SATURATED_AFTER_S = 5.0
# Twice a ping, so a limiter full for SATURATED_AFTER_S is reported within
# half a second of it.
LIMITER_CHECK_EVERY_S = 0.5
_STACK_FRAMES = 40


def _frame_lines(frame) -> str:
    """The stack as ``path:line in function``, innermost last. Source text and
    local values stay out, since a frame's locals can hold request data."""
    lines = []
    while frame is not None and len(lines) < _STACK_FRAMES:
        code = frame.f_code
        path = Path(code.co_filename)
        parts = path.parts
        # Name the module path from the package down, not the install prefix.
        for package in ("cowork", "anton", "starlette", "fastapi", "anyio", "sqlalchemy", "asyncio"):
            if package in parts:
                path = Path(*parts[parts.index(package):])
                break
        else:
            path = Path(path.name)
        lines.append(f"{path}:{frame.f_lineno} in {code.co_name}")
        frame = frame.f_back
    return "\n".join(reversed(lines))


class LoopWatchdog:
    """Construct and start it on the loop it watches, which is how it learns
    that loop's thread and limiters."""

    def __init__(self, limiters: dict[str, anyio.CapacityLimiter]):
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._limiters = limiters
        self._full_since: dict[str, float] = {}
        self._reported: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cowork-loop-watchdog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=PING_EVERY_S * 2)

    def _run(self) -> None:
        # When the loop last answered a ping. A stall is timed from here, not
        # from the unanswered ping, so the length logged is never shorter
        # than the stall.
        last_seen = time.monotonic()
        while not self._stop.is_set():
            answered = threading.Event()
            try:
                self._loop.call_soon_threadsafe(answered.set)
            except RuntimeError:
                return  # the loop closed
            if not answered.wait(STALL_AFTER_S):
                self._stalled(last_seen)
                while not answered.wait(LIMITER_CHECK_EVERY_S):
                    if self._stop.is_set():
                        return
                    self._check_limiters()
                logger.warning(
                    "Event loop answered again",
                    extra={
                        "event": "event_loop_recovered",
                        "duration_ms": round((time.monotonic() - last_seen) * 1000),
                    },
                )
            last_seen = time.monotonic()
            pinged = time.monotonic()
            self._check_limiters()
            while not self._stop.wait(LIMITER_CHECK_EVERY_S):
                self._check_limiters()
                if time.monotonic() - pinged >= PING_EVERY_S:
                    break

    def _stalled(self, last_seen: float) -> None:
        frame = sys._current_frames().get(self._loop_thread)
        logger.error(
            "Event loop stalled",
            extra={
                "event": "event_loop_stalled",
                "duration_ms": round((time.monotonic() - last_seen) * 1000),
                "stack": _frame_lines(frame) if frame is not None else "",
                "in_flight": request_logging.in_flight_summary(),
            },
        )

    def _check_limiters(self) -> None:
        now = time.monotonic()
        for name, limiter in self._limiters.items():
            if limiter.borrowed_tokens < limiter.total_tokens:
                self._full_since.pop(name, None)
                self._reported.discard(name)
                continue
            since = self._full_since.setdefault(name, now)
            if now - since >= SATURATED_AFTER_S and name not in self._reported:
                self._reported.add(name)
                logger.warning(
                    "Thread limiter stayed full",
                    extra={
                        "event": "limiter_saturated",
                        "limiter": name,
                        "count": limiter.statistics().tasks_waiting,
                        "duration_ms": round((now - since) * 1000),
                    },
                )
