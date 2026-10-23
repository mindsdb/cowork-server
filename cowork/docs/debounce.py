"""Run a call once its key has been quiet for a moment: bursts of writes collapse into one."""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Debouncer:
    """``schedule(key, fn)`` runs ``fn`` ``delay`` seconds after the last schedule for ``key``.

    A newer call for a key replaces the one waiting. Each run happens on a
    daemon timer thread, in a copy of the context ``schedule`` was called in
    (the turn's settings scope rides on a contextvar). A failing run is logged:
    the write it follows already landed, and the next one runs again.
    """

    def __init__(self, delay: float, *, what: str) -> None:
        self.delay = delay
        self.what = what
        self._lock = threading.Lock()
        self._waiting: dict[str, threading.Timer] = {}

    def schedule(self, key: str, fn: Callable[[], None]) -> None:
        ctx = contextvars.copy_context()
        timer = threading.Timer(self.delay, self._fire, args=(key, ctx, fn))
        timer.daemon = True
        with self._lock:
            old = self._waiting.get(key)
            if old is not None:
                old.cancel()
            self._waiting[key] = timer
        timer.start()

    def _fire(self, key: str, ctx: contextvars.Context, fn: Callable[[], None]) -> None:
        with self._lock:
            # A timer replaced (or cancelled) after it started firing finds
            # another in its place: only the newest runs.
            if self._waiting.get(key) is not threading.current_thread():
                return
            del self._waiting[key]
        try:
            ctx.run(fn)
        except Exception:  # noqa: BLE001 - see the class docstring
            logger.warning("%s failed for %s", self.what, key, exc_info=True)

    def cancel_all(self) -> None:
        with self._lock:
            timers, self._waiting = list(self._waiting.values()), {}
        for timer in timers:
            timer.cancel()

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._waiting)
