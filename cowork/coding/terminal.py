from __future__ import annotations

import threading
import time
from collections import deque

from cowork.coding.contracts import TerminalChunk, TerminalPage, TerminalStatus

_MAX_BUFFER_CHARS = 4 * 1024 * 1024


class TerminalBuffer:
    """Bounded, reconnectable terminal output for one task-owned shell."""

    def __init__(self, process_id: str) -> None:
        self.process_id = process_id
        self._condition = threading.Condition()
        self._items: deque[TerminalChunk] = deque()
        self._buffer_chars = 0
        self._next_seq = 0
        self._status = TerminalStatus.running
        self._exit_code: int | None = None
        self._error: str | None = None

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._status == TerminalStatus.running

    def append(self, data_base64: str, stream: str, cap_reached: bool) -> None:
        with self._condition:
            if self._status != TerminalStatus.running:
                return
            self._next_seq += 1
            chunk = TerminalChunk(
                seq=self._next_seq,
                data_base64=data_base64,
                stream="stderr" if stream == "stderr" else "stdout",
                cap_reached=cap_reached,
            )
            self._items.append(chunk)
            self._buffer_chars += len(data_base64)
            while self._buffer_chars > _MAX_BUFFER_CHARS and len(self._items) > 1:
                self._buffer_chars -= len(self._items.popleft().data_base64)
            self._condition.notify_all()

    def finish(self, exit_code: int | None, error: str | None) -> None:
        with self._condition:
            self._exit_code = exit_code
            self._error = error
            self._status = TerminalStatus.failed if error else TerminalStatus.exited
            self._condition.notify_all()

    def page(self, after: int = 0) -> TerminalPage:
        with self._condition:
            return self._page_locked(after)

    def wait(self, after: int, timeout: float) -> TerminalPage:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._next_seq <= after and self._status == TerminalStatus.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._page_locked(after)

    def _page_locked(self, after: int) -> TerminalPage:
        first_seq = self._items[0].seq if self._items else self._next_seq
        return TerminalPage(
            process_id=self.process_id,
            status=self._status,
            items=[item for item in self._items if item.seq > after],
            first_seq=first_seq,
            next_seq=self._next_seq,
            exit_code=self._exit_code,
            error=self._error,
        )
