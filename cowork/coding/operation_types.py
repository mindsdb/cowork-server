from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from cowork.coding.contracts import CodingEvent, CodingSession


class EventEmitter(Protocol):
    def __call__(
        self,
        session_id: str,
        event: CodingEvent,
        update: Callable[[CodingSession], None] | None = None,
    ) -> CodingEvent: ...


MaintenanceSession = Callable[[str, str], AbstractContextManager[CodingSession]]
GetSession = Callable[[str], CodingSession]
