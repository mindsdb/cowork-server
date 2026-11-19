"""Where a test's code ran: on the event loop, or in a worker thread."""
from __future__ import annotations

import asyncio


def on_the_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
