"""The clock behind the formatter's ``at_ms`` (stream_formatter._event, which
stamps every ``response.ask_user``) and the approvals-off stamp.

The app approves a held browser step for the person only when the step's
``response.ask_user`` event carries an ``at_ms`` later than the
``browser_approve_steps_off_at`` stamp (``heldAfterApprovalsOff`` in the app's
browserPrefs.js). Both are wall-clock readings, and the wall clock can step
backwards, for example on an NTP correction. So this module keeps the latest
``at_ms`` this process handed out, and the approvals-off stamp is never
earlier than it: a step held before approvals went off can't read as held
after. An equal stamp keeps the step the person's, since the app approves only
a later one. This orders stamps within one process; across replicas the order
still rests on the nodes' clock sync.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_latest_ms = 0


def event_at_ms() -> int:
    """Now in epoch milliseconds, for an event's ``at_ms``."""
    global _latest_ms
    now_ms = int(time.time() * 1000)
    with _lock:
        _latest_ms = max(_latest_ms, now_ms)
    return now_ms


def approvals_off_ms() -> int:
    """Now in epoch milliseconds, or the latest ``at_ms`` handed out if the
    clock has stepped back behind it."""
    with _lock:
        return max(int(time.time() * 1000), _latest_ms)
