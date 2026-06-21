"""Process-global rendezvous for mid-turn file/folder selections.

The agent's ``select_path`` tool blocks inside the in-flight turn awaiting a
user choice. The choice arrives out-of-band on a separate HTTP request
(``POST /responses/selection``). This gateway bridges the two: the streaming
elicitor ``open()``s a future keyed by ``(conversation_id, request_id)`` and
awaits it; the endpoint ``resolve()``s that future with the user's pick.

Single asyncio loop (the server's), so plain dict access needs no lock — the
endpoint coroutine and the awaiting turn run on the same loop.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# A resolved selection is the chosen path, or None when the user cancelled.
Selection = "str | None"


class SelectionGateway:
    """Maps in-flight selection requests to the futures their tools await."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], asyncio.Future] = {}

    def open(self, conversation_id: str, request_id: str) -> asyncio.Future:
        """Register a pending selection and return the future to await.

        A duplicate ``request_id`` (should not happen — ids are random per
        request) cancels the stale future first so nothing leaks.
        """
        key = (conversation_id, request_id)
        stale = self._pending.get(key)
        if stale is not None and not stale.done():
            stale.cancel()
        future = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        return future

    def resolve(self, conversation_id: str, request_id: str, selection: str | None) -> bool:
        """Deliver the user's choice (or None to cancel). Returns False when
        no matching request is pending — e.g. the turn ended or it was a
        stale/duplicate submit."""
        future = self._pending.get((conversation_id, request_id))
        if future is None or future.done():
            return False
        future.set_result(selection)
        return True

    def close(self, conversation_id: str, request_id: str) -> None:
        """Drop a request once its tool has consumed the result."""
        self._pending.pop((conversation_id, request_id), None)

    def cancel_all(self, conversation_id: str) -> None:
        """Cancel every pending request for a conversation (turn end/cancel),
        so an awaiting tool unblocks instead of hanging forever."""
        for key in [k for k in self._pending if k[0] == conversation_id]:
            future = self._pending.pop(key, None)
            if future is not None and not future.done():
                future.cancel()


# One instance per server process — mirrors `streaming.registry`.
selection_gateway = SelectionGateway()
