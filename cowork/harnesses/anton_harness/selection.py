"""Server-side wiring for anton's mid-turn file/folder selection.

anton's ``select_path`` tool depends on the abstract ``SelectionElicitor``
strategy. Here we supply the cowork-server implementation: it emits a
``SelectionRequestEvent`` into the turn's event stream (the stream formatter
turns it into a ``response.selection.requested`` SSE frame) and then awaits the
user's pick on a :class:`~cowork.streaming.selection_gateway.SelectionGateway`
future, resolved out-of-band by ``POST /responses/selection``.

The event is emitted *before* the tool blocks, so the frontend renders the
picker while the turn quietly waits — no extra user message, no turn boundary.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

from anton.core.interaction.selection import SelectionRequest

from cowork.streaming.selection_gateway import SelectionGateway

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionRequestEvent:
    """A request, flowing through the turn's event stream, for the UI to render
    an inline picker. ``options`` are plain dicts (value/label/kind/detail) so
    the formatter can serialise them straight to JSON.

    ``mode`` is ``"pick"`` (disambiguate ``options``) or ``"browse"`` (the user
    navigates a file tree from ``root``)."""

    request_id: str
    prompt: str
    kind: str
    mode: str = "pick"
    root: str = ""
    options: list[dict] = field(default_factory=list)


class StreamingSelectionElicitor:
    """anton ``SelectionElicitor`` that streams the request and awaits a reply.

    ``emit`` enqueues an event into the live turn stream (the harness drains
    that queue concurrently, so the request reaches the client before this
    coroutine blocks on the gateway future).
    """

    def __init__(
        self,
        *,
        emit: Callable[[object], None],
        gateway: SelectionGateway,
        conversation_id: str,
    ) -> None:
        self._emit = emit
        self._gateway = gateway
        self._conversation_id = conversation_id

    async def elicit(self, request: SelectionRequest) -> str | None:
        request_id = uuid.uuid4().hex[:12]
        future = self._gateway.open(self._conversation_id, request_id)
        self._emit(
            SelectionRequestEvent(
                request_id=request_id,
                prompt=request.prompt,
                kind=request.kind,
                mode=request.mode,
                root=request.root,
                options=[
                    {"value": o.value, "label": o.label, "kind": o.kind, "detail": o.detail}
                    for o in request.options
                ],
            )
        )
        try:
            return await future
        except asyncio.CancelledError:
            # Turn cancelled (Stop button / disconnect-driven teardown) while
            # waiting — treat as "no choice" so the tool returns cleanly.
            return None
        finally:
            self._gateway.close(self._conversation_id, request_id)
