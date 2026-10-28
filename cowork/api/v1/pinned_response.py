"""Streaming responses that own pinned file descriptors."""
from __future__ import annotations

from contextlib import ExitStack

from fastapi.responses import StreamingResponse


class PinnedStreamingResponse(StreamingResponse):
    """Always release pinned descriptors, including before iteration starts."""

    def __init__(self, *args, resources: ExitStack, **kwargs):
        self._resources = resources
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # StreamingResponse does not run a BackgroundTask when the ASGI
            # send callable raises before body iteration. This outer finally
            # covers that disconnect path; ExitStack.close is idempotent with
            # the generator's normal cleanup.
            self._resources.close()
