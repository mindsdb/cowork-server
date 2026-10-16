"""Transport ownership for an explicitly opted-in originating response POST."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

T = TypeVar("T")


class DisconnectWatch:
    """The only ASGI receive consumer after FastAPI has parsed the request body."""

    def __init__(self, receive: Receive) -> None:
        self.disconnected = False
        self.task = asyncio.create_task(self._listen(receive))

    async def _listen(self, receive: Receive) -> None:
        while True:
            if (await receive())["type"] == "http.disconnect":
                self.disconnected = True
                return

    async def prepare(self, operation: Awaitable[T]) -> T:
        """Stop abandoned setup, retaining any admitted handle on its handler."""
        work = asyncio.ensure_future(operation)
        try:
            await asyncio.wait((work, self.task), return_when=asyncio.FIRST_COMPLETED)
            if work.done():
                return work.result()
            # A receive failure is not evidence that the client disconnected.
            self.task.result()
            raise ClientDisconnect()
        finally:
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class DisconnectStreamingResponse(StreamingResponse):
    """Cancel only for observed transport loss, never for iterator exhaustion/shutdown.

    The receive watch begins during preparation and transfers here, so no
    disconnect is lost between routing and streaming. This owns the original
    POST only: the resumable tail endpoint continues using StreamingResponse.
    """

    def __init__(self, content, *, watch: DisconnectWatch, on_disconnect: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(content, **kwargs)
        self.watch = watch
        self.on_disconnect = on_disconnect

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        lost = False

        async def watched_send(message):
            nonlocal lost
            try:
                await send(message)
            except OSError:
                lost = True
                raise

        stream = asyncio.create_task(self.stream_response(watched_send))
        try:
            await asyncio.wait((stream, self.watch.task), return_when=asyncio.FIRST_COMPLETED)
            if stream.done():
                try:
                    stream.result()
                except OSError:
                    if not lost:
                        raise
            else:
                self.watch.task.result()
                lost = self.watch.disconnected
        finally:
            # Cancelling the outer ASGI task (server shutdown) propagates past
            # this finally and never reaches the disconnect callback below.
            stream.cancel()
            await asyncio.gather(stream, return_exceptions=True)
            await self.watch.close()
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
        if lost:
            await self.on_disconnect()
        if self.background is not None:
            await self.background()
