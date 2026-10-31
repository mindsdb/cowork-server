"""Streaming responses that own pinned file descriptors."""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack
from functools import partial

from fastapi.responses import StreamingResponse

from cowork.db.scoped import TenantScope
from cowork.db.units import run_efs, run_efs_cleanup


class PinnedStreamingResponse(StreamingResponse):
    """Stream the chunks a pinned descriptor yields, and always release the
    descriptors, including before iteration starts.

    Each chunk is read on the shared file store's threads (``run_efs``), never
    on anyio's default threads, where Starlette would iterate a sync iterator.
    The headers are already out when the first read runs, so a read waits for
    a slot as long as it takes rather than cut the body short.
    """

    def __init__(self, chunks: Iterator[bytes], *, resources: ExitStack, scope: TenantScope, **kwargs):
        self._resources = resources
        self._tenant_scope = scope
        super().__init__(self._read(chunks), **kwargs)

    async def _read(self, chunks: Iterator[bytes]) -> AsyncIterator[bytes]:
        while (chunk := await run_efs(partial(next, chunks, None), scope=self._tenant_scope, deadline=False)) is not None:
            yield chunk

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # The one release, after the last frame or whatever ended the
            # response early: a disconnect cancels the stream's task group,
            # and a send that raises skips the body. It runs on the file store's
            # threads even when the response is cancelled (run_efs_cleanup).
            await run_efs_cleanup(self._resources.close, scope=self._tenant_scope)
