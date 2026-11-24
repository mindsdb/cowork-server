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
            from fastapi import HTTPException
            from cowork.services.content_safety import ContentSafetySettings, MAX_FILE_BYTES, check_file_bytes
            if ContentSafetySettings().required:
                if self.status_code == 206:
                    raise HTTPException(503, 'Partial file safety checks are not available yet.')
                parts, size = [], 0
                async for chunk in self.body_iterator:
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(413, 'This file is too large for a safety check.')
                    parts.append(chunk)
                data = b''.join(parts)
                await check_file_bytes(data, content_type=self.media_type or '', purpose='output', scope=self._tenant_scope)
                async def checked_body():
                    yield data
                self.body_iterator = checked_body()
                self.headers['content-length'] = str(len(data))
                self.headers['cache-control'] = 'no-store'
                self.headers['x-content-type-options'] = 'nosniff'
            await super().__call__(scope, receive, send)
        finally:
            # The one release, after the last frame or whatever ended the
            # response early: a disconnect cancels the stream's task group,
            # and a send that raises skips the body. It runs on the file store's
            # threads even when the response is cancelled (run_efs_cleanup).
            await run_efs_cleanup(self._resources.close, scope=self._tenant_scope)
