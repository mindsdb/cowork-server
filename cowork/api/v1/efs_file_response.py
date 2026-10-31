"""A ``FileResponse`` that reads its file on the shared file store's threads.

Starlette's ``FileResponse`` stats and reads its file through
``anyio.to_thread``, on anyio's default threads. Every authenticated request's
sync dependencies and every sync route share those threads, so a file server
that stops answering would hold them until nothing else could run. This
response opens, stats and reads through ``run_efs`` (cowork/db/units.py)
instead: on the file store's own slots, within the caller's share. Range
requests are answered the way Starlette answers them.

The open and its ``fstat`` run in one call before any header goes out, so a
busy file store refuses the download there with ``FilesystemBusy``, which the
app answers with 503. Once the headers are out a refusal could only cut the
body short of its ``Content-Length``, so each read waits for a slot as long as
it takes, and the close runs even when the response is cancelled
(``run_efs_cleanup``), including a cancel that lands while the open runs.
"""
from __future__ import annotations

import asyncio
import os
import stat
from secrets import token_hex
from typing import IO, Any, NamedTuple

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import (
    FileResponse,
    MalformedRangeHeader,
    PlainTextResponse,
    RangeNotSatisfiable,
)
from starlette.types import Receive, Scope, Send

from cowork.common.paths import O_NONBLOCK, open_fd
from cowork.db.scoped import TenantScope
from cowork.db.units import run_efs, run_efs_cleanup


class _OpenedFile(NamedTuple):
    file: IO[bytes]
    stat: os.stat_result


def _open_regular_file(path: str | os.PathLike[str]) -> _OpenedFile:
    """Open ``path`` for reading and ``fstat`` the descriptor, refusing anything
    but a regular file. Non-blocking, so a FIFO at the path cannot hold the
    thread; a regular file ignores the flag."""
    fd = open_fd(path, os.O_RDONLY | O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"File at path {path} is not a file.")
        return _OpenedFile(file=os.fdopen(fd, "rb"), stat=opened)
    except BaseException:
        os.close(fd)
        raise


class EfsFileResponse(FileResponse):
    def __init__(self, path: str | os.PathLike[str], *, scope: TenantScope, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self._tenant_scope = scope

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        send_header_only = scope["type"] == "http" and scope["method"].upper() == "HEAD"
        try:
            opened = await self._open()
        except FileNotFoundError:
            raise RuntimeError(f"File at path {self.path} does not exist.")
        file = opened.file
        try:
            stat_result = self.stat_result
            if stat_result is None:
                stat_result = opened.stat
                self.set_stat_headers(stat_result)

            headers = Headers(scope=scope)
            http_range = headers.get("range")
            http_if_range = headers.get("if-range")
            if http_range is None or (http_if_range is not None and not self._should_use_range(http_if_range)):
                await self._send_whole(send, file, send_header_only=send_header_only)
            else:
                try:
                    ranges = self._parse_range_header(http_range, stat_result.st_size)
                except MalformedRangeHeader as exc:
                    return await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
                except RangeNotSatisfiable as exc:
                    response = PlainTextResponse(status_code=416, headers={"Content-Range": f"bytes */{exc.max_size}"})
                    return await response(scope, receive, send)
                await self._send_ranges(
                    send, file, ranges=ranges, file_size=stat_result.st_size, send_header_only=send_header_only
                )
        finally:
            await run_efs_cleanup(file.close, scope=self._tenant_scope)

        if self.background is not None:
            await self.background()

    async def _open(self) -> _OpenedFile:
        """Open the file on the file store's threads. A cancel that lands while
        the open runs waits for it (``run_efs``), so a file it opened is closed
        on the same threads before the cancel goes on."""
        opened: _OpenedFile | None = None

        def open_file() -> _OpenedFile:
            nonlocal opened
            opened = _open_regular_file(self.path)
            return opened

        try:
            return await run_efs(open_file, scope=self._tenant_scope)
        except asyncio.CancelledError:
            if opened is not None:
                await run_efs_cleanup(opened.file.close, scope=self._tenant_scope)
            raise

    async def _read(self, file: IO[bytes], *, start: int, size: int) -> bytes:
        def read_at() -> bytes:
            file.seek(start)
            return file.read(size)

        return await run_efs(read_at, scope=self._tenant_scope, deadline=False)

    async def _send_whole(self, send: Send, file: IO[bytes], *, send_header_only: bool) -> None:
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        offset = 0
        more_body = True
        while more_body:
            chunk = await self._read(file, start=offset, size=self.chunk_size)
            offset += len(chunk)
            more_body = len(chunk) == self.chunk_size
            await send({"type": "http.response.body", "body": chunk, "more_body": more_body})

    async def _send_ranges(
        self,
        send: Send,
        file: IO[bytes],
        *,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        headers = MutableHeaders(raw=list(self.raw_headers))
        if len(ranges) == 1:
            start, end = ranges[0]
            headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
            headers["content-length"] = str(end - start)
            boundary = None
        else:
            # In firefox and chrome, they use boundary with 95-96 bits entropy (that's roughly 13 bytes).
            boundary = token_hex(13)
            content_length, header_generator = self.generate_multipart(
                ranges, boundary, file_size, self.headers["content-type"]
            )
            headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
            headers["content-length"] = str(content_length)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        for start, end in ranges:
            if boundary is not None:
                await send({"type": "http.response.body", "body": header_generator(start, end), "more_body": True})
            while start < end:
                chunk = await self._read(file, start=start, size=min(self.chunk_size, end - start))
                if not chunk:
                    break
                start += len(chunk)
                more_body = boundary is not None or start < end
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
            if boundary is not None:
                await send({"type": "http.response.body", "body": b"\r\n", "more_body": True})
        if boundary is not None:
            await send({"type": "http.response.body", "body": f"--{boundary}--".encode("latin-1"), "more_body": False})
        elif start < end:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
