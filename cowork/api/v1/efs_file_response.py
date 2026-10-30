"""A ``FileResponse`` that reads its file on the shared file store's threads.

Starlette's ``FileResponse`` stats and reads its file through
``anyio.to_thread``, on anyio's default threads. Every authenticated request's
sync dependencies and every sync route share those threads, so a file server
that stops answering would hold them until nothing else could run. This
response does the same stat and reads through ``run_efs`` (cowork/db/units.py)
instead: on the file store's own slots, within the caller's share, and refused
with ``FilesystemBusy`` when no slot frees within POOL_TIMEOUT. Range requests
are answered the way Starlette answers them.
"""
from __future__ import annotations

import os
import stat
from functools import partial
from secrets import token_hex
from typing import IO, Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import (
    FileResponse,
    MalformedRangeHeader,
    PlainTextResponse,
    RangeNotSatisfiable,
)
from starlette.types import Receive, Scope, Send

from cowork.db.scoped import TenantScope
from cowork.db.units import run_efs


class EfsFileResponse(FileResponse):
    def __init__(self, path: str | os.PathLike[str], *, scope: TenantScope, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self._tenant_scope = scope

    async def _efs(self, fn, *args, deadline: bool = True):
        return await run_efs(partial(fn, *args), scope=self._tenant_scope, deadline=deadline)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        send_header_only = scope["type"] == "http" and scope["method"].upper() == "HEAD"
        if self.stat_result is None:
            try:
                stat_result = await self._efs(os.stat, self.path)
            except FileNotFoundError:
                raise RuntimeError(f"File at path {self.path} does not exist.")
            if not stat.S_ISREG(stat_result.st_mode):
                raise RuntimeError(f"File at path {self.path} is not a file.")
            self.set_stat_headers(stat_result)
        else:
            stat_result = self.stat_result

        headers = Headers(scope=scope)
        http_range = headers.get("range")
        http_if_range = headers.get("if-range")
        if http_range is None or (http_if_range is not None and not self._should_use_range(http_if_range)):
            await self._send_whole(send, send_header_only)
        else:
            try:
                ranges = self._parse_range_header(http_range, stat_result.st_size)
            except MalformedRangeHeader as exc:
                return await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
            except RangeNotSatisfiable as exc:
                response = PlainTextResponse(status_code=416, headers={"Content-Range": f"bytes */{exc.max_size}"})
                return await response(scope, receive, send)
            await self._send_ranges(send, ranges, stat_result.st_size, send_header_only)

        if self.background is not None:
            await self.background()

    async def _read(self, file: IO[bytes], start: int, size: int) -> bytes:
        def read_at() -> bytes:
            file.seek(start)
            return file.read(size)

        return await self._efs(read_at)

    async def _send_whole(self, send: Send, send_header_only: bool) -> None:
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        file = await self._efs(open, self.path, "rb")
        try:
            offset = 0
            more_body = True
            while more_body:
                chunk = await self._read(file, offset, self.chunk_size)
                offset += len(chunk)
                more_body = len(chunk) == self.chunk_size
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
        finally:
            await self._efs(file.close, deadline=False)

    async def _send_ranges(
        self, send: Send, ranges: list[tuple[int, int]], file_size: int, send_header_only: bool
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
        file = await self._efs(open, self.path, "rb")
        try:
            for start, end in ranges:
                if boundary is not None:
                    await send({"type": "http.response.body", "body": header_generator(start, end), "more_body": True})
                while start < end:
                    chunk = await self._read(file, start, min(self.chunk_size, end - start))
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
        finally:
            await self._efs(file.close, deadline=False)
