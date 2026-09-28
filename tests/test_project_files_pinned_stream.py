"""Project file streams send exactly the pinned descriptor's bytes (ENG-2950)."""
from __future__ import annotations

import os

import pytest
from fastapi.responses import StreamingResponse

from cowork.api.v1.endpoints import project_files as project_files_ep

CSV_CRLF = b"id,name\r\n1,alpha\r\n2,beta\r\n"
BINARY_WITH_EOF_MARK = b"PK\x03\x04\x1a\r\n\x00\x1a\n\xff"


def _project(tmp_path):
    base = tmp_path.resolve() / "project"
    base.mkdir()
    return base


def _stream(target, base):
    return project_files_ep._pinned_stream(
        target,
        base,
        media_type="text/csv",
        headers={"Cache-Control": "private, max-age=300"},
    )


async def _body(response: StreamingResponse) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


@pytest.mark.parametrize("payload", (CSV_CRLF, BINARY_WITH_EOF_MARK))
async def test_body_is_the_file_bytes(tmp_path, payload):
    base = _project(tmp_path)
    target = base / "data.bin"
    target.write_bytes(payload)

    response = _stream(target, base)

    assert await _body(response) == payload
    assert response.headers["cache-control"] == "private, max-age=300"


async def test_declares_no_length_when_the_file_grows(tmp_path):
    base = _project(tmp_path)
    target = base / "data.csv"
    target.write_bytes(b"a,b\n1,2\n")
    response = _stream(target, base)

    with target.open("ab") as handle:
        handle.write(b"3,4\n")

    assert "content-length" not in response.headers
    assert await _body(response) == b"a,b\n1,2\n3,4\n"


async def test_declares_no_length_when_the_file_is_rewritten_shorter(tmp_path):
    base = _project(tmp_path)
    target = base / "data.csv"
    target.write_bytes(b"a,b\n1,2\n3,4\n")
    response = _stream(target, base)

    with target.open("r+b") as handle:
        handle.truncate(0)
        handle.write(b"a,b\n")

    assert "content-length" not in response.headers
    assert await _body(response) == b"a,b\n"


async def test_keeps_the_old_bytes_when_the_file_is_replaced(tmp_path):
    base = _project(tmp_path)
    target = base / "data.csv"
    target.write_bytes(b"a,b\n1,2\n")
    response = _stream(target, base)

    replacement = base / ".data.csv.tmp"
    replacement.write_bytes(b"new\n")
    os.replace(replacement, target)

    assert "content-length" not in response.headers
    assert await _body(response) == b"a,b\n1,2\n"
