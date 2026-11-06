"""Project file streams send exactly the pinned descriptor's bytes (ENG-2950).

A pinned open never waits on what sits at the path.
"""
from __future__ import annotations

import concurrent.futures
import errno
import os
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cowork.api.v1.endpoints import project_files as project_files_ep
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.models.project import Project

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
        scope=LOCAL_SCOPE,
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


async def test_stops_at_the_authorized_size_when_the_file_grows(tmp_path):
    """Bytes appended after fstat are not sent: a file that keeps growing
    cannot keep the response open (ENG-2950)."""
    base = _project(tmp_path)
    target = base / "data.csv"
    target.write_bytes(b"a,b\n1,2\n")
    response = _stream(target, base)

    with target.open("ab") as handle:
        handle.write(b"3,4\n")

    assert "content-length" not in response.headers
    assert await _body(response) == b"a,b\n1,2\n"


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


#: Seconds a pinned open of a FIFO may take. An open that waits for a writer
#: never returns, so any bound catches it; this one leaves a loaded runner room.
FIFO_BUDGET_SECONDS = 2.0


def _read_in_a_workspace(*, base, rel_path):
    """GET /files/<path>'s unit for `rel_path` in a project at `base`."""
    name = f"fifo-{uuid4().hex[:8]}"
    with unit_session(scope=LOCAL_SCOPE) as session:
        session.add(Project(name=name, path=str(base)))
    try:
        with unit_session(scope=LOCAL_SCOPE) as session:
            return project_files_ep._read_project_file(session, project_name=name, path=rel_path, principal=None)
    finally:
        with unit_session(scope=LOCAL_SCOPE) as session:
            project = session.exec(session.select(Project).where(Project.name == name)).one()
            session.delete(project)


def _preview_in_a_workspace(*, base, rel_path):
    """The preview asset route's work for `rel_path`, under a token minted for
    the page beside it."""
    page = (base / rel_path).parent / "index.html"
    token = project_files_ep._register_preview_mount(page.resolve(), base, LOCAL_SCOPE)
    return project_files_ep._preview_asset(token=token, rel_path=os.path.basename(rel_path), scope=LOCAL_SCOPE)


def _release_a_waiting_reader(*, path) -> None:
    """Open `path` for writing without waiting, which lets a reader stuck in
    its open return, so a regression fails its test instead of keeping a
    thread for the rest of the run. ENXIO: no reader is waiting."""
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
    except OSError as exc:
        if exc.errno != errno.ENXIO:
            raise


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
@pytest.mark.parametrize("reader", [_read_in_a_workspace, _preview_in_a_workspace], ids=["read", "preview"])
def test_a_fifo_in_a_workspace_answers_404_without_waiting_for_a_writer(tmp_path, reader):
    """A pod can make a FIFO where a page expects a file (`mkfifo chart.png`
    beside an index.html that shows it). No writer on the server's node ever
    opens it, so an open that waited would hold its file store slot and its
    thread until the pod restarts. The read and the preview answer 404 at once
    instead."""
    base = _project(tmp_path)
    workspace = base / "conversations" / str(uuid4())
    workspace.mkdir(parents=True)
    (workspace / "index.html").write_text('<img src="chart.png">')
    fifo = workspace / "chart.png"
    os.mkfifo(fifo)
    rel_path = f"conversations/{workspace.name}/chart.png"
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        answer = pool.submit(reader, base=base, rel_path=rel_path)
        try:
            with pytest.raises(HTTPException) as refused:
                answer.result(timeout=FIFO_BUDGET_SECONDS)
        except concurrent.futures.TimeoutError:
            pytest.fail("the open waited for a writer on the FIFO")
    finally:
        _release_a_waiting_reader(path=fifo)
        pool.shutdown(wait=True)

    assert refused.value.status_code == 404
