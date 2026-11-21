"""Uploads are copied from their spools to storage in chunks, so the server's
memory stays flat whatever a file's size: the files upload (save_upload, and
store_file under it) and the project file upload (_write_uploads).

A large upload is a real file on disk, as Starlette's spool is once it passes
1 MiB, and tracemalloc counts every allocation the copy makes on any thread.
"""
from __future__ import annotations

import hashlib
import io
import os
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile
from sqlmodel import Session, select
from starlette.datastructures import Headers

from cowork.api.v1.endpoints import project_files
from cowork.db.scoped import LOCAL_SCOPE
from cowork.models.file import File
from cowork.services import content_safety as safety
from cowork.services import files
from cowork.services.files import UPLOAD_COPY_CHUNK_BYTES, save_upload
from test_files_tenancy import ORG_A, _scope, engine, units_read_engine  # noqa: F401  (engine and units_read_engine are fixtures)

#: Large enough that a copy holding the whole file peaks far above one that
#: holds a chunk at a time.
UPLOAD_BYTES = 64 * 1024 * 1024
#: A quarter of the upload: room for a chunk or two and the request's own
#: bookkeeping, while a copy that held the whole file goes past it.
PEAK_LIMIT_BYTES = 16 * 1024 * 1024


def _large_file(*, path: Path) -> str:
    """Write UPLOAD_BYTES to ``path`` one chunk at a time, each chunk marked
    with its index so a copy out of order differs, and return their SHA-256."""
    digest = hashlib.sha256()
    filler = os.urandom(UPLOAD_COPY_CHUNK_BYTES - 8)
    with path.open("wb") as out:
        for index in range(UPLOAD_BYTES // UPLOAD_COPY_CHUNK_BYTES):
            chunk = index.to_bytes(8, "big") + filler
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _sha256_of(*, path: Path) -> str:
    """The SHA-256 of the file at ``path``, read one chunk at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as stored:
        while chunk := stored.read(UPLOAD_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class _BreaksAfterOneChunk(io.BytesIO):
    """A source that gives its first chunk, then fails as a spool whose disk
    errs halfway through a copy."""

    def read(self, size: int | None = -1) -> bytes:
        if self.tell() >= UPLOAD_COPY_CHUNK_BYTES:
            raise OSError("the spool could not be read")
        return super().read(size)


def _serve_project(*, monkeypatch, base: Path) -> None:
    """Serve ``base`` to _write_uploads as the scope's only project, named
    "project", pinned as the real inventory pins it."""

    @contextmanager
    def opened_inventory(_scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    monkeypatch.setattr(project_files, "_opened_project_directory_inventory", opened_inventory)


def _write_project_uploads(*, upload: UploadFile):
    """Write ``upload`` to the project _serve_project serves, as the project
    upload route does on the file store's threads."""
    return project_files._write_uploads(
        SimpleNamespace(scope=LOCAL_SCOPE), project_name="project", uploads=[upload]
    )


async def test_a_large_file_upload_is_copied_in_chunks(units_read_engine, tmp_path):
    """save_upload copies the spool to the files store a chunk at a time, so
    its peak memory is a small part of the file. A save that read the whole
    upload first would peak at its 64 MiB."""
    source_path = tmp_path / "large.bin"
    expected = _large_file(path=source_path)

    with source_path.open("rb") as spool:
        upload = UploadFile(file=spool, filename="large.bin")
        tracemalloc.start()
        try:
            row = await save_upload(upload, purpose="assistants", scope=_scope(ORG_A))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    # At least one chunk, so the trace saw the copy on its worker thread.
    assert UPLOAD_COPY_CHUNK_BYTES <= peak < PEAK_LIMIT_BYTES
    assert row.size == UPLOAD_BYTES
    stored = Path(row.path)
    assert tmp_path.resolve() in stored.resolve().parents
    assert stored.stat().st_size == UPLOAD_BYTES
    assert _sha256_of(path=stored) == expected


async def test_a_file_upload_that_fails_halfway_leaves_no_folder_or_row(units_read_engine, tmp_path):
    """The source breaks after its first chunk is written: the upload's
    folder goes with it, and no row names it."""
    source = _BreaksAfterOneChunk(b"x" * (2 * UPLOAD_COPY_CHUNK_BYTES + 1))
    scope = _scope(ORG_A)

    with pytest.raises(OSError, match="the spool could not be read"):
        await save_upload(UploadFile(file=source, filename="big.bin"), purpose="assistants", scope=scope)

    assert source.tell() == UPLOAD_COPY_CHUNK_BYTES, "the copy broke after its first chunk"
    assert list(files._files_root(scope).iterdir()) == []
    with Session(units_read_engine) as session:
        assert session.exec(select(File)).all() == []


async def test_a_checked_upload_stores_exactly_the_bytes_that_were_checked(units_read_engine, monkeypatch):
    """With content safety required, the store gets the bytes the check
    approved, never the source again, which changes while it is checked."""
    monkeypatch.setenv("COWORK_CONTENT_SAFETY_REQUIRED", "true")
    source = io.BytesIO(b"original")
    checked: list[str] = []

    async def approve(content, **_kwargs):
        checked.append(content)
        source.seek(0)
        source.write(b"changed!")

    monkeypatch.setattr(safety, "check_content", approve)
    upload = UploadFile(file=source, filename="note.txt", headers=Headers({"content-type": "text/plain"}))

    row = await save_upload(upload, purpose="assistants", scope=_scope(ORG_A))

    assert checked == ["original"]
    assert row.size == len(b"original")
    assert Path(row.path).read_bytes() == b"original"


async def test_a_checked_upload_past_the_check_limit_is_refused_before_storage(engine, monkeypatch):
    """With content safety required, store_file reads one byte past the
    check's limit and no further, and the check refuses it with 413 before
    anything is written."""
    monkeypatch.setenv("COWORK_CONTENT_SAFETY_REQUIRED", "true")
    check = AsyncMock()
    monkeypatch.setattr(safety, "check_content", check)
    source = io.BytesIO(b"x" * (3 * safety.MAX_FILE_BYTES))
    scope = _scope(ORG_A)

    with pytest.raises(HTTPException) as refused:
        await files.store_file(
            scope=scope, filename="big.txt", content_type="text/plain", source=source, purpose="assistants"
        )

    assert refused.value.status_code == 413
    assert source.tell() == safety.MAX_FILE_BYTES + 1
    check.assert_not_called()
    assert not files._files_root(scope).exists()


def test_a_large_project_upload_is_copied_in_chunks(tmp_path, monkeypatch):
    """_write_uploads copies the spool into the project a chunk at a time, so
    its peak memory is a small part of the file. A write that read the whole
    upload first would peak at its 64 MiB."""
    base = tmp_path / "project"
    base.mkdir()
    _serve_project(monkeypatch=monkeypatch, base=base)
    source_path = tmp_path / "large.bin"
    expected = _large_file(path=source_path)

    with source_path.open("rb") as spool:
        upload = UploadFile(file=spool, filename="large.bin")
        tracemalloc.start()
        try:
            response = _write_project_uploads(upload=upload)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    # At least one chunk, so the trace saw the copy.
    assert UPLOAD_COPY_CHUNK_BYTES <= peak < PEAK_LIMIT_BYTES
    assert response.model_dump(exclude_unset=True) == {
        "results": [{"name": "large.bin", "ok": True, "size": UPLOAD_BYTES}]
    }
    assert (base / "large.bin").stat().st_size == UPLOAD_BYTES
    assert _sha256_of(path=base / "large.bin") == expected


def test_a_project_upload_that_fails_halfway_keeps_the_old_file_and_no_temporary(tmp_path, monkeypatch):
    """The source breaks after its first chunk is written to the temporary
    entry: the file it would replace keeps its bytes, and the temporary entry
    is gone."""
    base = tmp_path / "project"
    base.mkdir()
    (base / "big.bin").write_bytes(b"original")
    _serve_project(monkeypatch=monkeypatch, base=base)
    source = _BreaksAfterOneChunk(b"x" * (2 * UPLOAD_COPY_CHUNK_BYTES + 1))

    response = _write_project_uploads(upload=UploadFile(file=source, filename="big.bin"))

    assert source.tell() == UPLOAD_COPY_CHUNK_BYTES, "the copy broke after its first chunk"
    assert response.model_dump(exclude_unset=True) == {
        "results": [{"name": "big.bin", "ok": False, "error": "File write failed"}]
    }
    assert list(base.iterdir()) == [base / "big.bin"]
    assert (base / "big.bin").read_bytes() == b"original"
