from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.file import File
from cowork.schemas.files import FileResponse

# How much we pull off the upload stream per read. Bytes land on disk as we
# go, so peak memory is roughly one chunk — not the whole file. (Starlette's
# UploadFile is itself a SpooledTemporaryFile that spills to disk past ~1 MiB,
# so even its internal buffer isn't unbounded RAM.)
_STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def attachment_purpose(project_name: str, session_id: str) -> str:
    """Canonical purpose tag for conversation attachments. The composer
    uploads against a client-allocated conversation id, and the rail's
    Task Uploads list queries by the live conversation id — both must
    derive the tag from here or uploads strand (ENG-264)."""
    return f"attachment:{project_name}:{session_id}"


class FileValidationError(ValueError):
    """Raised when an upload fails a guardrail (too large, or a type the
    agent can't use). Carries a human-readable, user-facing message; the
    endpoints turn it into a 400 with that message as the detail. Kept a
    ValueError subclass so the existing `except ValueError` retrieval
    paths keep working."""


# Allow-list of attachable types, by MIME and by extension. We accept a
# file if EITHER its declared MIME or its extension is on the list — a
# browser drag/paste sometimes sends a bare `application/octet-stream`
# for an otherwise-fine `.csv`, and conversely some types (e.g. images
# from a screenshot paste) arrive with no filename at all. The set is the
# stuff the agent can actually read: documents, plain text/data, and
# common images. Anything else (executables, archives, disk images, …)
# is rejected with a clear message rather than silently stored.
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    # Images
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml",
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Text & data
    "text/plain", "text/markdown", "text/csv", "text/tab-separated-values",
    "application/json", "application/xml", "text/xml", "application/yaml", "text/yaml",
})

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Text & data
    ".txt", ".md", ".markdown", ".csv", ".tsv",
    ".json", ".xml", ".yaml", ".yml", ".log",
})

# Human-readable list of what we DO take, shown in the rejection message
# so the user isn't left guessing. Grouped, not the raw MIME soup.
_ALLOWED_SUMMARY = "images, PDFs, Office docs, and text/data files (txt, md, csv, json, …)"


def _humanize_bytes(num: int) -> str:
    """Compact, human-facing size like "41 MB" / "512 KB" — for error
    copy, so we round to whole units and drop trailing ".0"."""
    step = 1024.0
    value = float(num)
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < step or unit == "GB":
            if unit == "bytes":
                return f"{int(value)} {unit}"
            rounded = round(value, 1)
            text = f"{rounded:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
        value /= step
    return f"{num} bytes"  # unreachable; keeps type checkers happy


def _label_for(filename: str) -> str:
    name = (filename or "").strip()
    return name or "this file"


def validate_upload_type(filename: str, content_type: str | None) -> None:
    """Type half of the guardrail: reject anything off the MIME/extension
    allow-list. Split out so it can run BEFORE we read a single byte off the
    wire — a disallowed type never gets streamed to disk. Accepts if EITHER
    the declared MIME or the extension is on the list."""
    name = (filename or "").strip()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    ext = Path(name).suffix.lower()
    if mime in ALLOWED_MIME_TYPES or ext in ALLOWED_EXTENSIONS:
        return

    shown = ext or (mime or "unknown")
    raise FileValidationError(
        f"{_label_for(filename)} ({shown}) isn't a supported type. "
        f"Allowed: {_ALLOWED_SUMMARY}."
    )


def _raise_oversize(
    filename: str, size: int, max_bytes: int, *, at_least: bool = False
) -> None:
    # When streaming we stop reading the instant we cross the cap, so `size`
    # is a floor, not the true length — say "over" rather than quote a number
    # we deliberately never finished measuring.
    measure = f"over {_humanize_bytes(max_bytes)}" if at_least else _humanize_bytes(size)
    raise FileValidationError(
        f"{_label_for(filename)} is {measure} — max is {_humanize_bytes(max_bytes)}."
    )


def validate_upload(filename: str, content_type: str | None, size: int) -> None:
    """Guardrail an upload whose size is already known. Raises
    FileValidationError with a user-facing message if the file is over the
    configured size cap or is a type the agent can't use. The streaming
    ingest path enforces the same two rules incrementally (type up front,
    size mid-stream); this stays for callers that have the bytes in hand and
    for direct unit coverage of the rules."""
    max_bytes = get_app_settings().file.max_upload_bytes
    if size > max_bytes:
        _raise_oversize(filename, size, max_bytes)
    validate_upload_type(filename, content_type)


class UploadTooLarge(FileValidationError):
    """Cap breach detected up front from the declared Content-Length, before
    the body is read. Endpoints map this to 413 Payload Too Large (the size
    is known with certainty from the header); a breach only discovered
    mid-stream stays a 400. Subclasses FileValidationError so existing
    handlers still catch it."""


async def stream_upload_to_path(upload: UploadFile, dest: Path) -> int:
    """Copy ``upload`` to ``dest`` one chunk at a time, aborting the moment
    the running total crosses the configured cap. Never buffers the whole
    file — peak memory is ~one chunk. Returns the bytes written. On a cap
    breach (or any error mid-write) the partial file is removed and
    FileValidationError is raised. The caller owns ``dest``'s parent dir."""
    max_bytes = get_app_settings().file.max_upload_bytes
    filename = upload.filename or dest.name
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    # Stop reading immediately — we don't drain the rest of
                    # the (possibly huge) stream just to measure it.
                    _raise_oversize(filename, written, max_bytes, at_least=True)
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return written


def reject_if_content_length_over_cap(content_length: str | int | None) -> None:
    """Cheap early-out: if the request declares a body larger than the cap
    (plus a small allowance for the multipart envelope), reject before
    consuming the stream at all. The declared length is an UPPER bound on the
    file bytes — a multipart body is `file + boundaries/headers` — so we only
    trip when it's over the cap *with* headroom, leaving the authoritative
    decision to the mid-stream counter. A missing/garbage header is ignored
    (chunked-transfer uploads have none); the stream check still backstops."""
    if content_length is None:
        return
    try:
        declared = int(content_length)
    except (TypeError, ValueError):
        return
    max_bytes = get_app_settings().file.max_upload_bytes
    # 1 MiB of slack swallows multipart boundaries/part headers for any
    # realistic field count, so we never false-reject a file that's actually
    # under the cap.
    if declared > max_bytes + (1024 * 1024):
        raise UploadTooLarge(
            f"Upload is {_humanize_bytes(declared)} — max is {_humanize_bytes(max_bytes)}."
        )


class FileService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        return Path(get_app_settings().file.root_dir)

    def _to_response(self, file: File) -> FileResponse:
        return FileResponse(
            id=str(file.id),
            bytes=file.size,
            created_at=int(file.created_at.timestamp()) if file.created_at else 0,
            filename=file.filename,
            purpose=file.purpose,
        )

    def list_files(self, purpose: str | None = None) -> list[FileResponse]:
        stmt = select(File)
        if purpose is not None:
            stmt = stmt.where(File.purpose == purpose)
        return [self._to_response(f) for f in self.session.exec(stmt).all()]

    def list_file_rows(self, purpose: str) -> list[File]:
        """Raw File rows for callers that need fields the OpenAI-style
        FileResponse drops (content_type, timestamps) — e.g. the
        attachments compat endpoints."""
        return list(self.session.exec(select(File).where(File.purpose == purpose)).all())

    def get_file_row(self, file_id: UUID) -> File:
        return self._get_file_model(file_id)

    def relink_purpose(self, old_purpose: str, new_purpose: str) -> int:
        """Repoint every file stored under `old_purpose`. Used when a
        conversation ends up with a different id than the one the client
        uploaded attachments against. Returns the number relinked."""
        files = self.session.exec(select(File).where(File.purpose == old_purpose)).all()
        for file in files:
            file.purpose = new_purpose
            self.session.add(file)
        if files:
            self.session.commit()
        return len(files)

    def get_file(self, file_id: UUID) -> FileResponse:
        file = self.session.get(File, file_id)
        if file is None:
            raise ValueError("File not found")
        return self._to_response(file)

    async def create_file(self, upload: UploadFile, purpose: str) -> FileResponse:
        filename = upload.filename or "upload"

        # Type check first — a disallowed type is rejected before we pull a
        # single byte off the wire (the bytes never touch disk or RAM).
        validate_upload_type(filename, upload.content_type)

        file = File(
            filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            size=0,
            purpose=purpose,
            path="",
        )

        file_dir = self._root_dir() / str(file.id)
        file_dir.mkdir(parents=True)
        dest = file_dir / filename
        # Stream chunked to disk, enforcing the size cap mid-stream. On a cap
        # breach (or any failure) the partial file + its dir are removed so we
        # don't leave a truncated artifact behind, then a 4xx is raised.
        try:
            written = await stream_upload_to_path(upload, dest)
        except BaseException:
            shutil.rmtree(file_dir, ignore_errors=True)
            raise

        file.size = written
        file.path = str(dest)
        self.session.add(file)
        self.session.commit()
        self.session.refresh(file)
        return self._to_response(file)

    def create_file_from_bytes(self, *, filename: str, content_type: str, data: bytes, purpose: str) -> File:
        """Server-side ingestion (e.g. channel media); returns the model.
        The filename comes from an external platform, so keep only its basename."""
        safe_name = Path(filename).name.strip() or "file"
        file = File(
            filename=safe_name,
            content_type=content_type or "application/octet-stream",
            size=len(data),
            purpose=purpose,
            path="",
        )
        file_dir = self._root_dir() / str(file.id)
        file_dir.mkdir(parents=True)
        dest = file_dir / safe_name
        dest.write_bytes(data)
        file.path = str(dest)
        self.session.add(file)
        self.session.commit()
        self.session.refresh(file)
        return file

    def _get_file_model(self, file_id: UUID) -> File:
        file = self.session.get(File, file_id)
        if file is None:
            raise ValueError("File not found")
        return file

    def delete_file(self, file_id: UUID) -> bool:
        file = self.session.get(File, file_id)
        if file is None:
            return False
        file_dir = Path(file.path).parent
        self.session.delete(file)
        self.session.commit()
        if file_dir.exists():
            shutil.rmtree(file_dir)
        return True

    def get_file_content(self, file_id: UUID) -> tuple[str, str, Path]:
        file = self._get_file_model(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return file.content_type, file.filename, path
