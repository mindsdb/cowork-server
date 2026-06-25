from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.file import File
from cowork.schemas.files import FileResponse


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


def validate_upload(filename: str, content_type: str | None, size: int) -> None:
    """Guardrail an upload before it touches disk. Raises
    FileValidationError with a user-facing message if the file is over the
    configured size cap or is a type the agent can't use. Both the
    OpenAI-style /files endpoint and the attachments compat endpoint route
    through here (via create_file), so the rules are enforced once."""
    name = (filename or "").strip()
    label = name or "this file"

    max_bytes = get_app_settings().file.max_upload_bytes
    if size > max_bytes:
        raise FileValidationError(
            f"{label} is {_humanize_bytes(size)} — max is {_humanize_bytes(max_bytes)}."
        )

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    ext = Path(name).suffix.lower()
    if mime in ALLOWED_MIME_TYPES or ext in ALLOWED_EXTENSIONS:
        return

    shown = ext or (mime or "unknown")
    raise FileValidationError(
        f"{label} ({shown}) isn't a supported type. Allowed: {_ALLOWED_SUMMARY}."
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
        contents = await upload.read()
        filename = upload.filename or "upload"

        # Guardrail before anything touches disk. (Reading fully into memory
        # first is a known limitation — the streaming/OOM fix is a separate
        # slice; here we only enforce the cap + allow-list on the bytes we
        # already have.)
        validate_upload(filename, upload.content_type, len(contents))

        file = File(
            filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            size=len(contents),
            purpose=purpose,
            path="",
        )

        file_dir = self._root_dir() / str(file.id)
        file_dir.mkdir(parents=True)
        dest = file_dir / filename
        dest.write_bytes(contents)
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
