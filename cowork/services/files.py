from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.file import File
from cowork.schemas.files import FileResponse

logger = logging.getLogger(__name__)


def unlink_file_dirs(dirs: list[Path]) -> None:
    """Best-effort removal of the per-file `<root>/<file.id>/` directories whose
    rows the caller has already committed as deleted. Log-and-continue so a
    locked file can't abort the caller's own deletion — same policy as
    move-to-project. Call this AFTER the DB delete is committed, never before:
    the row is the source of truth, so bytes must outlive an uncommitted delete.
    """
    for d in dirs:
        try:
            if d.exists():
                shutil.rmtree(d)
        except OSError:
            logger.warning("could not remove file dir %s", d, exc_info=True)


def attachment_purpose(session_id: str) -> str:
    """Canonical purpose tag for conversation attachments. The composer
    uploads against a client-allocated conversation id, and the rail's
    Task Uploads list queries by the live conversation id — both must
    derive the tag from here or uploads strand (ENG-264).

    Keyed by the conversation/session id ONLY — never by the project name.
    The name is mutable, so embedding it stranded every attachment on a
    project rename (ENG-338) and let long names overflow the purpose column
    (ENG-333); the id is stable and fixed-width. Old-format tags
    ("attachment:{project}:{session}") are rewritten by migration
    f7d2b9e4a1c6."""
    return f"attachment:{session_id}"


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
        unlink_file_dirs([file_dir])
        return True

    def delete_by_purpose(self, purpose: str) -> list[Path]:
        """Stage deletion of every file row under `purpose`; return the on-disk
        dirs to unlink once the caller commits.

        Cleans up a conversation's attachments when the conversation (or its
        project) is deleted — otherwise the rows + bytes orphan forever
        (ENG-701). Follows the stage-only convention of
        `TaskObjectService.delete_for_conversation`: the caller owns the commit,
        so the attachment-row delete lands in the SAME transaction as the
        caller's own deletes (a crash mid-way can't leave a half-deleted
        conversation), then the caller unlinks the returned dirs via
        `unlink_file_dirs` AFTER committing.
        """
        rows = list(self.session.exec(select(File).where(File.purpose == purpose)).all())
        dirs = [Path(f.path).parent for f in rows if f.path]
        for f in rows:
            self.session.delete(f)
        return dirs

    def get_file_content(self, file_id: UUID) -> tuple[str, str, Path]:
        file = self._get_file_model(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return file.content_type, file.filename, path
