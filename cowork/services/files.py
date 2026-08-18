from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, scoped_storage_root
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
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        """Files root, org-keyed in org mode (same helper as skills/projects).

        Without the org segment every upload landed in one flat
        ``<root>/files/<uuid>/`` directory, a SIBLING of the org directories
        rather than inside one. A worker pod mounts only its own
        ``<env>/<org_id>``, so it could never see an uploaded attachment at
        all, and that is the primary use case this whole feature exists for.
        """
        return scoped_storage_root(Path(get_app_settings().file.root_dir), self.session.scope)

    def _to_response(self, file: File) -> FileResponse:
        return FileResponse(
            id=str(file.id),
            bytes=file.size,
            created_at=int(file.created_at.timestamp()) if file.created_at else 0,
            filename=file.filename,
            purpose=file.purpose,
        )

    def list_files(self, purpose: str | None = None) -> list[FileResponse]:
        stmt = self.session.select(File)
        if purpose is not None:
            stmt = stmt.where(File.purpose == purpose)
        return [self._to_response(f) for f in self.session.exec(stmt).all()]

    def list_file_rows(self, purpose: str) -> list[File]:
        """Raw File rows for callers that need fields the OpenAI-style
        FileResponse drops (content_type, timestamps) — e.g. the
        attachments compat endpoints."""
        return list(self.session.exec(self.session.select(File).where(File.purpose == purpose)).all())

    def get_file_row(self, file_id: UUID) -> File:
        return self._get_file_model(file_id)

    def relink_purpose(self, old_purpose: str, new_purpose: str) -> int:
        """Repoint every file stored under `old_purpose`. Used when a
        conversation ends up with a different id than the one the client
        uploaded attachments against. Returns the number relinked."""
        files = self.session.exec(self.session.select(File).where(File.purpose == old_purpose)).all()
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
        # Untrusted filename: keep the basename only so `../` or an absolute
        # path can't escape the per-file dir. `.name` also leaves bare "."/".." —
        # reject those.
        safe_name = Path(upload.filename or "upload").name.strip()
        if safe_name in ("", ".", ".."):
            safe_name = "upload"

        file = File(
            filename=safe_name,
            content_type=upload.content_type or "application/octet-stream",
            size=len(contents),
            purpose=purpose,
            path="",
        )
        # Stage the row first: the scope check (and stamping) happens in
        # add(), so a missing principal fails BEFORE any bytes hit disk.
        self.session.add(file)
        file_dir = self._root_dir() / str(file.id)
        try:
            file_dir.mkdir(parents=True)
            dest = file_dir / safe_name
            dest.write_bytes(contents)
        except Exception:
            self.session.rollback()
            shutil.rmtree(file_dir, ignore_errors=True)
            raise
        file.path = str(dest)
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
        # Row staged first — scope failure must precede any filesystem write.
        self.session.add(file)
        file_dir = self._root_dir() / str(file.id)
        try:
            file_dir.mkdir(parents=True)
            dest = file_dir / safe_name
            dest.write_bytes(data)
        except Exception:
            self.session.rollback()
            shutil.rmtree(file_dir, ignore_errors=True)
            raise
        file.path = str(dest)
        self.session.commit()
        self.session.refresh(file)
        return file

    def _get_file_model(self, file_id: UUID) -> File:
        file = self.session.get(File, file_id)
        if file is None:
            raise ValueError("File not found")
        return file

    def _warn_if_stored_path_diverges(self, file: File, expected_dir: Path) -> None:
        """Log when a row's stored `path` doesn't live under the directory the
        scoped delete below is about to rmtree.

        The delete always targets `expected_dir` (built from the file id, never
        the stored path), so a tampered path can't turn into an
        arbitrary-directory delete (see
        `test_delete_never_rmtrees_an_escaped_legacy_path`, which must keep
        passing unmodified). That's correct, but it means a row written under
        the pre-org-keyed flat layout points somewhere `expected_dir` was never
        created, so `unlink_file_dirs` finds nothing there, the row still gets
        deleted, and the caller is told the delete succeeded while the real
        bytes are orphaned on disk forever. This does not delete the stored
        path instead (same hole the test above guards against). It only makes
        the drift observable so an operator can grep for it and clean up by
        hand.
        """
        if not file.path:
            return
        stored_dir = Path(file.path).parent
        if stored_dir != expected_dir:
            logger.warning(
                "file %s: stored path %s is not under the scoped delete target "
                "%s; bytes at the stored location will not be removed",
                file.id, file.path, expected_dir,
            )

    def delete_file(self, file_id: UUID) -> bool:
        file = self.session.get(File, file_id)
        if file is None:
            return False
        # Dir from the file id, not the stored path: a legacy row could hold an
        # escaped path, and rmtree-ing its parent would delete an arbitrary dir.
        file_dir = self._root_dir() / str(file.id)
        self._warn_if_stored_path_diverges(file, file_dir)
        self.session.delete(file)
        self.session.commit()
        unlink_file_dirs([file_dir])
        return True

    def delete_by_purpose(self, purpose: str) -> list[Path]:
        """Stage deletion of every file row under `purpose`; return the on-disk
        dirs to unlink once the caller commits.

        Cleans up a conversation's attachments when the conversation (or its
        project) is deleted, otherwise the rows + bytes orphan forever
        (ENG-701). Follows the stage-only convention of
        `TaskObjectService.delete_for_conversation`: the caller owns the commit,
        so the attachment-row delete lands in the SAME transaction as the
        caller's own deletes (a crash mid-way can't leave a half-deleted
        conversation), then the caller unlinks the returned dirs via
        `unlink_file_dirs` AFTER committing.
        """
        rows = list(self.session.exec(self.session.select(File).where(File.purpose == purpose)).all())
        # Dir from the file id, not the stored path (see delete_file): a legacy
        # row could hold an escaped path, and rmtree-ing its parent would delete
        # an arbitrary directory.
        dirs = [self._root_dir() / str(f.id) for f in rows]
        for f, d in zip(rows, dirs):
            self._warn_if_stored_path_diverges(f, d)
        for f in rows:
            self.session.delete(f)
        return dirs

    def get_file_content(self, file_id: UUID) -> tuple[str, str, Path]:
        file = self._get_file_model(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return file.content_type, file.filename, path
