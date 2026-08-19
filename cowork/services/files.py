from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from cowork.common.paths import safe_join
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


def remove_conversation_workspace_dir(project_path: str | Path | None, conversation_id: str | UUID) -> None:
    """Remove ``<project>/conversations/<conv>/`` on conversation delete — the
    per-conversation workspace holding the staged attachments and instructions
    (and, in cloud, session state). Otherwise it orphans on the shared mount,
    keeping a duplicate of the very upload bytes delete_by_purpose reclaims
    (ENG-701 class). Best-effort; the id is validated so it can only ever target
    a real conversation dir. A no-op on desktop, where no such dir exists.
    """
    if project_path is None:
        return
    # Org mode only: the conversation workspace dir is a cloud artifact we
    # create. On desktop no such dir exists from our code, and a user's project
    # could coincidentally hold a `conversations/<id>/` folder — never rmtree it.
    if get_app_settings().tenancy_mode != "org":
        return
    try:
        conv_seg = str(UUID(str(conversation_id)))
    except (ValueError, TypeError):
        return
    target = Path(project_path) / "conversations" / conv_seg
    try:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
    except OSError:
        logger.warning("could not remove conversation workspace dir %s", target, exc_info=True)


def stage_project_instructions(project_path: str | Path, conversation_id: str | UUID) -> bool:
    """Copy the project's ``anton.md`` into the conversation workspace so the
    pod picks it up: anton reads ``<workspace>/.anton/anton.md``, but the pod's
    workspace is the conversation dir while the project's Instructions live at
    the project root — the same project-level/conversation-scoped delivery gap
    as attachments. Idempotent (skips when the copy is current), best-effort.
    Returns whether an instruction file is in place. ``.anton/anton.md`` matches
    anton's Workspace and cowork's project_files endpoint.
    """
    try:
        conv_seg = str(UUID(str(conversation_id)))  # path segment must be a real id, never `..`
    except (ValueError, TypeError):
        return False
    conv_root = Path(project_path) / "conversations" / conv_seg
    # Containment: the workspace is writable by the (untrusted) pod, so resolve
    # symlinks and refuse a dest that escapes the conversation dir — a planted
    # `.anton` symlink must not redirect the write into another tenant's tree.
    try:
        dest = safe_join(conv_root, ".anton", "anton.md")
    except ValueError:
        logger.warning("staged instructions path escapes workspace for conversation %s", conversation_id)
        return False
    src = Path(project_path) / ".anton" / "anton.md"
    if not src.is_file():
        # Instructions removed at the project → drop the stale staged copy so
        # the agent stops seeing them.
        try:
            if dest.is_file():
                dest.unlink()
        except OSError:
            pass
        return False
    try:
        s = src.stat()
        if dest.is_file():
            d = dest.stat()
            if d.st_size == s.st_size and d.st_mtime >= s.st_mtime:
                return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)  # preserves mtime, so the check above stays cheap
        return True
    except OSError:
        logger.warning("could not stage instructions for conversation %s", conversation_id, exc_info=True)
        return False


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
        """Local base on desktop, ``<shared>/<org>/files`` in org mode."""
        return scoped_storage_root(
            Path(get_app_settings().file.root_dir), self.session.scope, store="files"
        )

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

    def _doomed_dirs(self, file: File) -> list[Path]:
        """Dirs to unlink on delete: the id-derived dir under the current root,
        plus the stored path's parent (bytes live there if the root moved) —
        but only when the resolved parent is literally named ``<file.id>``, so
        an escaped legacy path can never aim rmtree at an arbitrary dir."""
        dirs = [self._root_dir() / str(file.id)]
        try:
            stored = Path(file.path).parent.resolve()
            if stored.name == str(file.id) and stored != dirs[0].resolve():
                dirs.append(stored)
        except (ValueError, OSError):
            pass
        return dirs

    def delete_file(self, file_id: UUID) -> bool:
        file = self.session.get(File, file_id)
        if file is None:
            return False
        doomed = self._doomed_dirs(file)
        self.session.delete(file)
        self.session.commit()
        unlink_file_dirs(doomed)
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
        rows = list(self.session.exec(self.session.select(File).where(File.purpose == purpose)).all())
        # Same validated candidates as delete_file (_doomed_dirs).
        dirs = [d for f in rows for d in self._doomed_dirs(f)]
        for f in rows:
            self.session.delete(f)
        return dirs

    def get_file_content(self, file_id: UUID) -> tuple[str, str, Path]:
        file = self._get_file_model(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return file.content_type, file.filename, path

    def stage_conversation_attachments(self, conversation_id: UUID | str, project_path: str | Path) -> int:
        """Copy this conversation's attachments into the pod-visible workspace at
        ``<project_path>/conversations/<conv>/attachments/<file_id>/<name>`` so a
        delegated (cloud) turn can read them off the shared mount — the flat
        files store stays the source of truth.

        Re-run every turn (idempotent: skips a file already staged with the same
        size), so a file attached at message 4 is still there at message 40. The
        staged dir is kept in sync with the live rows — a deleted attachment's
        staged copy is pruned, so the agent stops seeing content the user
        removed (the store stays the source of truth). Best-effort per file:
        one unreadable file is logged and skipped, never fails the turn. Returns
        the number staged.
        """
        try:
            conv_seg = str(UUID(str(conversation_id)))  # path segment must be a real id, never `..`
        except (ValueError, TypeError):
            return 0
        dest_root = Path(project_path) / "conversations" / conv_seg / "attachments"
        # Only copy bytes that live under THIS org's files root — a legacy
        # escaped row (path into another org) must not be dragged into the
        # workspace where the pod would read it.
        try:
            files_root = self._root_dir().resolve()
        except Exception:
            files_root = None
        rows = self.list_file_rows(attachment_purpose(str(conversation_id)))
        staged_ids: set[str] = set()
        staged = 0
        for row in rows:
            try:
                src = Path(row.path)
                if not src.is_file():
                    continue
                if files_root is not None and files_root not in src.resolve().parents:
                    logger.warning("attachment %s path is outside the org files root; skipping", row.id)
                    continue
                # Containment: the pod can write under attachments/, so resolve
                # symlinks and refuse a dest that escapes the conversation dir.
                dest = safe_join(dest_root, str(row.id), src.name)
                if not (dest.is_file() and dest.stat().st_size == row.size):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                staged_ids.add(str(row.id))
                staged += 1
            except (OSError, ValueError):
                logger.warning(
                    "could not stage attachment %s for conversation %s",
                    getattr(row, "id", "?"), conversation_id, exc_info=True,
                )
        # Prune every staged dir we did NOT just (re)stage — covers a deleted
        # row, a row whose source bytes vanished, and one skipped as out-of-root
        # — so the agent stops seeing content the store no longer backs.
        try:
            if dest_root.is_dir():
                for child in dest_root.iterdir():
                    if child.name in staged_ids:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink()
        except OSError:
            logger.warning("could not prune stale staged attachments for conversation %s",
                           conversation_id, exc_info=True)
        return staged
