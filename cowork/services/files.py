from __future__ import annotations

import logging
import os
import shutil
import stat
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_rmtree,
    dir_scandir,
    dir_stat,
    dir_unlink,
    open_pinned_child,
    pinned_dir,
    safe_join,
)
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


def remove_conversation_workspace_dir(
    project_path: str | Path | None, conversation_id: str | UUID
) -> None:
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
    project = Path(project_path)
    project_name = os.path.basename(project.name)
    if (
        project_name != project.name
        or project_name in {"", ".", ".."}
        or "\\" in project_name
        or "\0" in project_name
    ):
        return
    try:
        with pinned_dir(project.parent, nofollow_base=True) as parent:
            project_dir = open_pinned_child(parent, project_name)
            try:
                conversations = open_pinned_child(project_dir, "conversations")
                try:
                    dir_rmtree(conversations, conv_seg)
                finally:
                    conversations.close()
            finally:
                project_dir.close()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "could not remove conversation workspace dir %s",
            Path(project_path) / "conversations" / conv_seg,
            exc_info=True,
        )


def stage_project_instructions(
    project_path: str | Path, conversation_id: str | UUID
) -> bool:
    """Copy the project's ``anton.md`` into the conversation workspace so the
    pod picks it up: anton reads ``<workspace>/.anton/anton.md``, but the pod's
    workspace is the conversation dir while the project's Instructions live at
    the project root — the same project-level/conversation-scoped delivery gap
    as attachments. Re-copied every turn (the file is tiny), best-effort.
    Returns whether an instruction file is in place. ``.anton/anton.md`` matches
    anton's Workspace and cowork's project_files endpoint.
    """
    try:
        conv_seg = str(
            UUID(str(conversation_id))
        )  # path segment must be a real id, never `..`
    except (ValueError, TypeError):
        return False
    conv_root = Path(project_path) / "conversations" / conv_seg
    # Containment: the workspace is writable by the (untrusted) pod, so resolve
    # symlinks and refuse a dest that escapes the conversation dir — a planted
    # `.anton` symlink must not redirect the write into another tenant's tree.
    try:
        dest = safe_join(conv_root, ".anton", "anton.md")
    except ValueError:
        logger.warning(
            "staged instructions path escapes workspace for conversation %s",
            conversation_id,
        )
        return False
    src = Path(project_path) / ".anton" / "anton.md"
    if not src.is_file():
        # Instructions removed at the project → clear the stale staged copy so
        # the agent stops seeing them. Cleared by truncating IN PLACE, never by
        # unlink: the sandbox pod caches NFS handles for this file (gVisor
        # gofer), and deleting the inode leaves the pod failing every stat
        # with ESTALE until the pod is recycled — retries inside the pod
        # cannot recover it (ENG-1817). Truncation keeps the inode, so the
        # pod's cached handle stays valid and reads empty content instead.
        # O_NOFOLLOW + no O_CREAT: the workspace is writable by the untrusted
        # pod and safe_join is not atomic, so a symlink planted at dest after
        # the check must fail (ELOOP) rather than truncate its target, and a
        # file that vanished in between must not be recreated here.
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                fd = os.open(dest, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
                os.close(fd)
        except OSError:
            pass
        return False
    try:
        # anton.md is tiny — always overwrite rather than skip on a size/mtime
        # match: a same-length edit that lands on the same mtime second (both
        # files share the EFS clock) would otherwise serve stale instructions.
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    except OSError:
        logger.warning(
            "could not stage instructions for conversation %s",
            conversation_id,
            exc_info=True,
        )
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


def _secure_attachments_dir(conv_dir: Path) -> PinnedDir | None:
    """Return a symlink-safe handle for ``<conv_dir>/attachments``.

    The worker pod mounts the conversation dir read-write, so it can replace
    ``attachments`` (or any child) with a symlink pointing into another org's
    subtree. cowork-server mounts every org, so staging or pruning *through*
    that link escapes the tenant: worst case a namespace-wide ``rmtree`` of
    every organization's data. ``safe_join`` cannot catch this, because it
    resolves the base as well as the target, so a symlinked base has already
    escaped before the containment check runs.

    Pin the conversation dir by descriptor with ``O_NOFOLLOW``, drop any symlink
    or non-directory squatting the ``attachments`` name, recreate it as a real
    directory, and hand back a handle the caller acts relative to (the same
    defence ``ProjectService`` applies to the projects root). Attachments
    staging is cowork-server-owned state, so self-healing a planted link is
    safe. The caller owns the returned handle and must ``close()`` it. ``None``
    means the dir could not be secured and the caller must not fall back to a
    path.
    """
    try:
        # The conversation dir and its `conversations` parent are cowork-server
        # state (the pod mounts INTO <conv>, it does not create it), so creating
        # them is trusted. Only the `attachments` child below is agent-reachable.
        with pinned_dir(conv_dir, create=True, nofollow_base=True) as conv:
            try:
                st = dir_lstat(conv, "attachments")
                if not stat.S_ISDIR(st.st_mode):
                    # A symlink (S_ISLNK, never S_ISDIR under lstat) or a plain
                    # file squatting the name: remove it and recreate a real dir.
                    dir_unlink(conv, "attachments")
                    dir_mkdir(conv, "attachments")
            except FileNotFoundError:
                dir_mkdir(conv, "attachments")
            return open_pinned_child(conv, "attachments")
    except OSError:
        return None


def _stage_attachment(
    attach: PinnedDir, file_id: str, name: str, src: Path, size: int
) -> None:
    """Copy *src* to ``<attachments>/<file_id>/<name>`` relative to *attach*.

    Never follows a symlink the pod may have planted for the per-id directory or
    the file name (``O_NOFOLLOW`` on every component the agent can reach). Skips
    a copy already present at the right size so the caller stays idempotent.
    """
    try:
        st = dir_lstat(attach, file_id)
        if not stat.S_ISDIR(st.st_mode):
            dir_unlink(attach, file_id)
            dir_mkdir(attach, file_id)
    except FileNotFoundError:
        dir_mkdir(attach, file_id)
    id_dir = open_pinned_child(attach, file_id)
    try:
        try:
            dst_st = dir_stat(id_dir, name, follow_symlinks=False)
            if stat.S_ISREG(dst_st.st_mode) and dst_st.st_size == size:
                return
        except FileNotFoundError:
            pass
        dst_fd = dir_open(
            id_dir, name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_NOFOLLOW, 0o600
        )
        with open(dst_fd, "wb") as dst, open(src, "rb") as fsrc:
            shutil.copyfileobj(fsrc, dst)
    finally:
        id_dir.close()


def _prune_staged_attachments(attach: PinnedDir, keep: set[str]) -> None:
    """Drop every staged entry whose file id is not in *keep*.

    Acts relative to *attach* and never follows a symlink out of the attachments
    dir: a planted link is unlinked (the link only, never its target), a real
    dir is removed with ``rmtree``.
    """
    try:
        entries = list(dir_scandir(attach))
    except OSError:
        logger.warning("could not list staged attachments for pruning", exc_info=True)
        return
    for entry in entries:
        if entry.name in keep:
            continue
        try:
            if entry.is_symlink():
                dir_unlink(attach, entry.name)
            elif entry.is_dir(follow_symlinks=False):
                dir_rmtree(attach, entry.name)
            else:
                dir_unlink(attach, entry.name)
        except OSError:
            logger.warning(
                "could not prune staged attachment %s", entry.name, exc_info=True
            )


class FileService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        """Local base on desktop, ``<shared>/<org>/files`` in org mode."""
        return scoped_storage_root(
            Path(get_app_settings().file.root_dir), self.session.scope, store="files"
        )

    def _owned_select(self):
        """Files are personal. The scoped session enforces the org, but user_id
        has no automatic scoping (see PinService), so without this every org
        member could list/read/delete every other member's files."""
        stmt = self.session.select(File)
        if self.session.scope.org_mode:
            stmt = stmt.where(File.created_by == self.session.scope.user_id)
        return stmt

    def _owned(self, file_id: UUID) -> "File | None":
        """A file only if it belongs to the caller. A bare session.get by PK
        does no owner filter, so a guessed id must not resolve cross-user."""
        return self.session.exec(self._owned_select().where(File.id == file_id)).first()

    def _to_response(self, file: File) -> FileResponse:
        return FileResponse(
            id=str(file.id),
            bytes=file.size,
            created_at=int(file.created_at.timestamp()) if file.created_at else 0,
            filename=file.filename,
            purpose=file.purpose,
        )

    def list_files(self, purpose: str | None = None) -> list[FileResponse]:
        stmt = self._owned_select()
        if purpose is not None:
            stmt = stmt.where(File.purpose == purpose)
        return [self._to_response(f) for f in self.session.exec(stmt).all()]

    def list_file_rows(self, purpose: str) -> list[File]:
        """Raw File rows for callers that need fields the OpenAI-style
        FileResponse drops (content_type, timestamps) — e.g. the
        attachments compat endpoints."""
        return list(
            self.session.exec(self._owned_select().where(File.purpose == purpose)).all()
        )

    def get_file_row(self, file_id: UUID) -> File:
        return self._get_file_model(file_id)

    def relink_purpose(self, old_purpose: str, new_purpose: str) -> int:
        """Repoint every file stored under `old_purpose`. Used when a
        conversation ends up with a different id than the one the client
        uploaded attachments against. Returns the number relinked."""
        files = self.session.exec(
            self._owned_select().where(File.purpose == old_purpose)
        ).all()
        for file in files:
            file.purpose = new_purpose
            self.session.add(file)
        if files:
            self.session.commit()
        return len(files)

    def get_file(self, file_id: UUID) -> FileResponse:
        file = self._owned(file_id)
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

    def create_file_from_bytes(
        self, *, filename: str, content_type: str, data: bytes, purpose: str
    ) -> File:
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
        file = self._owned(file_id)
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
        file = self._owned(file_id)
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
        rows = list(
            self.session.exec(self._owned_select().where(File.purpose == purpose)).all()
        )
        # Same validated candidates as delete_file (_doomed_dirs).
        dirs = [d for f in rows for d in self._doomed_dirs(f)]
        for f in rows:
            self.session.delete(f)
        return dirs

    def delete_by_purpose_for_parent_cascade(self, purpose: str) -> list[Path]:
        """Stage all org-scoped rows owned by an authorized parent resource.

        Project deletion may cascade conversations belonging to several org
        members. The conversation UUID in ``purpose`` comes from an already
        scoped, authorized parent row, so applying the ordinary user filter
        here would orphan another member's attachment rows and bytes.
        """
        rows = list(
            self.session.exec(
                self.session.select(File).where(File.purpose == purpose)
            ).all()
        )
        dirs = [directory for row in rows for directory in self._doomed_dirs(row)]
        for row in rows:
            self.session.delete(row)
        return dirs

    def get_file_content(self, file_id: UUID) -> tuple[str, str, Path]:
        file = self._get_file_model(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return file.content_type, file.filename, path

    def stage_conversation_attachments(
        self, conversation_id: UUID | str, project_path: str | Path
    ) -> int:
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
            conv_seg = str(
                UUID(str(conversation_id))
            )  # path segment must be a real id, never `..`
        except (ValueError, TypeError):
            return 0
        # The pod mounts this conversation dir read-write, so it can replace
        # `attachments` (or a child) with a symlink into another org's subtree.
        # Pin the dir by an O_NOFOLLOW descriptor and do every copy/prune
        # relative to it, so a planted link cannot redirect us out of the tenant
        # (see _secure_attachments_fd). Fail closed rather than fall back to a
        # path the agent can still swap.
        conv_dir = Path(project_path) / "conversations" / conv_seg
        attach = _secure_attachments_dir(conv_dir)
        if attach is None:
            logger.warning(
                "could not secure attachments dir for conversation %s", conversation_id
            )
            return 0
        # Only copy bytes that live under THIS org's files root: a legacy
        # escaped row (path into another org) must not be dragged into the
        # workspace where the pod would read it.
        try:
            files_root = self._root_dir().resolve()
        except Exception:
            files_root = None
        staged_ids: set[str] = set()
        staged = 0
        try:
            rows = self.list_file_rows(attachment_purpose(str(conversation_id)))
            for row in rows:
                try:
                    src = Path(row.path)
                    if not src.is_file():
                        continue
                    if (
                        files_root is not None
                        and files_root not in src.resolve().parents
                    ):
                        logger.warning(
                            "attachment %s path is outside the org files root; skipping",
                            row.id,
                        )
                        continue
                    _stage_attachment(attach, str(row.id), src.name, src, row.size)
                    staged_ids.add(str(row.id))
                    staged += 1
                except (OSError, ValueError):
                    logger.warning(
                        "could not stage attachment %s for conversation %s",
                        getattr(row, "id", "?"),
                        conversation_id,
                        exc_info=True,
                    )
            # Prune every staged entry we did NOT just (re)stage: a deleted row, a
            # row whose source bytes vanished, or one skipped as out-of-root, so
            # the agent stops seeing content the store no longer backs.
            _prune_staged_attachments(attach, staged_ids)
        finally:
            attach.close()
        return staged
