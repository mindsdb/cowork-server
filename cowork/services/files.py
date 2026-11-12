from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import os
import re
import shutil
import stat
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

from fastapi import UploadFile
from pydantic import BaseModel, ValidationError

from cowork.common.logger import log_context
from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    cowork_home,
    dir_atomic_write,
    dir_chmod_folder,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_rename,
    dir_rmdir,
    dir_rmtree,
    dir_scandir,
    dir_stat,
    dir_unlink,
    open_fd,
    open_pinned_child,
    pinned_dir,
    read_bounded,
    safe_join,
    write_all,
)
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope, scoped_storage_root
from cowork.db.units import run_db, run_efs
from cowork.models.file import File
from cowork.schemas.files import FileResponse, StoredFileContent
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME, PROJECT_FILES_DIRNAME, REMOVED_DIRNAME
from cowork.services.project_file_view import (
    ABSENT_ERRNOS,
    OrdinaryFile,
    WalkState,
    utf8_path,
    walk_ordinary_files,
)

logger = logging.getLogger(__name__)

#: Removals of an upload's folder whose row was refused, held here until each
#: ends so the event loop keeps a reference to it.
_upload_removals: set[asyncio.Task[None]] = set()


def _remove_refused_upload(*, file_dir: Path, scope: TenantScope) -> None:
    """Remove ``file_dir`` in the background, waiting as long as it takes for
    a file store slot, so a busy or stuck mount never holds the caller's own
    error, and the folder never outlives a row that was not written."""
    removal = asyncio.ensure_future(
        run_efs(partial(shutil.rmtree, file_dir, ignore_errors=True), scope=scope, deadline=False)
    )
    _upload_removals.add(removal)
    removal.add_done_callback(_upload_removals.discard)


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
    src = _instructions_path(project_path)
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
        # file that vanished in between must not be recreated here. Windows
        # has no O_NOFOLLOW (the constant is 0 there), which leaves only that
        # post-check race open; Windows runs local mode only, with no untrusted
        # pod to plant the link (see ``cowork.common.paths``).
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                fd = open_fd(dest, os.O_WRONLY | os.O_TRUNC | O_NOFOLLOW)
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


# Files the OS leaves in a folder, which are not the user's.
_OS_FILES = frozenset({"desktop.ini", "thumbs.db", "icon\r"})


def project_holds_files(project_path: str | Path, *, cloud: bool) -> bool:
    """Whether a project folder holds files the turn's agent can read. Stops
    at the first. A missing folder holds none; any other read error raises,
    and the caller decides.

    A cloud turn (``cloud``) reads the copy stage_project_files makes, so it
    counts by the copy's own rule, an ordinary file
    (``cowork.services.project_file_view``): a project routes as holding files
    exactly when its turn gets one, copied, or named or counted as not copied
    (a file whose path is not valid UTF-8 is counted). A turn in
    this process works in the project folder itself, so anything there the
    user or the agent put counts: a file, or a folder with something in it,
    other than hidden ones (``.anton``), the server's own folders and OS
    files."""
    if cloud:
        try:
            with pinned_dir(project_path) as project, closing(
                walk_ordinary_files(project, state=WalkState())
            ) as walk:
                return next(walk, None) is not None
        except (FileNotFoundError, NotADirectoryError):
            return False
    from cowork.services.artifact_roots import SERVER_PROJECT_FOLDERS

    try:
        with os.scandir(project_path) as entries:
            for entry in entries:
                name = entry.name.casefold()
                if name.startswith(".") or name in SERVER_PROJECT_FOLDERS or name in _OS_FILES:
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    return True
                try:
                    with os.scandir(entry.path) as inner:
                        if next(inner, None) is not None:
                            return True
                except (FileNotFoundError, NotADirectoryError):
                    continue
    except (FileNotFoundError, NotADirectoryError):
        return False
    return False


def _instructions_path(project_path: str | Path) -> Path:
    """Where a project keeps its instructions for the agent."""
    return Path(project_path) / ".anton" / "anton.md"


# The boilerplate of the template anton writes into ``.anton/anton.md`` when it
# sets up a workspace (``anton.workspace.ANTON_MD_TEMPLATE``), matched as
# loosely as anton matches it: any date, CRLF, a BOM, a reflowed comment, and
# the heading or the date line deleted. Mirrors ``anton.workspace.
# anton_md_user_text``, which the locked anton-agent does not ship yet;
# tests/test_response_routing.py pins it to the installed anton's template.
_TEMPLATE_HEADER_RE = re.compile(
    r"\A\ufeff?\s*(?:# Anton Workspace[ \t\r]*$)?"
    r"(?:\s*Created:[ \t]*\d{4}-\d{2}-\d{2}[ \t\r]*$)?",
    re.MULTILINE,
)
_TEMPLATE_COMMENT_RE = re.compile(
    r"<!--\s*Add project context, conventions, and notes below\.\s*"
    r"Anton reads this file at the start of every conversation\.\s*-->"
)


def instructions_user_text(content: str) -> str:
    """What the user wrote in an ``anton.md``, without anton's template
    boilerplate: "" for a blank file or the untouched template."""
    text = _TEMPLATE_HEADER_RE.sub("", content)
    return _TEMPLATE_COMMENT_RE.sub("", text).strip()


def project_has_instructions(project_path: str | Path) -> bool:
    """Whether the project has instructions: a regular ``.anton/anton.md``
    holding text the user wrote (``stage_project_instructions`` gives the
    agent that file, and only when it is a regular file). The template anton
    writes when it sets up a workspace is not instructions, so a project the
    agent has worked in once still routes like one with none. A missing path
    or a non-file there means none; any other read error raises. ``is_file``
    comes first so a pipe or a device at that path is never opened."""
    path = _instructions_path(project_path)
    try:
        if not path.is_file():
            return False
        content = path.read_bytes().decode("utf-8", errors="replace")
    except (FileNotFoundError, NotADirectoryError):
        return False
    return bool(instructions_user_text(content))


class ProjectContext(NamedTuple):
    """What a project's folder says about routing a turn."""

    #: Files the user or the agent put there, or an attachment on any turn.
    has_files: bool
    #: Project instructions, which only the agent's prompt carries.
    has_instructions: bool


def read_project_context(project_path: str | Path, *, files_known: bool, cloud: bool) -> ProjectContext:
    """Both routing facts about a project's folder in one pass on the file
    store. ``files_known`` says an attachment already counts as files, so
    the folder is not listed; ``cloud`` says the turn runs in a pod
    (project_holds_files). A missing folder holds neither; any read error
    other than a missing path raises, and the caller decides."""
    return ProjectContext(
        has_files=files_known or project_holds_files(project_path, cloud=cloud),
        has_instructions=project_has_instructions(project_path),
    )


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
            return _secure_workspace_child(conv, "attachments")
    except OSError:
        return None


def _secure_workspace_child(parent: PinnedDir, name: str) -> PinnedDir:
    """Open the folder ``name`` in ``parent``, a folder the pod can write, as
    a pinned handle the caller must ``close()``. A link or anything else that
    is not a folder squatting the name is removed (the link only, never its
    target) and a real folder made in its place. The folder is opened by
    _open_workspace_folder, so one the pod made unreadable still opens, and a
    link planted between the check and the open raises ``OSError``."""
    try:
        st = dir_lstat(parent, name)
        if not stat.S_ISDIR(st.st_mode):
            # A symlink (S_ISLNK, never S_ISDIR under lstat) or a plain
            # file squatting the name: remove it and recreate a real dir.
            dir_unlink(parent, name)
            _make_dir(parent, name)
    except FileNotFoundError:
        _make_dir(parent, name)
    return _open_workspace_folder(parent, name)


#: The mode a folder the pod can write gets back when the pod took away any of
#: its owner's read, write or search access. The pod runs as the server's own
#: user, so it can chmod any folder in its conversation's folder, and a folder
#: it made unreadable (``0o000``) or unwritable would refuse every later turn's
#: open, copy or removal there.
_WORKSPACE_FOLDER_MODE = 0o700


def _lacks_owner_access(info: os.stat_result) -> bool:
    return info.st_mode & stat.S_IRWXU != stat.S_IRWXU


def _restore_folder_mode(parent: PinnedDir, name: str) -> os.stat_result:
    """Give the folder ``name`` in ``parent`` _WORKSPACE_FOLDER_MODE when the
    pod took its owner's access away, without opening it for reading and
    never through a link (dir_chmod_folder), and return its stat. The folder
    a later open finds may not be this one, so the caller compares their
    ``st_dev`` and ``st_ino`` (_open_workspace_folder). Anything but a folder
    is left alone, for the caller's open to refuse."""
    info = dir_lstat(parent, name)
    if parent.fd is not None and stat.S_ISDIR(info.st_mode) and _lacks_owner_access(info):
        # Windows (no descriptor) runs local mode only, with no pod.
        return dir_chmod_folder(parent, name, mode=_WORKSPACE_FOLDER_MODE)
    return info


def _open_workspace_folder(parent: PinnedDir, name: str) -> PinnedDir:
    """Open the folder ``name`` in ``parent``, a folder the pod can write, as
    a pinned handle the caller must ``close()``, giving its owner's access
    back first (_restore_folder_mode) so a folder the pod made unreadable or
    unwritable still opens and can be written. The open never follows a link,
    and must find the folder whose mode was set: one the pod swapped in
    between raises ``OSError``."""
    if parent.fd is None:
        # Windows runs local mode only, with no pod (cowork.common.paths).
        return open_pinned_child(parent, name)
    restored = _restore_folder_mode(parent, name)
    folder = open_pinned_child(parent, name)
    try:
        info = os.fstat(folder.fd)
        if (info.st_dev, info.st_ino) != (restored.st_dev, restored.st_ino):
            raise OSError(errno.EAGAIN, "the folder was replaced while its mode was restored", name)
        _own_folder(folder)
    except BaseException:
        folder.close()
        raise
    return folder


def _own_folder(folder: PinnedDir) -> None:
    """Give an open folder _WORKSPACE_FOLDER_MODE back when the pod took its
    owner's access away after it was opened."""
    if folder.fd is not None and _lacks_owner_access(os.fstat(folder.fd)):
        os.fchmod(folder.fd, _WORKSPACE_FOLDER_MODE)


def _make_dir(parent: PinnedDir, name: str) -> None:
    """Make the folder ``name``. One that appears between the check and the
    make (another staging of the same conversation, or the pod) is left to
    the open that follows, which refuses anything but a real folder."""
    try:
        dir_mkdir(parent, name)
    except FileExistsError:
        pass


def _stage_attachment(
    attach: PinnedDir, file_id: str, name: str, src: Path, size: int
) -> None:
    """Copy *src* to ``<attachments>/<file_id>/<name>`` relative to *attach*.

    Never follows a symlink the pod may have planted for the per-id directory or
    the file name (``O_NOFOLLOW`` on every component the agent can reach). Skips
    a copy already present at the right size so the caller stays idempotent.
    """
    id_dir = _secure_workspace_child(attach, file_id)
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


def upload_basename(*, name: str | None) -> str | None:
    """The last component of an untrusted file name, stripped, or None when
    nothing usable is left (``''``, ``.`` or ``..``).

    Keeping only the last component stops ``../`` or an absolute path from
    leaving the file's own folder; ``.`` and ``..`` would name that folder or
    the files root itself. Uploads and channel attachments both go through
    here before store_file.
    """
    basename = Path(name or "").name.strip()
    return None if basename in ("", ".", "..") else basename


def _write_upload(file_dir: Path, name: str, contents: bytes) -> Path:
    """Write an upload into its own new folder, or leave no folder behind."""
    try:
        file_dir.mkdir(parents=True)
        dest = file_dir / name
        dest.write_bytes(contents)
    except Exception:
        shutil.rmtree(file_dir, ignore_errors=True)
        raise
    return dest


def _insert_file(session, *, file: File) -> File:
    """Insert an upload's row, as a unit. Stamps its organization and author
    (ScopedSession.add) and returns it loaded."""
    session.add(file)
    session.commit()
    session.refresh(file)
    return file


def _files_root(scope: TenantScope) -> Path:
    """Local base on desktop, ``<shared>/<org>/files`` in org mode."""
    return scoped_storage_root(Path(get_app_settings().file.root_dir), scope, store="files")


async def store_file(
    *, scope: TenantScope, filename: str, content_type: str, data: bytes, purpose: str
) -> File:
    """Save ``data`` as a new file named ``filename``, which the caller has
    already reduced to a safe basename (upload_basename), and return its
    loaded row.

    The storage root refuses a scope with no organization where files are
    partitioned (scoped_storage_root), so a missing principal fails before any
    bytes hit disk. The bytes go to the file store, then the row to the
    database, each in a worker thread of its own: no pooled connection waits on
    the file server, and the event loop waits for neither.
    """
    file = File(
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size=len(data),
        purpose=purpose,
        path="",
    )
    file_dir = _files_root(scope) / str(file.id)
    dest = await run_efs(partial(_write_upload, file_dir, filename, data), scope=scope)
    file.path = str(dest)
    try:
        return await run_db(partial(_insert_file, file=file), scope=scope)
    except Exception:
        # Not on a cancel: run_db lets a unit that already holds its
        # connection commit before the cancel propagates, so the row may name
        # these bytes. A cancel before that leaves a folder no row names.
        _remove_refused_upload(file_dir=file_dir, scope=scope)
        raise


async def save_upload(upload: UploadFile, *, purpose: str, scope: TenantScope) -> File:
    """Save an uploaded file in ``scope`` (store_file) and return its row."""
    contents = await upload.read()
    return await store_file(
        scope=scope,
        filename=upload_basename(name=upload.filename) or "upload",
        content_type=upload.content_type,
        data=contents,
        purpose=purpose,
    )


def file_response(file: File) -> FileResponse:
    return FileResponse(
        id=str(file.id),
        bytes=file.size,
        created_at=int(file.created_at.timestamp()) if file.created_at else 0,
        filename=file.filename,
        purpose=file.purpose,
    )


class FileService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        return _files_root(self.session.scope)

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
        return file_response(file)

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

    def stored_file(self, *, file_id: UUID) -> StoredFileContent:
        """The file's row: where its bytes are, read from the database only."""
        file = self._get_file_model(file_id)
        return StoredFileContent(content_type=file.content_type, filename=file.filename, path=Path(file.path))

    def get_file_content(self, file_id: UUID) -> StoredFileContent:
        """``stored_file``, refused with ValueError when its bytes are gone."""
        stored = self.stored_file(file_id=file_id)
        if not stored.path.exists():
            raise ValueError("File content not found on disk")
        return stored

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


# A cloud turn's copy of its project's files.
#
# The pod mounts its conversation's folder and not the project, so before each
# cloud turn the server copies the project's ordinary files (the rule in
# cowork.services.project_file_view, which routing uses too) into
# ``conversations/<id>/project-files/``. The pod can write that folder, so
# every step there is pinned and never follows a link, and what the server
# copied is recorded outside it, where the pod can't reach.

#: The list a copy writes into the project-files folder, naming the files it
#: left out and why. A hidden name, so no project file can take it.
NOT_COPIED_FILENAME = ".not-copied.json"
#: The most left-out files the list names; it counts the rest.
MAX_NAMED_NOT_COPIED = 1000
_COPY_CHUNK_BYTES = 1 << 20
#: The mode of every file in a copy.
_COPY_FILE_MODE = 0o600
#: How many entries one removal step examines, and for how long, before it
#: stops (_RemovalBudget). Removing what a copy should not hold, and deleting
#: what REMOVED_DIRNAME holds, each get this much per turn.
_REMOVAL_MAX_ENTRIES = 10_000
_REMOVAL_SECONDS = 2.0
#: Errors from opening a copy that mean something the pod put at its name (a
#: folder, a link, a pipe, a file it made unwritable), so the name is cleared
#: and the copy made afresh.
_REPLACEABLE_ERRNOS = frozenset({errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.EACCES, errno.EPERM, errno.ETXTBSY})


class ProjectFileCopyLimits(NamedTuple):
    """How much of a project one conversation's copy holds, and how long one
    turn spends walking the project and copying (TurnQueueSettings.project_files_*)."""

    max_file_bytes: int
    max_files: int
    max_total_bytes: int
    seconds: float

    @classmethod
    def from_settings(cls) -> ProjectFileCopyLimits:
        settings = TurnQueueSettings()
        return cls(
            max_file_bytes=settings.project_files_max_file_bytes,
            max_files=settings.project_files_max_files,
            max_total_bytes=settings.project_files_max_total_bytes,
            seconds=settings.project_files_copy_seconds,
        )


class NotCopiedReason(StrEnum):
    TOO_LARGE = "too_large"
    TOO_MANY_FILES = "too_many_files"
    OVER_TOTAL_BYTES = "over_total_bytes"
    OUT_OF_TIME = "out_of_time"


class NotCopiedFile(BaseModel):
    #: The file's path below the project folder, ``/``-separated.
    path: str
    size: int
    reason: NotCopiedReason


class NotCopiedFiles(BaseModel):
    """The list a copy writes into the project-files folder as
    NOT_COPIED_FILENAME, so the agent can name the project files it did not
    get. Written only when a file was left out, the walk stopped early or the
    copy failed."""

    files: list[NotCopiedFile] = []
    #: Files left out and counted, not named: those past
    #: MAX_NAMED_NOT_COPIED, and those whose path is not valid UTF-8, which
    #: the list can't hold.
    unnamed: int = 0
    #: The walk stopped before it saw every file: the project holds more
    #: entries than one walk examines, folders deeper than it descends, or the
    #: turn's copy time ran out. Files past that point were neither copied
    #: nor named.
    listing_truncated: bool = False
    #: The copy could not finish, so the server emptied the folder.
    copy_failed: bool = False


class FileIdentity(BaseModel):
    """What a stat says about one file. The change time is the kernel's: no
    process can set it, so a copy the pod rewrote and then dated back still
    differs here."""

    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> FileIdentity:
        return cls(ino=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


class StagedCopy(BaseModel):
    """One project file the server copied, and its copy as the server left it."""

    source: FileIdentity
    staged: FileIdentity


class CopyRecord(BaseModel):
    """What the server copied into one conversation's project-files folder,
    kept in the project-file-copies store (copy_records_root), which no pod
    mounts and no route serves. A file is skipped as unchanged only when both
    its source and its copy still match the record."""

    #: By the file's path below the project folder, ``/``-separated.
    files: dict[str, StagedCopy] = {}


class ProjectFilesOutcome(StrEnum):
    #: The copy finished, whatever it left out.
    STAGED = "staged"
    #: The copy could not finish. The folder was emptied and its list says so.
    FAILED = "failed"
    #: The folder may still hold copies the project no longer backs: the copy
    #: could not finish and the folder could not be emptied, or the project is
    #: not one of the turn's organization. The turn must not run.
    REFUSED = "refused"


class ProjectFilesStaged(NamedTuple):
    """What one turn's copy did, for its log line."""

    #: Files in the copy once staging ends, and their bytes.
    files: int
    bytes: int
    #: Files written this turn, and their bytes; the rest were unchanged.
    copied: int
    copied_bytes: int
    not_copied: int
    outcome: ProjectFilesOutcome
    elapsed_ms: int

    @property
    def failed(self) -> bool:
        return self.outcome is not ProjectFilesOutcome.STAGED


def _nothing_staged(outcome: ProjectFilesOutcome, elapsed_ms: int = 0) -> ProjectFilesStaged:
    return ProjectFilesStaged(0, 0, 0, 0, 0, outcome=outcome, elapsed_ms=elapsed_ms)


class ProjectFilesRefused(RuntimeError):
    """A cloud turn's project-files folder may hold copies its project no
    longer backs (ProjectFilesOutcome.REFUSED), so the turn is refused rather
    than run on them."""


class _Copied(NamedTuple):
    staged: StagedCopy
    #: The copy holds the bytes the server wrote and nothing changed it while
    #: they were read back, so the record may trust it on a later turn.
    verified: bool


def copy_records_root(scope: TenantScope) -> Path:
    """The store that keeps each conversation's CopyRecord:
    ``<shared>/<org>/project-file-copies`` in org mode, beside the projects
    store, so outside every pod's mounts and every project file route."""
    return scoped_storage_root(cowork_home() / "project-file-copies", scope, store="project-file-copies")


def copy_record_path(scope: TenantScope, conversation_id: str | UUID) -> Path:
    """Where the conversation's CopyRecord is kept."""
    return copy_records_root(scope) / f"{UUID(str(conversation_id))}.json"


def remove_copy_record(path: Path) -> None:
    """Remove a deleted conversation's CopyRecord. Best-effort, logged."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove the project files record %s", path.name, exc_info=True)


def stage_project_files(
    project_path: str | Path,
    conversation_id: str | UUID,
    *,
    scope: TenantScope,
    limits: ProjectFileCopyLimits | None = None,
) -> ProjectFilesStaged:
    """Copy the project's ordinary files into
    ``<project>/conversations/<conv>/project-files/``, mirroring their folders,
    for a cloud turn's pod, which mounts the conversation's folder only.

    A file whose source and copy both still match the conversation's
    CopyRecord is left as it is. A changed file is rewritten in place
    (truncated, never removed and made again), so a pod that holds the copy
    open keeps a valid handle. A copy whose project file is gone, or is now
    left out, is removed. Files over ``limits`` are left out and named in
    NOT_COPIED_FILENAME; a file whose path is not valid UTF-8 is left out
    and counted there, not named.

    ``limits.seconds`` bounds the walk and the copying: once it passes, the
    walk stops, a file being copied is removed and named as out of time, and
    the list says the walk stopped early. Copies the walk did not reach this
    turn are removed with the rest, since nothing checked them against the
    project, and a later turn copies them again. Removing those and saving
    the record run after the budget.

    Removing what the folder should not hold runs within a budget
    (_RemovalBudget). A folder is removed by moving it into REMOVED_DIRNAME,
    which each turn deletes within another budget after the copy, a level at
    a time, so no tree the pod builds there, however deep or wide, costs a
    turn more than its budgets. When the removal runs out of budget, the
    whole project-files folder is moved there and the copy runs again into a
    new one, with ``limits.seconds`` again, so the pod's leftovers never fail
    a turn.

    Fails closed: when the copy can't finish (a read or write error other
    than a file that went away mid-copy), the project-files folder is emptied
    and its list says the copy failed, so the turn never reads a copy the
    project no longer backs. When even that fails, or the project is not in
    the scope's projects store, the outcome is REFUSED and the caller must
    not run the turn. Org mode only: the caller is the remote turn's staging
    (stage_remote_workspace in cowork.handlers.responses), on the file
    store's threads.
    """
    from cowork.services.projects import projects_root

    started = time.monotonic()

    def elapsed_ms() -> int:
        return round((time.monotonic() - started) * 1000)

    try:
        conv_seg = str(UUID(str(conversation_id)))
    except (ValueError, TypeError):
        return _nothing_staged(ProjectFilesOutcome.REFUSED)
    project = Path(project_path)
    if project.parent != projects_root(scope=scope):
        # Every other step trusts this path: the record is keyed by the
        # conversation alone, and the copy reads whatever folder it names.
        logger.error(
            "refusing to copy project files from outside the organization's projects",
            extra=log_context(conversation_id=conv_seg),
        )
        return _nothing_staged(ProjectFilesOutcome.REFUSED)
    limits = limits or ProjectFileCopyLimits.from_settings()
    try:
        staged = _stage_project_files(
            project, conv_seg, record_path=copy_record_path(scope, conv_seg), limits=limits,
            deadline=started + limits.seconds,
        )
    except Exception:
        logger.warning(
            "could not copy the project's files for conversation %s; emptying its copy",
            conv_seg,
            exc_info=True,
        )
        emptied = _empty_project_files_copy(project, conv_seg)
        outcome = ProjectFilesOutcome.FAILED if emptied else ProjectFilesOutcome.REFUSED
        return _nothing_staged(outcome, elapsed_ms())
    return staged._replace(elapsed_ms=elapsed_ms())


def _stage_project_files(
    project: Path, conv_seg: str, *, record_path: Path, limits: ProjectFileCopyLimits, deadline: float,
) -> ProjectFilesStaged:
    record = _read_copy_record(record_path)
    with _opened_copy_folders(project, conv_seg) as place:
        if place is None:
            # No project folder, so no conversation folder inside it either.
            return _nothing_staged(ProjectFilesOutcome.STAGED)
        try:
            copy = _copy_pass(place, record=record, limits=limits, deadline=deadline)
        except _RemovalBudgetSpent:
            # The pod left more in the folder than one turn removes: set the
            # folder aside whole and copy again into a new one, with the
            # copy's full time, since nothing copied so far is left.
            logger.warning(
                "the project files copy of conversation %s holds too much to remove; setting it aside",
                conv_seg,
            )
            place.folders.set_aside()
            copy = _copy_pass(place, record=record, limits=limits, deadline=time.monotonic() + limits.seconds)
        place.folders.delete_removed(budget=_RemovalBudget.start())
    _write_copy_record(record_path, copy.new_record)
    return ProjectFilesStaged(
        files=len(copy.kept), bytes=copy.total, copied=copy.copied, copied_bytes=copy.copied_bytes,
        not_copied=len(copy.not_copied) + copy.unnameable, outcome=ProjectFilesOutcome.STAGED, elapsed_ms=0,
    )


def _copy_pass(
    place: _CopyPlace, *, record: CopyRecord, limits: ProjectFileCopyLimits, deadline: float,
) -> _Copy:
    """Copy each ordinary file of the project into the project-files folder,
    then remove everything else there and write the not-copied list when
    something was left out. Raises _RemovalBudgetSpent, with the folder only
    partly pruned, when the removal runs out of its budget."""
    copy = _Copy(record=record, limits=limits, deadline=deadline)
    state = WalkState()
    with _MirrorFolders(place.folders) as mirror_folders, closing(
        walk_ordinary_files(place.source, state=state)
    ) as walk:
        for found in walk:
            if time.monotonic() >= deadline:
                # Stop the walk too, so a large project holds the slot no
                # longer than the budget.
                state.truncated = True
                break
            copy.add(found, mirror_folders)
    not_copied = NotCopiedFiles(
        files=copy.not_copied[:MAX_NAMED_NOT_COPIED],
        unnamed=max(0, len(copy.not_copied) - MAX_NAMED_NOT_COPIED) + copy.unnameable,
        listing_truncated=state.truncated,
    )
    keep = set(copy.kept)
    list_it = bool(copy.not_copied) or copy.unnameable > 0 or state.truncated
    if list_it:
        keep.add((NOT_COPIED_FILENAME,))
    _prune_folder(place.folders, keep=keep, budget=_RemovalBudget.start())
    if list_it:
        _write_in_place(place.folders, NOT_COPIED_FILENAME, not_copied.model_dump_json().encode())
    return copy


class _RemovalBudgetSpent(Exception):
    """A removal step examined as many entries, or ran as long, as its
    _RemovalBudget allows."""


@dataclass
class _RemovalBudget:
    """How much more one removal step may do: entries it may still examine,
    and the time it must stop by."""

    entries: int
    deadline: float

    @classmethod
    def start(cls) -> _RemovalBudget:
        return cls(entries=_REMOVAL_MAX_ENTRIES, deadline=time.monotonic() + _REMOVAL_SECONDS)

    def out_of_time(self) -> bool:
        return time.monotonic() >= self.deadline

    def spend(self) -> bool:
        """Count one entry examined: False, and nothing counted, once the
        budget is spent."""
        if self.entries <= 0 or self.out_of_time():
            return False
        self.entries -= 1
        return True


class _CopyFolders:
    """The folders one staging works in, all inside the conversation's
    folder, which the pod can write: the conversation's own, its
    project-files folder (``mirror``), and REMOVED_DIRNAME, opened when first
    needed. The caller must ``close()`` it."""

    def __init__(self, conversation: PinnedDir) -> None:
        self.conversation = conversation
        self.mirror = _secure_workspace_child(conversation, PROJECT_FILES_DIRNAME)
        self._removed: PinnedDir | None = None

    def close(self) -> None:
        self.mirror.close()
        self.close_removed()

    def removed(self) -> PinnedDir:
        if self._removed is None:
            self._removed = _secure_workspace_child(self.conversation, REMOVED_DIRNAME)
        return self._removed

    def remove(self, parent: PinnedDir, name: str) -> None:
        """Remove ``name`` from ``parent``: anything but a folder by its name
        alone (a link, never its target), and a folder by moving it, with
        whatever it holds, into REMOVED_DIRNAME under a new name, which takes
        one step however much the folder holds. delete_removed deletes it.

        The open REMOVED_DIRNAME may be gone by the time the folder moves:
        another staging of the conversation deleted it once it was empty, or
        the pod removed it. The move then fails as if ``name`` were gone, so
        a folder still at ``name`` is moved once more into REMOVED_DIRNAME
        made afresh, and a second failure raises."""
        try:
            if not stat.S_ISDIR(dir_lstat(parent, name).st_mode):
                dir_unlink(parent, name)
                return
            # Moving a folder to another folder rewrites its "..", which needs
            # its owner's write access.
            _restore_folder_mode(parent, name)
        except FileNotFoundError:
            return
        for _attempt in range(2):
            try:
                dir_rename(parent, name, self.removed(), uuid4().hex)
                return
            except FileNotFoundError:
                pass
            try:
                dir_lstat(parent, name)
            except FileNotFoundError:
                return
            self.close_removed()
        raise OSError(errno.ENOENT, "the removed folder went away twice while a folder moved into it", name)

    def set_aside(self) -> None:
        """Move the whole project-files folder into REMOVED_DIRNAME and make
        a new, empty one in its place."""
        self.remove(self.conversation, PROJECT_FILES_DIRNAME)
        fresh = _secure_workspace_child(self.conversation, PROJECT_FILES_DIRNAME)
        self.mirror.close()
        self.mirror = fresh

    def delete_removed(self, *, budget: _RemovalBudget) -> None:
        """Delete what REMOVED_DIRNAME holds until it is empty, then the
        folder itself, or until ``budget`` is spent; a later turn deletes the
        rest. Each folder there is emptied one level at a time: its files are
        removed and each folder in it is moved up into REMOVED_DIRNAME, so
        however deep a tree the pod built, the deletion holds two folders
        open and never recurses. Never raises: what the pod left there must
        not fail the turn, so an error is logged and the rest left for later."""
        try:
            try:
                dir_lstat(self.conversation, REMOVED_DIRNAME)
            except FileNotFoundError:
                return
            emptied = False
            while not emptied:
                emptied = True
                # self.removed() each time, never a handle kept from before:
                # a move into it may have made it afresh (remove).
                for name in _each_name(self.removed(), budget=budget):
                    emptied = False
                    self._empty_removed_entry(name, budget=budget)
            self.close_removed()
            dir_rmdir(self.conversation, REMOVED_DIRNAME)
        except _RemovalBudgetSpent:
            return
        except OSError:
            logger.warning(
                "could not delete the removed project files of conversation %s",
                self.conversation.path.name, exc_info=True,
            )

    def close_removed(self) -> None:
        if self._removed is not None:
            self._removed.close()
            self._removed = None

    def _empty_removed_entry(self, name: str, *, budget: _RemovalBudget) -> None:
        removed = self.removed()
        try:
            if not stat.S_ISDIR(dir_lstat(removed, name).st_mode):
                dir_unlink(removed, name)
                return
            folder = _open_workspace_folder(removed, name)
        except FileNotFoundError:
            return
        try:
            for child in _each_name(folder, budget=budget):
                self.remove(folder, child)
        finally:
            folder.close()
        try:
            # A move into the removed folder may have made it afresh.
            dir_rmdir(self.removed(), name)
        except FileNotFoundError:
            pass


class _CopyPlace(NamedTuple):
    #: The project folder the copy reads.
    source: PinnedDir
    folders: _CopyFolders


@contextmanager
def _opened_copy_folders(project: Path, conv_seg: str) -> Iterator[_CopyPlace | None]:
    """The project folder and the conversation's copy folders (_CopyFolders),
    the project-files folder made when missing; None when there is no project
    folder. The conversations folder is the server's: the pod mounts its
    conversation's folder inside it and can't reach it. The conversation's
    folder is the pod's mount, so the pod can't replace it but can change its
    mode, and is opened as one of its folders (_open_workspace_folder)."""
    with ExitStack() as stack:
        try:
            parent = stack.enter_context(pinned_dir(project.parent, nofollow_base=True))
            source: PinnedDir | None = open_pinned_child(parent, project.name)
        except (FileNotFoundError, NotADirectoryError):
            source = None
        if source is None:
            yield None
            return
        stack.callback(source.close)
        conversations = _open_or_make_dir(source, CONVERSATIONS_DIRNAME)
        stack.callback(conversations.close)
        _make_dir(conversations, conv_seg)
        conversation = _open_workspace_folder(conversations, conv_seg)
        stack.callback(conversation.close)
        folders = _CopyFolders(conversation)
        stack.callback(folders.close)
        yield _CopyPlace(source, folders)


class _MirrorFolders:
    """The copy's folders on the way to the file being copied. The walk is
    depth first, so each folder is opened (and made, when missing) once, while
    the walk is inside it, and closed when the walk leaves it."""

    def __init__(self, folders: _CopyFolders) -> None:
        self.folders = folders
        self._open: list[tuple[str, PinnedDir]] = []

    def __enter__(self) -> _MirrorFolders:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def folder(self, parts: tuple[str, ...]) -> PinnedDir:
        """The copy's folder at ``parts`` below the project-files folder."""
        depth = 0
        while depth < min(len(parts), len(self._open)) and self._open[depth][0] == parts[depth]:
            depth += 1
        while len(self._open) > depth:
            self._open.pop()[1].close()
        for name in parts[depth:]:
            parent = self._open[-1][1] if self._open else self.folders.mirror
            self._open.append((name, _secure_workspace_child(parent, name)))
        return self._open[-1][1] if self._open else self.folders.mirror

    def close(self) -> None:
        while self._open:
            self._open.pop()[1].close()


class _Copy:
    """One turn's copy, file by file, in the order the walk finds them."""

    def __init__(self, *, record: CopyRecord, limits: ProjectFileCopyLimits, deadline: float) -> None:
        self.record = record
        self.limits = limits
        self.deadline = deadline
        self.new_record = CopyRecord()
        self.kept: list[tuple[str, ...]] = []
        self.not_copied: list[NotCopiedFile] = []
        #: Files left out because their path is not valid UTF-8: the list
        #: and the record are JSON, which can't hold such a name, so they are
        #: counted only (utf8_path).
        self.unnameable = 0
        self.total = 0
        self.copied = 0
        self.copied_bytes = 0

    def _leave_out(self, found: OrdinaryFile, size: int, reason: NotCopiedReason) -> None:
        self.not_copied.append(NotCopiedFile(path="/".join(found.parts), size=size, reason=reason))

    def add(self, found: OrdinaryFile, folders: _MirrorFolders) -> None:
        if not utf8_path(found.parts):
            self.unnameable += 1
            return None
        size = found.info.st_size
        if size > self.limits.max_file_bytes:
            return self._leave_out(found, size, NotCopiedReason.TOO_LARGE)
        if len(self.kept) >= self.limits.max_files:
            return self._leave_out(found, size, NotCopiedReason.TOO_MANY_FILES)
        if self.total + size > self.limits.max_total_bytes:
            return self._leave_out(found, size, NotCopiedReason.OVER_TOTAL_BYTES)
        previous = self.record.files.get("/".join(found.parts))
        if previous is None and time.monotonic() >= self.deadline:
            # Nothing to check, and no time to copy: no folder is opened.
            return self._leave_out(found, size, NotCopiedReason.OUT_OF_TIME)
        target = folders.folder(found.parts[:-1])
        if previous is not None and _unchanged(previous, source=found.info, target=target, name=found.name):
            self._keep(found, previous, record=True)
            return None
        if time.monotonic() >= self.deadline:
            return self._leave_out(found, size, NotCopiedReason.OUT_OF_TIME)
        outcome = _copy_file(
            found, target, folders=folders.folders,
            max_file_bytes=self.limits.max_file_bytes,
            room=self.limits.max_total_bytes - self.total,
            deadline=self.deadline,
        )
        if isinstance(outcome, NotCopiedReason):
            return self._leave_out(found, size, outcome)
        if outcome is not None:
            self._keep(found, outcome.staged, record=outcome.verified)
            self.copied += 1
            self.copied_bytes += outcome.staged.staged.size
        return None

    def _keep(self, found: OrdinaryFile, staged: StagedCopy, *, record: bool) -> None:
        if record:
            self.new_record.files["/".join(found.parts)] = staged
        self.kept.append(found.parts)
        self.total += staged.staged.size


def _unchanged(previous: StagedCopy, *, source: os.stat_result, target: PinnedDir, name: str) -> bool:
    """Whether the source is the one the record copied and the copy is still
    the file the server wrote: same inode, size, and modification and change
    times on both, and the copy still readable at its mode. The pod can
    change the mode while the server writes the copy, before the stat the
    record keeps, so the mode is checked here too."""
    if previous.source != FileIdentity.of(source):
        return False
    try:
        info = dir_lstat(target, name)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == _COPY_FILE_MODE
        and FileIdentity.of(info) == previous.staged
    )


def _copy_file(
    found: OrdinaryFile, target: PinnedDir, *, folders: _CopyFolders, max_file_bytes: int, room: int,
    deadline: float,
) -> _Copied | NotCopiedReason | None:
    """Copy one project file into ``target`` under its own name. None when the
    file went away, changed size or stopped being an ordinary file since the
    walk saw it (its copy is then pruned); a reason when it no longer fits, or
    the budget ran out mid-copy (the partial copy is removed)."""
    try:
        src = dir_open(found.directory, found.name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    except OSError as error:
        if error.errno in ABSENT_ERRNOS:
            return None
        raise
    try:
        source = os.fstat(src)
        if not stat.S_ISREG(source.st_mode) or source.st_nlink != 1:
            return None
        if source.st_size > max_file_bytes:
            return NotCopiedReason.TOO_LARGE
        if source.st_size > room:
            return NotCopiedReason.OVER_TOTAL_BYTES
        dst = _open_copy_target(target, found.name, folders=folders)
        out_of_time = False
        try:
            digest = hashlib.sha256()
            written = 0
            for chunk in read_bounded(src, source.st_size, _COPY_CHUNK_BYTES):
                if time.monotonic() >= deadline:
                    out_of_time = True
                    break
                digest.update(chunk)
                write_all(dst, chunk)
                written += len(chunk)
            if not out_of_time:
                if written != source.st_size:
                    return None
                os.fsync(dst)
                checked = _read_back(dst, size=written, digest=digest.digest(), deadline=deadline)
                staged = checked if checked is not None else os.fstat(dst)
        finally:
            os.close(dst)
        if out_of_time:
            folders.remove(target, found.name)
            return NotCopiedReason.OUT_OF_TIME
        return _Copied(
            StagedCopy(source=FileIdentity.of(source), staged=FileIdentity.of(staged)),
            verified=checked is not None,
        )
    finally:
        os.close(src)


def _read_back(fd: int, *, size: int, digest: bytes, deadline: float) -> os.stat_result | None:
    """The copy's stat when it holds exactly the bytes the server wrote and
    nothing changed it while they were read back; None otherwise, or when the
    budget runs out first. The pod can write the copy while the server copies
    it, and a stat taken after such a write would record the pod's bytes as
    the server's: a copy that is not verified here is kept for this turn but
    left out of the record, so the next turn copies it again."""
    before = os.fstat(fd)
    if before.st_size != size:
        return None
    os.lseek(fd, 0, os.SEEK_SET)
    check = hashlib.sha256()
    # One byte past the size, so bytes the pod appended show.
    for chunk in read_bounded(fd, size + 1, _COPY_CHUNK_BYTES):
        if time.monotonic() >= deadline:
            return None
        check.update(chunk)
    after = os.fstat(fd)
    if FileIdentity.of(after) != FileIdentity.of(before) or check.digest() != digest:
        return None
    return after


def _open_copy_target(target: PinnedDir, name: str, *, folders: _CopyFolders) -> int:
    """A descriptor for reading and writing the copy ``name`` in ``target``,
    emptied.

    An existing copy that is a regular file with one name is emptied in place,
    keeping its inode for a pod that holds it open. Anything else the pod put
    there (a link, a folder, a pipe, a second name for one of its own files,
    a file it made unreadable) is removed (_CopyFolders.remove), never
    written through, and a new private file made. A name
    made between the check and the create (another staging of the same
    conversation, or the pod) is checked once more."""
    flags = os.O_RDWR | O_NOFOLLOW | O_NONBLOCK
    for _attempt in range(2):
        try:
            fd = dir_open(target, name, flags)
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno not in _REPLACEABLE_ERRNOS:
                raise
            folders.remove(target, name)
        else:
            info = os.fstat(fd)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                try:
                    os.ftruncate(fd, 0)
                    os.fchmod(fd, _COPY_FILE_MODE)
                except BaseException:
                    os.close(fd)
                    raise
                return fd
            os.close(fd)
            folders.remove(target, name)
        try:
            return dir_open(target, name, flags | os.O_CREAT | os.O_EXCL, _COPY_FILE_MODE)
        except FileExistsError:
            continue
    raise FileExistsError(errno.EEXIST, "a copy's name was taken twice while it was made", name)


def _write_in_place(folders: _CopyFolders, name: str, data: bytes) -> None:
    """Write ``data`` as the file ``name`` in the project-files folder."""
    fd = _open_copy_target(folders.mirror, name, folders=folders)
    try:
        write_all(fd, data)
    finally:
        os.close(fd)


def _folders_of(keep: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """Every folder on the way to a path in ``keep``."""
    return {parts[:depth] for parts in keep for depth in range(1, len(parts))}


def _each_name(folder: PinnedDir, *, budget: _RemovalBudget) -> Iterator[str]:
    """Each name in ``folder``, for the caller to remove or keep, from a
    listing taken first: removing names while a listing of the same folder
    runs may skip others, on a network file store above all. Each name listed
    spends ``budget``, so a folder the pod filled is never read past it, and
    the names listed before it ran out are still handed out, so a later turn
    always gets further. Raises _RemovalBudgetSpent before the next name once
    the time is up, and after the last one when the listing was cut short."""
    names: list[str] = []
    complete = True
    with dir_scandir(folder) as scan:
        for entry in scan:
            if not budget.spend():
                complete = False
                break
            names.append(entry.name)
    for name in names:
        if budget.out_of_time():
            raise _RemovalBudgetSpent
        yield name
    if not complete:
        raise _RemovalBudgetSpent


def _prune_folder(folders: _CopyFolders, *, keep: set[tuple[str, ...]], budget: _RemovalBudget) -> None:
    """Remove every entry below the project-files folder whose path is not
    in ``keep``, nor a folder on the way to one. A folder to remove is moved
    out whole (_CopyFolders.remove), so the prune descends only into folders
    on the way to a kept path, which the walk's depth bound keeps shallow. It
    does so from a list, not by recursion, with one of them open at a time.
    Each entry examined spends ``budget``, which raises _RemovalBudgetSpent
    once spent."""
    _own_folder(folders.mirror)
    on_the_way = _folders_of(keep)
    pending: list[tuple[str, ...]] = [()]
    while pending:
        prefix = pending.pop()
        try:
            folder = _open_copy_path(folders.mirror, prefix)
        except FileNotFoundError:
            continue
        try:
            for name in _each_name(folder, budget=budget):
                parts = (*prefix, name)
                if parts in keep:
                    continue
                if parts in on_the_way:
                    try:
                        is_folder = stat.S_ISDIR(dir_lstat(folder, name).st_mode)
                    except FileNotFoundError:
                        continue
                    if is_folder:
                        pending.append(parts)
                        continue
                folders.remove(folder, name)
        finally:
            if folder is not folders.mirror:
                folder.close()


def _open_copy_path(mirror: PinnedDir, parts: tuple[str, ...]) -> PinnedDir:
    """The copy's folder at ``parts`` below ``mirror``, or ``mirror`` itself
    for none, opening each folder on the way (_open_workspace_folder) and
    closing it once the next is open, or once opening the next fails, since
    the pod can remove or replace a folder on the way at any time. A new
    handle is the caller's to close."""
    current = mirror
    for name in parts:
        try:
            child = _open_workspace_folder(current, name)
        finally:
            if current is not mirror:
                current.close()
        current = child
    return current


def _open_or_make_dir(parent: PinnedDir, name: str) -> PinnedDir:
    _make_dir(parent, name)
    return open_pinned_child(parent, name)


#: How many times a copy that failed tries to empty its folder when the pod
#: took a folder's access away. The pod runs as the server's user, so it can
#: take that access away again right after the server gives it back
#: (_open_workspace_folder), and each try opens the folders afresh, which
#: gives it back once more.
_EMPTY_ATTEMPTS = 3


def _empty_project_files_copy(project: Path, conv_seg: str) -> bool:
    """Remove everything in the conversation's project-files folder, making
    the folder when missing, and write a list there that says the copy failed
    (NotCopiedFiles.copy_failed), so the agent tells the user rather than
    answering without the project's files. A folder that can't be emptied
    within its budget is set aside whole (_CopyFolders.set_aside). A try
    refused access is made again, up to _EMPTY_ATTEMPTS. True when that is
    done, or there is no project folder to copy from; False when it could
    not be, logged. The CopyRecord needs no change, since a removed copy
    never matches it."""
    for attempt in range(1, _EMPTY_ATTEMPTS + 1):
        try:
            _empty_project_files_copy_once(project, conv_seg)
        except PermissionError:
            if attempt < _EMPTY_ATTEMPTS:
                logger.warning(
                    "access refused while emptying the project files copy of conversation %s; trying again",
                    conv_seg, exc_info=True,
                )
                continue
            logger.error(
                "could not empty the project files copy of conversation %s", conv_seg, exc_info=True,
            )
            return False
        except Exception:
            logger.error(
                "could not empty the project files copy of conversation %s", conv_seg, exc_info=True,
            )
            return False
        return True
    return False


def _empty_project_files_copy_once(project: Path, conv_seg: str) -> None:
    """One try of _empty_project_files_copy, which raises when it fails."""
    with _opened_copy_folders(project, conv_seg) as place:
        if place is None:
            return
        try:
            _prune_folder(place.folders, keep=set(), budget=_RemovalBudget.start())
        except (_RemovalBudgetSpent, OSError):
            logger.warning(
                "could not empty the project files copy of conversation %s; setting it aside",
                conv_seg, exc_info=True,
            )
            place.folders.set_aside()
        _write_in_place(
            place.folders, NOT_COPIED_FILENAME, NotCopiedFiles(copy_failed=True).model_dump_json().encode(),
        )


def _read_copy_record(record_path: Path) -> CopyRecord:
    """The conversation's CopyRecord, or an empty one when there is none or it
    does not parse. Any other read error raises."""
    try:
        raw = record_path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return CopyRecord()
    try:
        return CopyRecord.model_validate_json(raw)
    except ValidationError:
        logger.warning("discarding an unreadable project files record %s", record_path.name)
        return CopyRecord()


def _write_copy_record(record_path: Path, record: CopyRecord) -> None:
    """Save the record. A record that could not be saved leaves the old one,
    which is safe: every copy rewritten since has a new change time, so the
    old record matches none of them, and they are copied again."""
    try:
        with pinned_dir(record_path.parent, create=True) as root:
            dir_atomic_write(root, record_path.name, content=record.model_dump_json().encode())
    except OSError:
        logger.warning("could not save the project files record %s", record_path.name, exc_info=True)
