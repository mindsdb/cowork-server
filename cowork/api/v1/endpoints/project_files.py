"""Project file-browsing endpoints.

Ported from cowork/server/routes/projects.py in the old server.
This is not necessarily final state — it was migrated to eliminate
compat stubs and may be refactored later.
"""

import logging
import mimetypes
import ntpath
import os
import secrets
import shutil
import stat
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_lstat,
    dir_open,
    dir_scandir,
    dir_unlink,
    open_pinned_child,
    opened_subdir_nofollow,
    pinned_dir,
    safe_join,
)
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME
from cowork.services.projects import ProjectService


logger = logging.getLogger(__name__)
router = APIRouter()

ANTON_INSTRUCTIONS_FILENAME = "anton.md"
TEXT_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

#: How long a preview token stays usable. Long enough that a preview left open
#: keeps loading its sub-assets, short enough that a token which escapes the
#: browser it was minted in dies the same session.
PREVIEW_TOKEN_TTL_SECONDS = 30 * 60

#: Most live preview mounts one process holds. A mount is a few hundred bytes
#: and only one browser tab spends each one, so a real deployment sits in the
#: tens; the cap is here because nothing but the TTL evicts, and the container
#: limit is 256Mi.
PREVIEW_MOUNT_LIMIT = 2048


@dataclass(frozen=True)
class _PreviewMount:
    """A directory one preview token may read, and who may read it.

    The token is a bearer string in a URL, so an iframe can load it and a
    screenshot, a proxy log or a browser history can leak it. The record is
    therefore what carries the authority, not the string: whoever presents the
    token still has to be the member it was minted for, in the organization it
    was minted in, before it expires.

    Being the minter is not enough on its own, because a mount grants a whole
    directory and the directory it grants may sit ABOVE the private workspaces.
    An .html at the project root mounts the project root, and every member's
    `conversations/<id>/` hangs off it. So the record also carries the project
    root and which workspace, if any, the mounted file lived in, and
    `serves` refuses a path that reaches into a different one.
    """

    parent: Path
    project_base: Path
    workspace: str | None
    org_id: str | None
    user_id: str | None
    expires_at: float

    def readable_by(self, scope: TenantScope) -> bool:
        """Desktop has one user and no organization, so nothing to compare."""
        if not scope.org_mode:
            return True
        return self.org_id == scope.org_id and self.user_id == scope.user_id

    def serves(self, target: Path) -> bool:
        """Whether this mount may hand out `target`.

        Containment in `parent` is checked by the caller. This is the second
        question: a mount parented at or above `conversations/` must not be a
        way into a workspace the minter does not own. Only the workspace the
        mounted file itself lived in is reachable, and a mount taken on a shared
        file reaches no workspace at all.
        """
        if self.org_id is None:
            # Desktop: one user, nothing to keep apart. An org-mode mount always
            # carries an org, because minting reads the project through a scoped
            # session and that raises without one.
            return True
        try:
            rel = target.relative_to(self.project_base).as_posix()
        except ValueError:
            return False
        parts = rel.split("/", 2)
        if len(parts) < 2 or parts[0] != CONVERSATIONS_DIRNAME:
            return True  # shared project file
        return parts[1] == self.workspace


_PROJECT_PREVIEW_MOUNTS: dict[str, _PreviewMount] = {}


def _mount_workspace(target: Path, base: Path) -> str | None:
    """The `conversations/<id>` segment `target` sits in, or None when it is a
    shared project file. `_require_workspace_access` has already established
    that the caller owns it."""
    try:
        rel = target.relative_to(base.resolve()).as_posix()
    except (ValueError, OSError):
        return None
    parts = rel.split("/", 2)
    if len(parts) < 2 or parts[0] != CONVERSATIONS_DIRNAME:
        return None
    return parts[1]


def _register_preview_mount(target: Path, base: Path, scope: TenantScope) -> str:
    """Mint a token for `target`'s directory and bind it to the caller.

    Expired records are dropped here rather than on read: minting is rare and
    reading is per sub-asset, and nothing else ever removes an entry.

    Dropping the expired ones is not enough on its own. The old token was
    `sha256(parent)`, so N mounts of one directory reused one key and the
    registry was bounded by the number of distinct directories; a random token
    per mint means every call inserts, and nothing evicts a record that is still
    inside its 30-minute TTL. So the registry is also capped, oldest expiry
    first. Overrunning the cap costs an early 404 on the least fresh preview,
    which reloads, rather than an unbounded dict in a 256Mi container.
    """
    now = time.time()
    for stale in [t for t, m in _PROJECT_PREVIEW_MOUNTS.items() if m.expires_at <= now]:
        _PROJECT_PREVIEW_MOUNTS.pop(stale, None)
    while len(_PROJECT_PREVIEW_MOUNTS) >= PREVIEW_MOUNT_LIMIT:
        oldest = min(_PROJECT_PREVIEW_MOUNTS, key=lambda t: _PROJECT_PREVIEW_MOUNTS[t].expires_at)
        _PROJECT_PREVIEW_MOUNTS.pop(oldest, None)
    token = secrets.token_urlsafe(32)
    # `target` is already fully resolved by `_safe_relpath`, so its parent is too.
    # Resolving again would be a no-op, and it reads as a containment check that
    # has in fact already happened one frame up.
    _PROJECT_PREVIEW_MOUNTS[token] = _PreviewMount(
        parent=target.parent,
        project_base=base.resolve(),
        workspace=_mount_workspace(target, base),
        org_id=scope.org_id,
        user_id=scope.user_id,
        expires_at=now + PREVIEW_TOKEN_TTL_SECONDS,
    )
    return token


class _FileWriteRequest(BaseModel):
    content: str


class _PreviewMountRequest(BaseModel):
    name: str
    path: str


@dataclass(frozen=True)
class _ValidatedProjectPath:
    """A request path reduced to non-traversing relative components."""

    parts: tuple[str, ...]

    @property
    def value(self) -> str:
        return "/".join(self.parts)


def _validated_project_path(path: str) -> _ValidatedProjectPath:
    """Validate write/delete paths before either handler reaches the disk.

    These routes support nested project files, so a basename-only policy would
    break normal use. Splitting once and rejecting empty, dot and dot-dot
    components gives the handlers the nested shape they need without retaining
    an unchecked path expression. The later resolved-containment and dirfd /
    ``O_NOFOLLOW`` checks remain in place for symlinks and races.
    """
    if not path or len(path) > 4096 or "\x00" in path:
        raise HTTPException(status_code=400, detail="invalid path")
    cleaned = path.replace("\\", "/")
    if cleaned.startswith("/") or ntpath.splitdrive(cleaned)[0]:
        raise HTTPException(status_code=400, detail="invalid path")
    parts = tuple(cleaned.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise HTTPException(status_code=400, detail="invalid path")
    return _ValidatedProjectPath(parts)


ProjectMutationPathDep = Annotated[_ValidatedProjectPath, Depends(_validated_project_path)]


def _project_name_selector(name: str) -> str:
    """Validate an HTTP project name for equality-only catalog selection."""
    selected_name = os.path.basename(name)
    if (
        not selected_name
        or selected_name != name
        or selected_name in {".", ".."}
        or "\\" in selected_name
        or "\x00" in selected_name
    ):
        raise HTTPException(status_code=404, detail="Project not found")
    return selected_name


@contextmanager
def _opened_project_directory_inventory(
    scoped: ScopedSession,
) -> Iterator[tuple[tuple[str, PinnedDir | None], ...]]:
    """Open every scoped project root before a request name selects one.

    This context intentionally has no request-derived argument. Each root is
    built from its scoped database row and pinned before the inventory reaches
    the caller. The surrounding ``ExitStack`` keeps every descriptor alive
    while the caller compares names and closes all of them on every exit path.
    """
    service = ProjectService(scoped)
    projects = service.list_projects()
    with ExitStack() as resources:
        opened: list[tuple[str, PinnedDir | None]] = []
        for project in projects:
            name = str(project.name)
            directory: PinnedDir | None = None
            try:
                # Preserve the historical self-heal for a missing project
                # directory before attempting to pin the row's stored path.
                service.ensure_dir_exists(project)
                path = Path(project.path)
                if path.is_dir():
                    directory = resources.enter_context(
                        pinned_dir(path, nofollow_base=True)
                    )
            except (OSError, TypeError, ValueError):
                pass
            opened.append((name, directory))
        yield tuple(opened)


def _project_dir(name: str, scoped: ScopedSession) -> Path:
    """Resolve a sanitized project name to its scoped on-disk directory."""
    selected_name = _project_name_selector(name)
    service = ProjectService(scoped)
    try:
        project = service.get_project_by_name(selected_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Any project can lose its directory (fresh pod, wiped volume), not just the
    # default one. The row is authoritative, so provision rather than 404 the
    # owner out of their own project.
    service.ensure_dir_exists(project)
    base = Path(project.path)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Project directory not found on disk")
    return base


def _anton_md_path(base: Path) -> Path:
    return base / ".anton" / ANTON_INSTRUCTIONS_FILENAME


def _safe_relpath(rel: str | _ValidatedProjectPath, base: Path) -> Path:
    if isinstance(rel, _ValidatedProjectPath):
        try:
            return safe_join(base, *rel.parts)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
    if not rel:
        raise HTTPException(status_code=400, detail="path required")
    cleaned = rel.replace("\\", "/").lstrip("/")
    candidate = (base / cleaned).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    return candidate


def _file_meta(p: Path, base: Path) -> dict[str, Any] | None:
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    try:
        resolved = p.resolve()
        rel = resolved.relative_to(base.resolve())
    except ValueError:
        return None
    return {
        "path": str(rel).replace("\\", "/"),
        "name": p.name,
        "size": st.st_size,
        "modified": st.st_mtime,
        "is_dir": p.is_dir(),
    }


def _conversation_workspace_ok(
    rel_posix: str, scoped: ScopedSession, cache: dict | None = None
) -> bool:
    """`conversations/<id>/…` is a per-conversation PRIVATE workspace even
    though the project directory is org-shared, so only that conversation's
    owner may list/read/write/delete inside it. Anything else is a shared
    project file. Local mode has one user, so nothing is gated."""
    if not scoped.scope.org_mode:
        return True
    parts = rel_posix.split("/", 2)
    if len(parts) < 2 or parts[0] != CONVERSATIONS_DIRNAME:
        return True  # shared project file
    seg = parts[1]
    if cache is not None and seg in cache:
        return cache[seg]
    try:
        conv_id = UUID(seg)
    except ValueError:
        ok = True  # not a real conversation dir → treat as shared
    else:
        from cowork.services.conversations import ConversationService

        ok = ConversationService(scoped)._owned(conv_id) is not None
    if cache is not None:
        cache[seg] = ok
    return ok


def _require_workspace_access(target: Path, base: Path, scoped: ScopedSession) -> None:
    """404 (no existence oracle) if `target` is inside another member's
    conversation workspace.

    `target` came from `_safe_relpath`, which already confirmed it sits under
    `base`, but re-derive the relative path defensively: a symlink that makes
    the resolved path escape `base` must 404, never raise (CodeQL: user data in
    a path expression)."""
    try:
        rel = target.relative_to(base.resolve()).as_posix()
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="File not found")
    if not _conversation_workspace_ok(rel, scoped):
        raise HTTPException(status_code=404, detail="File not found")


def _require_workspace_path(path: _ValidatedProjectPath, scoped: ScopedSession) -> None:
    """Authorize a validated lexical path before an fd-relative mutation.

    Mutation helpers refuse symlinks in every component, so the lexical
    conversation id is the only workspace the operation can reach.
    """
    if not _conversation_workspace_ok(path.value, scoped):
        raise HTTPException(status_code=404, detail="File not found")


@contextmanager
def _pinned_fd(target: Path, base: Path, flags: int) -> Iterator[int]:
    """Yield a descriptor for `target`, opening every component below `base`
    with O_NOFOLLOW so no planted link can redirect the open.

    `_require_workspace_access` decides ownership from a resolved path string,
    and re-opening that string by name hands the whole chain back to the kernel
    to walk again. A pod mounts its own subtree read-write and can swap a
    directory component for a link in between, so the decision has to be carried
    to the open rather than re-derived from a name afterwards. Same defence
    `ProjectService` applies to the projects root and `_secure_attachments_dir`
    to the attachments tree; see `opened_subdir_nofollow`.

    A planted link raises `OSError` (ELOOP), which callers map to the same 404 a
    genuine miss returns rather than letting it surface as a 500.
    """
    rel = target.relative_to(base.resolve())
    *dirs, name = rel.parts
    with opened_subdir_nofollow(base, *dirs) as pinned:
        fd = dir_open(pinned, name, flags | O_NOFOLLOW)
        try:
            yield fd
        finally:
            os.close(fd)


def _pinned_regular_file(target: Path, base: Path, flags: int = os.O_RDONLY):
    """`_pinned_fd`, plus the "is it actually a file" check every route needs,
    with every failure collapsed onto 404."""
    try:
        cm = _pinned_fd(target, base, flags)
        fd = cm.__enter__()
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        st = os.fstat(fd)
    except OSError:
        cm.__exit__(None, None, None)
        raise HTTPException(status_code=404, detail="File not found")
    if not stat.S_ISREG(st.st_mode):
        cm.__exit__(None, None, None)
        raise HTTPException(status_code=404, detail="File not found")
    return cm, fd, st


def _existing_entry_name(directory: PinnedDir, requested: str) -> str:
    """Return the directory's own name for an exact request component.

    A validated request component is safe to compare, but keeping it as the
    argument to ``openat``/``unlinkat`` still leaves static analysis (and a
    future relaxed validator) with an HTTP-to-filesystem data flow.  Resolve
    the selector against the already-pinned directory instead.  The returned
    value originates in ``scandir``, so every filesystem operation below uses
    a name supplied by that directory, never the HTTP string.
    """
    with dir_scandir(directory) as entries:
        for entry in entries:
            if entry.name == requested:
                return entry.name
    raise FileNotFoundError


@contextmanager
def _opened_existing_project_entry(
    root: PinnedDir, requested_parts: tuple[str, ...]
) -> Iterator[tuple[PinnedDir, str]]:
    """Pin the parent of an existing project entry and yield its disk name.

    Each descent name is obtained from the pinned directory itself and each
    directory is opened with ``O_NOFOLLOW``.  This preserves nested project
    paths without ever joining a request value into a path or handing one to a
    filesystem syscall.
    """
    if not requested_parts:
        raise FileNotFoundError
    with ExitStack() as descendants:
        current = root
        for requested in requested_parts[:-1]:
            name = _existing_entry_name(current, requested)
            parent_stat = dir_lstat(current, name)
            if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(
                parent_stat.st_mode
            ):
                raise FileNotFoundError
            current = open_pinned_child(current, name)
            descendants.callback(current.close)
        yield current, _existing_entry_name(current, requested_parts[-1])


def _pinned_stream(target: Path, base: Path, *, media_type: str, headers: dict) -> StreamingResponse:
    """Serve `target` from a pinned descriptor.

    A `FileResponse` takes a path and opens it after the handler returns, which
    is the one thing the pinning exists to avoid, so the bytes come off the
    descriptor instead. Content-Length comes from `fstat` on that same
    descriptor, so it describes the file being sent rather than whatever the name
    resolves to next.
    """
    cm, fd, st = _pinned_regular_file(target, base)

    def _chunks():
        try:
            while chunk := os.read(fd, 1 << 16):
                yield chunk
        finally:
            cm.__exit__(None, None, None)

    return StreamingResponse(
        _chunks(),
        media_type=media_type,
        headers={**headers, "Content-Length": str(st.st_size)},
    )


@router.get("/{project_name}/instructions")
def get_project_instructions(project_name: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    p = _anton_md_path(base)
    rel = p.relative_to(base).as_posix()
    if p.is_file():
        try:
            st = p.stat()
        except OSError:
            return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": 0, "modified": None, "is_dir": False, "synthetic": True}}
        return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": st.st_size, "modified": st.st_mtime, "is_dir": False}}
    return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": 0, "modified": None, "is_dir": False, "synthetic": True}}


@router.get("/{project_name}/files")
def list_project_files(project_name: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    files: list[dict[str, Any]] = []
    _conv_cache: dict = {}
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        meta = _file_meta(p, base)
        if meta and _conversation_workspace_ok(meta["path"], scoped, _conv_cache):
            files.append(meta)

    anton_rel = _anton_md_path(base).relative_to(base).as_posix()
    if not any(f["path"] == anton_rel for f in files):
        files.insert(0, {
            "path": anton_rel,
            "name": ANTON_INSTRUCTIONS_FILENAME,
            "size": 0,
            "modified": None,
            "is_dir": False,
            "synthetic": True,
        })
    else:
        files.sort(key=lambda f: (f["path"] != anton_rel, f["path"]))

    return {"files": files}


@router.get("/{project_name}/files/{path:path}")
def read_project_file(project_name: str, path: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    _require_workspace_access(target, base, scoped)
    if not target.exists():
        anton_rel = _anton_md_path(base).relative_to(base).as_posix()
        if path == anton_rel:
            return {"path": path, "content": "", "size": 0, "modified": None}
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    # Everything past the gate reads through a pinned chain, so the file whose
    # ownership was just checked is the file that gets opened.
    cm, fd, st = _pinned_regular_file(target, base)
    try:
        if st.st_size > TEXT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large to read inline")
        try:
            content = os.read(fd, TEXT_MAX_BYTES + 1).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=415, detail="File is not valid UTF-8 text") from exc
    finally:
        cm.__exit__(None, None, None)
    return {"path": path, "content": content, "size": st.st_size, "modified": st.st_mtime}


@router.put("/{project_name}/files/{path:path}")
def write_project_file(
    project_name: str,
    path: ProjectMutationPathDep,
    req: _FileWriteRequest,
    scoped: ScopedSessionDep,
):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    _require_workspace_access(target, base, scoped)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    body = req.content or ""
    encoded = body.encode("utf-8")
    if len(encoded) > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MiB cap")
    # Pinned like the read: without it a link planted under the caller's own
    # workspace between the gate and the open lands this write in another
    # member's directory, and `.anton/anton.md` there is an instruction file
    # their agent reads.
    rel = target.relative_to(base.resolve())
    *dirs, name = rel.parts
    try:
        with opened_subdir_nofollow(base, *dirs, create=True) as pinned:
            fd = dir_open(
                pinned, name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_NOFOLLOW, 0o644
            )
            try:
                os.write(fd, encoded)
                st = os.fstat(fd)
            finally:
                os.close(fd)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="File not found")
    return {"path": path.value, "size": st.st_size, "modified": st.st_mtime}


@router.post("/{project_name}/files/upload")
async def upload_project_files(
    project_name: str,
    scoped: ScopedSessionDep,
    files: list[UploadFile] = File(...),
):
    base = _project_dir(project_name, scoped)
    results: list[dict[str, Any]] = []
    for f in files:
        if not f.filename:
            results.append({"name": "", "ok": False, "error": "filename missing"})
            continue
        safe_name = os.path.basename(f.filename).strip()
        if not safe_name or safe_name.startswith("."):
            results.append({"name": f.filename, "ok": False, "error": "invalid filename"})
            continue
        target = base / safe_name
        try:
            data = await f.read()
            target.write_bytes(data)
            results.append({"name": safe_name, "ok": True, "size": len(data)})
        except Exception as exc:
            logger.error("Failed to write file %s: %s", safe_name, exc)
            results.append({"name": safe_name, "ok": False, "error": "File write failed"})
    return {"results": results}


@router.delete("/{project_name}/files/{path:path}")
def delete_project_file(
    project_name: str, path: ProjectMutationPathDep, scoped: ScopedSessionDep
):
    _require_workspace_path(path, scoped)
    try:
        # Each candidate is resolved and pinned from the scoped catalog before
        # the HTTP name is compared with it. The request can only select the
        # already-open handle; it never reaches a Path constructor or open.
        with _opened_project_directory_inventory(scoped) as inventory:
            selected_name = os.path.basename(project_name)
            if (
                not selected_name
                or selected_name != project_name
                or selected_name in {".", ".."}
                or "\\" in selected_name
                or "\x00" in selected_name
            ):
                raise HTTPException(status_code=404, detail="Project not found")
            for server_name, directory in inventory:
                if server_name != selected_name:
                    continue
                if directory is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Project directory not found on disk",
                    )

                # Path components are selectors too. Every syscall name comes
                # back from scanning this already-pinned directory.
                with _opened_existing_project_entry(
                    directory, path.parts
                ) as (parent, disk_name):
                    target_stat = dir_lstat(parent, disk_name)
                    if stat.S_ISLNK(target_stat.st_mode):
                        raise HTTPException(status_code=404, detail="File not found")
                    if stat.S_ISDIR(target_stat.st_mode):
                        raise HTTPException(
                            status_code=400, detail="Path is a directory"
                        )
                    dir_unlink(parent, disk_name)
                break
            else:
                raise HTTPException(status_code=404, detail="Project not found")
    except HTTPException:
        raise
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "deleted", "path": path.value}


@router.delete("/{project_name}/skill_drafts/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill_draft(project_name: str, slug: str, scoped: ScopedSessionDep):
    """Remove a staged skill draft once it is Saved (or dismissed).

    Idempotent: a missing draft is a no-op — Save may race the sweep, and a
    lingering draft is the safe default we're clearing, not a hard error. The
    slug is confined to a direct child of the drafts dir (no traversal).
    """
    # Resolve, then require the target to stay inside the drafts dir — rejects any
    # traversal in `slug` regardless of what it contains.
    drafts_root = os.path.realpath(_project_dir(project_name, scoped) / ".anton" / "skill_drafts")
    folder = os.path.realpath(os.path.join(drafts_root, slug))
    if not folder.startswith(drafts_root + os.sep):
        raise HTTPException(status_code=400, detail="invalid slug")
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


@router.post("/preview-mount-file")
def preview_mount_file(req: _PreviewMountRequest, scoped: ScopedSessionDep):
    base = _project_dir(req.name, scoped)
    target = _safe_relpath(req.path, base)
    _require_workspace_access(target, base, scoped)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Preview mount is only available for HTML files")
    token = _register_preview_mount(target, base, scoped.scope)
    return {
        "token": token,
        "entry": target.name,
        "relUrl": f"/projects/preview-asset/{token}/{target.name}",
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
def preview_asset(
    token: str, rel_path: str, scope: TenantScope = Depends(get_tenant_scope)
):
    # An unknown token, an expired one and another member's one all answer the
    # same 404: which of the three it was is itself information about somebody
    # else's files.
    mount = _PROJECT_PREVIEW_MOUNTS.get(token)
    if mount is None or mount.expires_at <= time.time() or not mount.readable_by(scope):
        raise HTTPException(status_code=404, detail="Preview mount has expired or is unknown")
    parent = mount.parent
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=403, detail="Asset is outside the mounted directory")
    # Containment in the mounted directory is not ownership. A mount taken on an
    # .html at the project root is parented ABOVE `conversations/`, so every
    # member's workspace hangs off it and the containment check above passes for
    # all of them. Same 404 as an unknown token: which it was is information.
    if not mount.serves(target):
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return _pinned_stream(
        target,
        mount.project_base,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.get("/{project_name}/files-raw/{path:path}")
def download_project_file(project_name: str, path: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    _require_workspace_access(target, base, scoped)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return _pinned_stream(
        target,
        base,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{target.name}"'},
    )
