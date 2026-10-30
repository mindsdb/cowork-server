"""Project file-browsing endpoints.

Ported from cowork/server/routes/projects.py in the old server.
This is not necessarily final state — it was migrated to eliminate
compat stubs and may be refactored later.
"""

import errno
import logging
import mimetypes
import ntpath
import os
import secrets
import stat
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cowork.api.v1.content_disposition import attachment_disposition
from cowork.api.v1.endpoints.connectors.sharepoint import (
    _KEY_NO_LONGER_WORKS,
    _granted_site,
    _stored_connection,
)
from cowork.api.v1.permissions import AuthenticatedInOrgMode, DesktopOnly, require
from cowork.api.v1.pinned_response import PinnedStreamingResponse

from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_replace,
    dir_rmdir,
    dir_rmtree,
    dir_scandir,
    dir_unlink,
    open_pinned_child,
    opened_subdir_nofollow,
    pinned_dir,
    read_bounded,
    safe_join,
)
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import (
    ScopedSession,
    ScopedSessionDep,
    TenantScope,
    get_tenant_scope,
)
from cowork.db.units import run_efs, run_efs_unit
from cowork.models.project import Project
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.principal import Principal, get_principal
from cowork.schemas.project_files import (
    CANONICAL_PROJECT_MEMORY_PATHS,
    ProjectFileDeleteResponse,
    ProjectFileListResponse,
    ProjectFileReadResponse,
    ProjectFileWriteResponse,
    ProjectInstructionsResponse,
)
from cowork.schemas.shared_resources import MutableResourceCapabilities
from cowork.schemas.sharepoint import (
    SharePointImportedFile,
    SharePointImportResponse,
    SharePointSkippedItem,
)
from cowork.services.artifact_roots import CONVERSATIONS_DIRNAME
from cowork.services.connectors.sharepoint import (
    DownloadBudget,
    SharePointClient,
    SharePointError,
    SharePointErrorKind,
)
from cowork.services.projects import ProjectService
from cowork.services.product_permissions import require_product_permission
from cowork.services.shared_resources import (
    PROJECT,
    PROJECT_INSTRUCTIONS,
    SharedResourceAccess,
    project_resource_key,
)


logger = logging.getLogger(__name__)

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py. (preview_asset checks a mount token
# instead of taking ScopedSessionDep, but the token is scoped per-mount, not
# a bypass, so it needs nothing of its own here.)
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

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


def _workspaces_private(scope: TenantScope) -> bool:
    """Whether `conversations/<id>/` is a per-conversation PRIVATE workspace
    under `scope`: org mode, or a trusted-identity deployment that shares a
    project while keeping conversations per person. Elsewhere (desktop, the
    operator view, a partition's own projects) every workspace is the one
    reader's, so nothing is kept apart."""
    return scope.org_mode or (
        scope.personal
        and scope.partitioned_for(PersonalResource.CONVERSATIONS)
        and not scope.partitioned_for(PersonalResource.PROJECTS)
    )


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
    # Whether conversation workspaces are private under the scope that minted
    # this token (org mode, or a shared project with per-person conversations).
    # The token stores it so the session-free asset route keeps the decision
    # without re-reading the scope.
    workspaces_private: bool
    expires_at: float

    def readable_by(self, scope: TenantScope) -> bool:
        """Where projects are partitioned (org mode, or a person's partition)
        the reader must be the minter. Where projects are shared but
        conversations are private, a workspace mount must also be the minter's,
        so a teammate's token doesn't open another person's workspace. A shared
        project's root mount stays shareable (desktop has one user, the operator
        view reads everyone's)."""
        if scope.partitioned_for(PersonalResource.PROJECTS) or (
            self.workspaces_private and self.workspace is not None
        ):
            return self.org_id == scope.org_id and self.user_id == scope.user_id
        return True

    def serves(self, target: Path) -> bool:
        """Whether this mount may hand out `target`.

        Containment in `parent` is checked by the caller. This is the second
        question: a mount parented at or above `conversations/` must not be a
        way into a workspace the minter does not own. Only the workspace the
        mounted file itself lived in is reachable, and a mount taken on a shared
        file reaches no workspace at all.
        """
        if not self.workspaces_private:
            # Nowhere keeps workspaces apart (desktop, the operator view, a
            # partition's own projects, or a deployment that shares
            # conversations too), so every path the mount contains is served.
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
        oldest = min(
            _PROJECT_PREVIEW_MOUNTS, key=lambda t: _PROJECT_PREVIEW_MOUNTS[t].expires_at
        )
        _PROJECT_PREVIEW_MOUNTS.pop(oldest, None)
    token = secrets.token_urlsafe(32)
    # Always the minter's real partition and sub, so a token minted where
    # projects are shared but conversations are private still names the one
    # person who may open a workspace mount through it.
    # `target` is already fully resolved by `_safe_relpath`, so its parent is too.
    # Resolving again would be a no-op, and it reads as a containment check that
    # has in fact already happened one frame up.
    _PROJECT_PREVIEW_MOUNTS[token] = _PreviewMount(
        parent=target.parent,
        project_base=base.resolve(),
        workspace=_mount_workspace(target, base),
        org_id=scope.org_id,
        user_id=scope.user_id,
        workspaces_private=_workspaces_private(scope),
        expires_at=now + PREVIEW_TOKEN_TTL_SECONDS,
    )
    return token


class _FileWriteRequest(BaseModel):
    content: str


class _PreviewMountRequest(BaseModel):
    name: str
    path: str


@dataclass(frozen=True)
class _InstructionsWriteContext:
    project: Project
    access: SharedResourceAccess
    existed: bool
    previous: bytes | None
    claim: SharedResourceAttribution | None
    claim_token: str | None
    mutation_context: Any | None
    coordination_context: ExitStack


@dataclass(frozen=True)
class _InstructionsDeleteContext:
    project: Project
    access: SharedResourceAccess
    previous: bytes | None
    mutation_context: Any
    coordination_context: ExitStack


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
    parts: list[str] = []
    for raw_part in cleaned.split("/"):
        # Keep a standard, sink-recognized sanitizer on every value retained by
        # the dependency.  The equality check preserves the route's existing
        # rejection semantics instead of silently accepting only a suffix.
        part = os.path.basename(raw_part)
        if not part or part != raw_part or part in {".", ".."}:
            raise HTTPException(status_code=400, detail="invalid path")
        parts.append(part)
    return _ValidatedProjectPath(tuple(parts))


async def _project_mutation_path(path: str) -> _ValidatedProjectPath:
    """The validation as a coroutine, so a request needs no default thread
    for it."""
    return _validated_project_path(path)


ProjectMutationPathDep = Annotated[
    _ValidatedProjectPath, Depends(_project_mutation_path)
]


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
                child_name = os.path.basename(path.name)
                if (
                    not child_name
                    or child_name != path.name
                    or child_name in {".", ".."}
                ):
                    raise ValueError("invalid project directory name")

                # Pin the projects-store directory itself without following a
                # link, then open the project as its descriptor-relative child.
                # Opening ``path`` directly with O_NOFOLLOW protects the final
                # component but still lets a swapped ``path.parent`` redirect
                # the lookup before that final open.
                parent = resources.enter_context(
                    pinned_dir(path.parent, nofollow_base=True)
                )
                directory = open_pinned_child(parent, child_name)
                resources.callback(directory.close)
            except (OSError, TypeError, ValueError):
                pass
            opened.append((name, directory))
        yield tuple(opened)


@contextmanager
def _opened_selected_project_directory(
    project_name: str,
    scoped: ScopedSession,
) -> Iterator[PinnedDir]:
    """Select one already-pinned scoped project directory by exact name."""
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
            if not secrets.compare_digest(server_name, selected_name):
                continue
            if directory is None:
                raise HTTPException(
                    status_code=404,
                    detail="Project directory not found on disk",
                )
            yield directory
            return
    raise HTTPException(status_code=404, detail="Project not found")


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
        raise HTTPException(
            status_code=404, detail="Project directory not found on disk"
        )
    return base


def _anton_md_path(base: Path) -> Path:
    return base / ".anton" / ANTON_INSTRUCTIONS_FILENAME


def _resolved_project_parts(
    base: Path,
    path: _ValidatedProjectPath,
) -> tuple[str, ...]:
    target = _safe_relpath(path, base)
    try:
        return target.relative_to(base.resolve()).parts
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc


def _is_instructions_path(
    path: _ValidatedProjectPath,
    base: Path | None = None,
) -> bool:
    parts = _resolved_project_parts(base, path) if base is not None else path.parts
    return parts == (".anton", ANTON_INSTRUCTIONS_FILENAME)


def _reject_generic_project_memory_mutation(
    path: _ValidatedProjectPath,
    scoped: ScopedSession,
    *,
    base: Path | None = None,
) -> None:
    """Keep org memory ownership/audit behind its canonical API.

    Desktop remains a single-user filesystem surface. In org mode these exact
    shared slots have first-nonempty-writer semantics that the generic bytes
    response cannot represent, so direct PUT/DELETE must fail before disk I/O.
    """
    if not scoped.scope.org_mode:
        return
    parts = _resolved_project_parts(base, path) if base is not None else path.parts
    instructions = (".anton", ANTON_INSTRUCTIONS_FILENAME)
    protected = (*CANONICAL_PROJECT_MEMORY_PATHS, instructions)
    if parts == instructions:
        return
    if parts in CANONICAL_PROJECT_MEMORY_PATHS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Project memory must be changed through the canonical "
                "PUT/DELETE /api/v1/memory/ endpoint"
            ),
        )
    if any(
        parts == canonical[: len(parts)] or canonical == parts[: len(canonical)]
        for canonical in protected
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This path is reserved for a protected .anton resource",
        )


def _project_for_name(name: str, scoped: ScopedSession) -> Project:
    try:
        return ProjectService(scoped).get_project_by_name(_project_name_selector(name))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _instructions_fields_from_access(
    project: Project,
    access: SharedResourceAccess,
    *,
    modified: float | None,
) -> dict[str, Any]:
    """Render instruction metadata with the caller's current lock owner."""
    key = project_resource_key(project.id)
    pending = access.claim_is_pending(PROJECT_INSTRUCTIONS, key)
    fallback_modified_at = (
        datetime.fromtimestamp(modified, tz=timezone.utc)
        if modified is not None
        else None
    )
    can_change = not pending and access.can_change(project.created_by)
    return {
        "attribution": access.attribution(
            PROJECT_INSTRUCTIONS,
            key,
            fallback_modified_at=fallback_modified_at,
        ),
        "capabilities": MutableResourceCapabilities(
            can_edit=can_change,
            can_delete=can_change,
        ),
    }


def _instructions_file_exists(scoped: ScopedSession, project_id: UUID) -> bool:
    """Whether `.anton/anton.md` exists, resolved from the project's live row.

    `recover_stale_claim` calls this while holding the instructions coordination
    lock, which is the same lock a rename holds across its directory move, so
    reading the row here sees either the pre-rename or the post-rename path and
    never a half-applied one.
    """
    current = scoped.get(Project, project_id)
    if current is None:
        return False
    if scoped.scope.org_mode:
        scoped.refresh(current)
    return _anton_md_path(Path(current.path)).is_file()


def _instructions_fields(
    project: Project,
    scoped: ScopedSession,
    principal: Principal | None = None,
    *,
    modified: float | None,
) -> dict[str, Any]:
    """Instruction metadata for a read.

    This runs once per listed file and once per instructions read, so it takes
    no lock of its own: `recover_stale_claim` decides lock-free that there is
    nothing to recover, and only takes the coordination lock (and with it a
    dedicated unpooled connection) when a claim really may have expired.
    """
    access = SharedResourceAccess(scoped, principal)
    key = project_resource_key(project.id)
    if access.org_mode and access.has_trusted_actor:
        access.recover_stale_claim(
            PROJECT_INSTRUCTIONS,
            key,
            resource_exists=lambda: _instructions_file_exists(scoped, project.id),
        )
    return _instructions_fields_from_access(
        project,
        access,
        modified=modified,
    )


def _require_instructions_change(
    project_name: str,
    scoped: ScopedSession,
    principal: Principal | None = None,
) -> tuple[Project, SharedResourceAccess, ExitStack]:
    selected_name = _project_name_selector(project_name)
    project = _project_for_name(selected_name, scoped)
    access = SharedResourceAccess(scoped, principal)
    coordination = ExitStack()
    try:
        coordination.enter_context(
            access.coordination_lock(PROJECT, project_resource_key(project.id))
        )
        # Rename/delete may have completed while this request waited. Refresh
        # only after taking the parent lock and reject a stale route name rather
        # than recreating its former directory.
        project = ProjectService(scoped).get_project(project.id)
        if scoped.scope.org_mode:
            scoped.refresh(project)
        if project.name != selected_name:
            raise HTTPException(status_code=404, detail="Project not found")
        access.require_change(
            project.created_by,
            detail="Only the project creator or an organization admin can edit project instructions",
        )
        coordination.enter_context(
            access.coordination_lock(
                PROJECT_INSTRUCTIONS,
                project_resource_key(project.id),
            )
        )
    except Exception:
        coordination.close()
        raise
    return project, access, coordination


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


#: A project directory Cowork allocated holds the agent's own output, so its
#: size is bounded by what the agent wrote. A folder the user chose can be a
#: repository or a home directory, and one request used to materialise every
#: path beneath it with a stat and a resolve each before returning anything.
_MAX_LISTED_FILES = 2000

#: Entries examined, as opposed to returned. The cap above counts files the
#: caller may actually see, so on its own an unreadable subtree could still be
#: walked without limit before any of them were found.
_MAX_EXAMINED_ENTRIES = 50_000


@dataclass
class _WalkBudget:
    """Whether the entry ceiling, rather than the file cap, stopped the walk."""

    exhausted: bool = False


def _iter_project_files(base: Path, budget: _WalkBudget) -> Iterator[Path]:
    """Candidate files under `base`, breadth-first, bounded by entries seen.

    Breadth-first so a truncated listing shows the user's own top-level files
    instead of whatever a depth-first walk reached inside the first large
    subdirectory it happened to enter.

    A symlinked directory is neither descended into nor listed, which is what
    `Path.rglob` did: it yielded the link itself and the caller skipped it as a
    directory. Every desktop project has `skills/<slug>` directory symlinks
    from `reconcile_project`, so descending would spend the budget on trees
    whose entries `_file_meta` then discards for resolving outside `base`.
    """
    queue: deque[Path] = deque([base])
    examined = 0
    while queue:
        try:
            entries = sorted(queue.popleft().iterdir())
        except OSError:
            continue
        for entry in entries:
            examined += 1
            if examined > _MAX_EXAMINED_ENTRIES:
                budget.exhausted = True
                return
            try:
                if entry.is_symlink():
                    if entry.is_dir():
                        continue
                elif entry.is_dir():
                    queue.append(entry)
                    continue
            except OSError:
                continue
            yield entry


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
    though the project directory is shared, so only that conversation's owner
    may list/read/write/delete inside it: in org mode, and in a project a
    trusted-identity deployment shares while it keeps conversations per person.
    Anything else is a shared project file. Elsewhere nothing is gated: the
    desktop has one user, the operator view reads everyone's, and a personal
    partition's own projects hold only that person's conversations."""
    scope = scoped.scope
    if not _workspaces_private(scope):
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


async def _authorize_artifact_file_mutation(
    path: ProjectMutationPathDep, scope: ScopeDep,
) -> None:
    """Protect artifact bytes reached through the generic file adapter.

    Mutations reject symlinks at every path component. Case-folding also covers
    the supported case-insensitive desktop filesystems. Artifact storage and
    its parent directories stay behind the same grant.
    """
    parts = tuple(part.casefold() for part in path.parts)
    if parts[0] == CONVERSATIONS_DIRNAME:
        if len(parts) == 1:
            await require_product_permission(scope, "artifact.manage")
            return
        if len(parts) == 2:
            try:
                UUID(parts[1])
            except ValueError:
                return
            await require_product_permission(scope, "artifact.manage")
            return
        parts = parts[2:]
    artifact_root = (".anton", "artifacts")
    if parts[:2] == artifact_root or parts == artifact_root[:1]:
        await require_product_permission(scope, "artifact.manage")


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


@contextmanager
def _opened_pinned_descendant(
    root: PinnedDir,
    names: tuple[str, ...],
    *,
    create: bool,
    created: list[tuple[str, ...]] | None = None,
) -> Iterator[PinnedDir]:
    """Descend from an already-pinned project root without rebuilding its Path.

    Every component is reduced to a basename again beside the descriptor-based
    operation. POSIX resolves it with ``dir_fd`` and ``O_NOFOLLOW``; the local
    Windows fallback receives the same single-component guarantee. With
    ``create``, each folder this call made is appended to ``created`` as its
    path below ``root``, so a caller can remove what it added.
    """
    with ExitStack() as descendants:
        current = root
        for depth, raw_name in enumerate(names, start=1):
            name = os.path.basename(raw_name)
            if not name or name != raw_name or name in {".", ".."}:
                raise ValueError("invalid path component")
            if create:
                try:
                    dir_mkdir(current, name)
                except FileExistsError:
                    pass
                else:
                    if created is not None:
                        created.append(names[:depth])
            current = open_pinned_child(current, name)
            descendants.callback(current.close)
        yield current


def _create_project_temp(parent: PinnedDir, prefix: str, mode: int) -> tuple[str, int]:
    """Create a new, empty temporary entry in ``parent`` and open it for writing.

    Returns the entry's name and a descriptor the caller closes. ``O_EXCL``
    and ``O_NOFOLLOW`` make it a fresh file, never an existing one or a link.
    """
    temporary = f"{prefix}{secrets.token_hex(12)}.tmp"
    fd = dir_open(parent, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, mode)
    return temporary, fd


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``; a write that makes no progress is an error."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written == 0:
            raise OSError("project file write made no progress")
        remaining = remaining[written:]


def _write_bytes_at_project_root(
    root: PinnedDir, path: _ValidatedProjectPath, data: bytes
) -> os.stat_result:
    """Write a validated relative path below an already-pinned project root."""
    dirs = tuple(os.path.basename(part) for part in path.parts[:-1])
    name = os.path.basename(path.parts[-1])
    if dirs != path.parts[:-1] or name != path.parts[-1]:
        raise ValueError("invalid path component")

    with _opened_pinned_descendant(root, dirs, create=True) as parent:
        try:
            existing = dir_lstat(parent, name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise HTTPException(status_code=404, detail="File not found")
            if stat.S_ISDIR(existing.st_mode):
                raise HTTPException(status_code=400, detail="Path is a directory")
        # Replacing the entry detaches any pre-existing hardlink. Truncating
        # its inode could edit protected artifact bytes through an ordinary name.
        mode = stat.S_IMODE(existing.st_mode) if existing is not None else 0o644
        temporary, fd = _create_project_temp(parent, ".cowork-file-", mode)
        try:
            try:
                if existing is not None and hasattr(os, "fchmod"):
                    os.fchmod(fd, mode)
                _write_all(fd, data)
                result = os.fstat(fd)
            finally:
                os.close(fd)
            dir_replace(parent, temporary, name)
            return result
        finally:
            try:
                dir_unlink(parent, temporary)
            except FileNotFoundError:
                pass



_IMPORT_PREFIX = ".cowork-import-"
_IMPORT_MAX_DEPTH = 32
_IMPORT_STALE_SECONDS = 60 * 60
_IMPORT_TEMP_NAME_LENGTH = len(_IMPORT_PREFIX) + 24 + len(".tmp")
_WINDOWS_MAX_PATH = 259
_WINDOWS_MAX_FOLDER = 247
_MOVE_TRIES = 3
_MOVE_RETRY_SECONDS = 0.35


class _ImportWriteError(Exception):
    """An import could not change the project; ``path`` is relative to the project root."""

    def __init__(self, message: str, path: str) -> None:
        """Keep a user-safe message and the project-relative path it is about."""
        super().__init__(message)
        self.path = path


class _ImportConflict(_ImportWriteError):
    """A target can't be written as asked; found before any byte is written."""


class _ImportFileInUse(_ImportWriteError):
    """Windows refused to replace a file another program holds open."""


class _ImportDiskError(_ImportWriteError):
    """The project folder refused a write, a move or a folder."""


class _ImportRestoreFailed(_ImportWriteError):
    """A failed import could not put a file back; ``kept`` is where its old copy still is."""

    def __init__(self, message: str, path: str, kept: str | None) -> None:
        """Name the file that could not be restored and the old copy kept beside it."""
        super().__init__(message, path)
        self.kept = kept


@dataclass
class _ImportTarget:
    parts: tuple[str, ...]
    old_mode: int | None = None
    temporary: str | None = None
    backup: str | None = None
    state: str = "untouched"

    @property
    def path(self) -> str:
        """The target as a project-relative path."""
        return "/".join(self.parts)

    @property
    def folder(self) -> str:
        """The target's folder as a project-relative path, ``""`` for the root."""
        return "/".join(self.parts[:-1])


def _folded(name: str) -> str:
    """A name as a case-insensitive, Unicode-normalising filesystem compares it."""
    return unicodedata.normalize("NFC", name).casefold()


def _utf16_length(path: str) -> int:
    """A path's length as Windows counts it."""
    return len(path.encode("utf-16-le")) // 2


class _ProjectImport:
    """Write a set of files into a project so that all of them land or none do.

    ``check`` refuses every conflict before a byte is written. ``stage`` writes
    one file's bytes to a temporary entry beside its target. ``commit`` moves
    each old copy to a backup and each temporary onto its target, and only
    after all of them moved removes the backups. Leaving the ``with`` block
    with an exception undoes everything: files walked back in reverse, every
    temporary removed, folders this import created removed if empty. A backup
    is kept whenever its file could not be put back.

    An old copy is only ever an entry with exactly the target's name: a name
    the filesystem matches by folding case or Unicode form is a conflict, so
    no file is touched that the import was not asked to replace.
    """

    def __init__(
        self,
        root: PinnedDir,
        targets: list[tuple[str, ...]],
        *,
        windows: bool = os.name == "nt",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Hold the pinned project root and each target's path below it."""
        self._root = root
        self._targets = [_ImportTarget(tuple(parts)) for parts in targets]
        self._windows = windows
        self._clock = clock
        self._sleep = sleep
        self._created: list[tuple[str, ...]] = []
        self._swept: set[tuple[str, ...]] = set()

    def __enter__(self) -> "_ProjectImport":
        """Start an import; leaving with an exception rolls it back."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, _tb: object) -> bool:
        """On an exception, undo everything; a file that could not be put back replaces the error."""
        if exc is None:
            return False
        failure = self._roll_back()
        if failure is not None:
            raise failure
        return False

    def check(self) -> None:
        """Refuse any target that can't be written as asked, before anything is written."""
        files: set[tuple[str, ...]] = set()
        folders: dict[tuple[str, ...], tuple[str, ...]] = {}
        for target in self._targets:
            if not target.parts or len(target.parts) > _IMPORT_MAX_DEPTH:
                raise _ImportConflict(f"{target.path} is more than {_IMPORT_MAX_DEPTH} folders deep.", target.path)
            key = tuple(_folded(part) for part in target.parts)
            if key in files:
                raise _ImportConflict(f"Two picked files would both be saved as {target.path}.", target.path)
            files.add(key)
            for depth in range(1, len(key)):
                spelled = folders.setdefault(key[:depth], target.parts[:depth])
                if spelled != target.parts[:depth]:
                    raise _ImportConflict(
                        f"{'/'.join(target.parts[:depth])} and {'/'.join(spelled)} would be one folder "
                        "on some computers. Rename one of them.",
                        target.path,
                    )
        for target in self._targets:
            if tuple(_folded(part) for part in target.parts) in folders:
                raise _ImportConflict(f"{target.path} would be both a file and a folder.", target.path)
        if self._windows:
            self._check_windows_lengths()
        for target in self._targets:
            try:
                self._check_on_disk(target)
            except OSError as exc:
                raise _ImportDiskError(f"Couldn't read {target.path} in the project folder.", target.path) from exc

    def stage(self, index: int, feed: Callable[[Callable[[bytes], None]], object]) -> None:
        """Write one target's bytes to a new temporary entry beside it.

        ``feed`` is called with a sink and pushes the file's bytes into it.
        An ``OSError`` or ``ValueError``, from the disk or from ``feed``,
        becomes a disk error; any other error from ``feed`` propagates.
        """
        target = self._targets[index]
        try:
            with _opened_pinned_descendant(
                self._root, target.parts[:-1], create=True, created=self._created
            ) as parent:
                self._sweep(parent, target.parts[:-1])
                mode = target.old_mode if target.old_mode is not None else 0o644
                temporary, fd = _create_project_temp(parent, _IMPORT_PREFIX, mode)
                target.temporary = temporary
                try:
                    if target.old_mode is not None and hasattr(os, "fchmod"):
                        os.fchmod(fd, mode)
                    feed(lambda chunk: _write_all(fd, chunk))
                finally:
                    os.close(fd)
        except (OSError, ValueError) as exc:
            raise _ImportDiskError(f"Couldn't write {target.path} to the project folder.", target.path) from exc

    def commit(self) -> None:
        """Move every staged file into place, then drop the backups of the files it replaced."""
        for target in self._targets:
            try:
                with _opened_pinned_descendant(self._root, target.parts[:-1], create=False) as parent:
                    name = target.parts[-1]
                    if target.old_mode is not None:
                        backup = f"{_IMPORT_PREFIX}{secrets.token_hex(12)}.bak"
                        # A held old copy means the user has it open, so it is not retried.
                        dir_replace(parent, name, backup)
                        target.backup = backup
                        target.state = "backed_up"
                    self._retrying(lambda: dir_replace(parent, target.temporary, name))
                    target.temporary = None
                    target.state = "replaced"
            except (OSError, ValueError) as exc:
                if self._windows and isinstance(exc, PermissionError):
                    raise _ImportFileInUse(
                        f"{target.path} is open in another program. Close it and import again.", target.path
                    ) from exc
                raise _ImportDiskError(f"Couldn't move {target.path} into place.", target.path) from exc
        for target in self._targets:
            if target.backup is not None:
                self._discard(target, target.backup)
                target.backup = None

    def _check_windows_lengths(self) -> None:
        """Refuse a target whose path, temporary entry or new folder would pass Windows' limits."""
        base = str(self._root.path)
        for target in self._targets:
            folder = os.path.join(base, *target.parts[:-1])
            if (
                _utf16_length(os.path.join(base, *target.parts)) > _WINDOWS_MAX_PATH
                or _utf16_length(folder) + 1 + _IMPORT_TEMP_NAME_LENGTH > _WINDOWS_MAX_PATH
                or any(
                    _utf16_length(os.path.join(base, *target.parts[:depth])) > _WINDOWS_MAX_FOLDER
                    for depth in range(1, len(target.parts))
                )
            ):
                raise _ImportConflict(f"{target.path} would be too long a path for Windows.", target.path)

    def _check_on_disk(self, target: _ImportTarget) -> None:
        """Refuse links, folders and case or Unicode variants in the way; note an old copy's mode."""
        with ExitStack() as opened:
            current = self._root
            for depth, name in enumerate(target.parts, start=1):
                shown = "/".join(target.parts[:depth])
                if not name or os.path.basename(name) != name or name in {".", ".."}:
                    raise _ImportConflict(f"{shown} isn't a valid name.", target.path)
                try:
                    found = dir_lstat(current, name)
                except FileNotFoundError:
                    return
                if stat.S_ISLNK(found.st_mode):
                    raise _ImportConflict(f"{shown} is a link, so the import won't write through it.", target.path)
                try:
                    _existing_entry_name(current, name)
                except FileNotFoundError:
                    raise _ImportConflict(
                        f"{shown} matches an existing name that differs only in case or accents. Rename one of them.",
                        target.path,
                    ) from None
                if depth < len(target.parts):
                    if not stat.S_ISDIR(found.st_mode):
                        raise _ImportConflict(f"{shown} is a file, but the import needs a folder there.", target.path)
                    current = open_pinned_child(current, name)
                    opened.callback(current.close)
                elif stat.S_ISDIR(found.st_mode):
                    raise _ImportConflict(f"{shown} is a folder, so the file can't be saved there.", target.path)
                elif not stat.S_ISREG(found.st_mode):
                    raise _ImportConflict(f"{shown} isn't a regular file.", target.path)
                else:
                    target.old_mode = stat.S_IMODE(found.st_mode)

    def _sweep(self, parent: PinnedDir, folder: tuple[str, ...]) -> None:
        """Remove import temporaries a killed import left in this folder over an hour ago."""
        if folder in self._swept:
            return
        self._swept.add(folder)
        cutoff = self._clock() - _IMPORT_STALE_SECONDS
        with dir_scandir(parent) as entries:
            names = [e.name for e in entries if e.name.startswith(_IMPORT_PREFIX) and e.name.endswith(".tmp")]
        for name in names:
            try:
                found = dir_lstat(parent, name)
                if stat.S_ISREG(found.st_mode) and found.st_mtime < cutoff:
                    dir_unlink(parent, name)
            except OSError as exc:
                logger.warning("Could not remove a stale import temporary in %r: %s", "/".join(folder), exc)

    def _retrying(self, action: Callable[[], None]) -> None:
        """Run a move or removal, retrying briefly on Windows where a scanner may hold the file."""
        tries = _MOVE_TRIES if self._windows else 1
        for attempt in range(tries):
            try:
                action()
                return
            except PermissionError:
                if attempt == tries - 1:
                    raise
                self._sleep(_MOVE_RETRY_SECONDS)

    def _discard(self, target: _ImportTarget, name: str) -> None:
        """Remove a temporary or backup beside ``target``; a failure is logged, not raised."""
        try:
            with _opened_pinned_descendant(self._root, target.parts[:-1], create=False) as parent:
                self._retrying(lambda: dir_unlink(parent, name))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            logger.warning("Import could not remove %r beside %r: %s", name, target.path, exc)

    def _roll_back(self) -> _ImportRestoreFailed | None:
        """Undo every target in reverse, then remove the folders this import made; return the first restore failure."""
        failure: _ImportRestoreFailed | None = None
        for target in reversed(self._targets):
            failed = self._undo(target)
            if failure is None:
                failure = failed
        for parts in sorted(self._created, key=len, reverse=True):
            try:
                with _opened_pinned_descendant(self._root, parts[:-1], create=False) as parent:
                    dir_rmdir(parent, parts[-1])
            except FileNotFoundError:
                pass
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    logger.warning("Import could not remove the folder %r it created: %s", "/".join(parts), exc)
        return failure

    def _undo(self, target: _ImportTarget) -> _ImportRestoreFailed | None:
        """Put one target back as it was and remove its temporary; report a file that stays changed."""
        name = target.parts[-1]
        try:
            if target.state != "untouched":
                with _opened_pinned_descendant(self._root, target.parts[:-1], create=False) as parent:
                    if target.backup is not None:
                        backup = target.backup
                        self._retrying(lambda: dir_replace(parent, backup, name))
                        target.backup = None
                    else:
                        self._retrying(lambda: dir_unlink(parent, name))
                    target.state = "untouched"
        except (OSError, ValueError) as exc:
            if target.backup is not None:
                kept = "/".join((*target.parts[:-1], target.backup))
                failure = _ImportRestoreFailed(
                    f"The import failed and {target.path} couldn't be put back. "
                    f"Its old copy was kept as {kept}.",
                    target.path,
                    kept,
                )
            else:
                failure = _ImportRestoreFailed(
                    f"The import failed and couldn't remove {target.path}, which it had added.",
                    target.path,
                    None,
                )
            failure.__cause__ = exc
            return failure
        finally:
            if target.temporary is not None:
                self._discard(target, target.temporary)
                target.temporary = None
        return None


def _write_project_bytes(
    project_name: str,
    path: _ValidatedProjectPath,
    data: bytes,
    scoped: ScopedSession,
) -> os.stat_result:
    with _opened_selected_project_directory(project_name, scoped) as directory:
        return _write_bytes_at_project_root(directory, path, data)


def _delete_project_entry(
    project_name: str,
    path: _ValidatedProjectPath,
    scoped: ScopedSession,
) -> None:
    with _opened_selected_project_directory(project_name, scoped) as directory:
        with _opened_existing_project_entry(directory, path.parts) as (
            parent,
            disk_name,
        ):
            target_stat = dir_lstat(parent, disk_name)
            if stat.S_ISLNK(target_stat.st_mode):
                raise HTTPException(status_code=404, detail="File not found")
            if stat.S_ISDIR(target_stat.st_mode):
                raise HTTPException(status_code=400, detail="Path is a directory")
            dir_unlink(parent, disk_name)


def _read_project_bytes(
    base: Path,
    path: _ValidatedProjectPath,
) -> bytes | None:
    target = safe_join(base, *path.parts)
    try:
        cm, fd, st = _pinned_regular_file(target, base)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise
    try:
        if st.st_size > TEXT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large to edit")
        return b"".join(read_bounded(fd, st.st_size))
    finally:
        cm.__exit__(None, None, None)


def _restore_project_bytes(
    project_name: str,
    path: _ValidatedProjectPath,
    previous: bytes | None,
    scoped: ScopedSession,
) -> None:
    if previous is not None:
        _write_project_bytes(project_name, path, previous, scoped)
        return
    try:
        _delete_project_entry(project_name, path, scoped)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise


def _compensate_instruction_write(
    context: _InstructionsWriteContext,
    project_name: str,
    path: _ValidatedProjectPath,
    scoped: ScopedSession,
) -> None:
    # Roll back first. A failed audit commit leaves the session in a pending
    # rollback, and the restore below reaches ProjectService.list_projects() on
    # that same session, so every query would raise and the swallowed exception
    # would leave the instructions overwritten. skills.py rolls back the same
    # way before its restore.
    try:
        context.access.session.rollback()
    except Exception:
        logger.exception("Could not roll back the session before restoring bytes")
    try:
        _restore_project_bytes(project_name, path, context.previous, scoped)
    except Exception:
        logger.exception(
            "Could not restore project instructions after a failed mutation"
        )
    if context.claim is not None and context.claim_token is not None:
        try:
            context.access.release_claim(
                context.claim,
                claim_token=context.claim_token,
            )
        except Exception:
            logger.exception("Could not release a failed project-instruction claim")
    _close_instruction_context(context)


def _close_instruction_context(
    context: _InstructionsWriteContext | _InstructionsDeleteContext,
) -> None:
    if context.mutation_context is not None:
        context.mutation_context.__exit__(None, None, None)
    context.coordination_context.close()


def _pinned_stream(
    target: Path, base: Path, *, media_type: str, headers: dict, scope: TenantScope
) -> StreamingResponse:
    """Serve `target` from a pinned descriptor.

    A `FileResponse` takes a path and opens it after the handler returns, which
    is the one thing the pinning exists to avoid, so the bytes come off the
    descriptor instead. No Content-Length is declared: the file can change
    while it is read (an agent still writing it), and a length taken from
    `fstat` would then disagree with the bytes sent, which browsers report as
    `ERR_CONTENT_LENGTH_MISMATCH` (ENG-2950). Chunked transfer sends exactly
    what the descriptor yields. The `fstat` size still bounds the body, so a
    file that keeps growing cannot keep the response open.
    """
    cm, fd, st = _pinned_regular_file(target, base)
    resources = ExitStack()
    resources.push(cm)
    reader = read_bounded(fd, st.st_size)

    # Each read runs on the file store's threads, not on anyio's default
    # threads, which Starlette would iterate a sync generator on.
    async def _chunks():
        try:
            while (chunk := await run_efs(partial(next, reader, None), scope=scope)) is not None:
                yield chunk
        finally:
            await run_efs(resources.close, scope=scope, deadline=False)

    try:
        return PinnedStreamingResponse(
            _chunks(), resources=resources, media_type=media_type, headers=headers
        )
    except BaseException:
        # `_chunks` never ran, so neither it nor the response can close the descriptor.
        resources.close()
        raise


@router.get(
    "/{project_name}/instructions",
    response_model=ProjectInstructionsResponse,
    response_model_exclude_unset=True,
)
async def get_project_instructions(
    project_name: str,
    scope: ScopeDep,
    principal: Principal | None = Depends(get_principal),
):
    return await run_efs_unit(lambda session: _get_project_instructions(project_name, session, principal), scope=scope)


def _get_project_instructions(
    project_name: str,
    scoped: ScopedSession,
    principal: Principal | None = None,
):
    project = _project_for_name(project_name, scoped)
    base = _project_dir(project_name, scoped)
    p = _anton_md_path(base)
    rel = p.relative_to(base).as_posix()
    if p.is_file():
        try:
            st = p.stat()
        except OSError:
            file = {
                "path": rel,
                "name": ANTON_INSTRUCTIONS_FILENAME,
                "size": 0,
                "modified": None,
                "is_dir": False,
                "synthetic": True,
            }
        else:
            file = {
                "path": rel,
                "name": ANTON_INSTRUCTIONS_FILENAME,
                "size": st.st_size,
                "modified": st.st_mtime,
                "is_dir": False,
            }
    else:
        file = {
            "path": rel,
            "name": ANTON_INSTRUCTIONS_FILENAME,
            "size": 0,
            "modified": None,
            "is_dir": False,
            "synthetic": True,
        }
    file.update(
        _instructions_fields(
            project,
            scoped,
            principal,
            modified=file["modified"],
        )
    )
    return {"file": file}


@router.get(
    "/{project_name}/files",
    response_model=ProjectFileListResponse,
    response_model_exclude_unset=True,
)
async def list_project_files(
    project_name: str,
    scope: ScopeDep,
    principal: Principal | None = Depends(get_principal),
):
    return await run_efs_unit(lambda session: _list_project_files(project_name, session, principal), scope=scope)


def _list_project_files(
    project_name: str,
    scoped: ScopedSession,
    principal: Principal | None = None,
):
    project = _project_for_name(project_name, scoped)
    base = _project_dir(project_name, scoped)
    files: list[dict[str, Any]] = []
    _conv_cache: dict = {}
    budget = _WalkBudget()
    truncated = False
    for p in _iter_project_files(base, budget):
        meta = _file_meta(p, base)
        if not (meta and _conversation_workspace_ok(meta["path"], scoped, _conv_cache)):
            continue
        # Counted after the filters, not before: budgeting candidates let a
        # subtree the caller cannot see (another member's conversation
        # workspace) spend the whole listing on rows that are then dropped.
        if len(files) >= _MAX_LISTED_FILES:
            truncated = True
            break
        files.append(meta)
    truncated = truncated or budget.exhausted
    # The walk yields shallowest first, so the response is re-sorted by path.
    # Sibling prefixes order slightly differently from the old `sorted()` over
    # Path objects, which compared parts rather than the joined string.
    files.sort(key=lambda f: f["path"])

    anton_rel = _anton_md_path(base).relative_to(base).as_posix()
    # Resolved from disk, not from the capped walk: a truncated listing would
    # otherwise report a real anton.md as synthetic with size 0.
    if truncated and not any(f["path"] == anton_rel for f in files):
        instructions_meta = _file_meta(_anton_md_path(base), base)
        if instructions_meta:
            files.append(instructions_meta)
    if not any(f["path"] == anton_rel for f in files):
        files.insert(
            0,
            {
                "path": anton_rel,
                "name": ANTON_INSTRUCTIONS_FILENAME,
                "size": 0,
                "modified": None,
                "is_dir": False,
                "synthetic": True,
            },
        )
    else:
        files.sort(key=lambda f: (f["path"] != anton_rel, f["path"]))

    instructions = next(file for file in files if file["path"] == anton_rel)
    instructions.update(
        _instructions_fields(
            project,
            scoped,
            principal,
            modified=instructions["modified"],
        )
    )

    response: dict[str, Any] = {"files": files}
    if truncated:
        response["truncated"] = True
    return response


@router.get(
    "/{project_name}/files/{path:path}",
    response_model=ProjectFileReadResponse,
    response_model_exclude_unset=True,
)
async def read_project_file(
    project_name: str,
    path: str,
    scope: ScopeDep,
    principal: Principal | None = Depends(get_principal),
):
    return await run_efs_unit(lambda session: _read_project_file(project_name, path, session, principal), scope=scope)


def _read_project_file(
    project_name: str,
    path: str,
    scoped: ScopedSession,
    principal: Principal | None = None,
):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    _require_workspace_access(target, base, scoped)
    try:
        resolved_parts = target.relative_to(base.resolve()).parts
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    instructions_project = (
        _project_for_name(project_name, scoped)
        if resolved_parts == (".anton", ANTON_INSTRUCTIONS_FILENAME)
        else None
    )
    if not target.exists():
        if resolved_parts == (".anton", ANTON_INSTRUCTIONS_FILENAME):
            response = {"path": path, "content": "", "size": 0, "modified": None}
            if instructions_project is not None:
                response.update(
                    _instructions_fields(
                        instructions_project,
                        scoped,
                        principal,
                        modified=None,
                    )
                )
            return response
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
            raise HTTPException(
                status_code=415, detail="File is not valid UTF-8 text"
            ) from exc
    finally:
        cm.__exit__(None, None, None)
    response = {
        "path": path,
        "content": content,
        "size": st.st_size,
        "modified": st.st_mtime,
    }
    if instructions_project is not None:
        response.update(
            _instructions_fields(
                instructions_project,
                scoped,
                principal,
                modified=st.st_mtime,
            )
        )
    return response


@router.put(
    "/{project_name}/files/{path:path}",
    response_model=ProjectFileWriteResponse,
    dependencies=[Depends(_authorize_artifact_file_mutation)],
    response_model_exclude_unset=True,
)
async def write_project_file(
    project_name: str,
    path: ProjectMutationPathDep,
    req: _FileWriteRequest,
    scope: ScopeDep,
    principal: Principal | None = Depends(get_principal),
):
    return await run_efs_unit(lambda session: _write_project_file(project_name, path, req, session, principal), scope=scope)


def _write_project_file(
    project_name: str,
    path: ProjectMutationPathDep,
    req: _FileWriteRequest,
    scoped: ScopedSession,
    principal: Principal | None = None,
):
    _require_workspace_path(path, scoped)
    classification_base = (
        _project_dir(project_name, scoped) if scoped.scope.org_mode else None
    )
    _reject_generic_project_memory_mutation(
        path,
        scoped,
        base=classification_base,
    )
    is_instructions_path = _is_instructions_path(path, classification_base)
    body = req.content or ""
    encoded = body.encode("utf-8")
    if len(encoded) > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MiB cap")

    instructions_context: _InstructionsWriteContext | None = None
    if is_instructions_path:
        project, access, coordination_context = _require_instructions_change(
            project_name,
            scoped,
            principal,
        )
        claim = None
        claim_token = None
        mutation_context = None
        mutation_entered = False
        try:
            base = Path(project.path)
            if not _is_instructions_path(path, base):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Protected project path changed while the request waited",
                )
            previous = _read_project_bytes(base, path)
            existed = previous is not None
            key = project_resource_key(project.id)
            access.recover_stale_claim(
                PROJECT_INSTRUCTIONS,
                key,
                resource_exists=lambda: _read_project_bytes(base, path) is not None,
            )
            if access.claim_is_pending(PROJECT_INSTRUCTIONS, key):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Another write is establishing these project instructions",
                )
            if scoped.scope.org_mode and not existed:
                claim, claim_token = access.reserve_claim(
                    PROJECT_INSTRUCTIONS,
                    key,
                )
                if claim is None:
                    raise RuntimeError(
                        "Project-instruction ownership could not be reserved"
                    )
                if claim_token is None and claim.pending_claim_token:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Another write is establishing these project instructions",
                    )
            if scoped.scope.org_mode and claim_token is None:
                mutation_context = access.mutation_lock(
                    PROJECT_INSTRUCTIONS,
                    key,
                    resource_exists=lambda: _read_project_bytes(base, path)
                    is not None,
                )
                mutation_context.__enter__()
                mutation_entered = True
                previous = _read_project_bytes(base, path)
                existed = previous is not None
        except Exception:
            if mutation_entered and mutation_context is not None:
                mutation_context.__exit__(None, None, None)
            if claim is not None and claim_token is not None:
                access.session.rollback()
                access.release_claim(claim, claim_token=claim_token)
            coordination_context.close()
            raise
        instructions_context = _InstructionsWriteContext(
            project=project,
            access=access,
            existed=existed,
            previous=previous,
            claim=claim,
            claim_token=claim_token,
            mutation_context=mutation_context,
            coordination_context=coordination_context,
        )
    try:
        # The selector only chooses an already-pinned scoped project handle.
        st = _write_project_bytes(project_name, path, encoded, scoped)
    except (OSError, ValueError):
        if instructions_context is not None:
            _compensate_instruction_write(
                instructions_context,
                project_name,
                path,
                scoped,
            )
        raise HTTPException(status_code=404, detail="File not found")
    except Exception:
        if instructions_context is not None:
            _compensate_instruction_write(
                instructions_context,
                project_name,
                path,
                scoped,
            )
        raise
    response = {"path": path.value, "size": st.st_size, "modified": st.st_mtime}
    if instructions_context is not None:
        project = instructions_context.project
        access = instructions_context.access
        existed = instructions_context.existed
        key = project_resource_key(project.id)
        action = "clear" if not body.strip() else "update"
        try:
            if (
                instructions_context.claim is not None
                and instructions_context.claim_token is not None
            ):
                finalized = access.finalize_claim(
                    instructions_context.claim,
                    instructions_context.claim_token,
                    action="create" if body.strip() else "clear",
                )
                if finalized is None:
                    raise RuntimeError(
                        "Project-instruction claim changed before it could be finalized"
                    )
            elif access.has_attribution(PROJECT_INSTRUCTIONS, key) or existed:
                # Existing attributed and legacy files both retain their
                # original creator semantics while recording this editor.
                access.record_update(
                    PROJECT_INSTRUCTIONS,
                    key,
                    action=action,
                )
        except Exception:
            _compensate_instruction_write(
                instructions_context,
                project_name,
                path,
                scoped,
            )
            raise
        try:
            response.update(
                _instructions_fields_from_access(
                    project,
                    access,
                    modified=st.st_mtime,
                )
            )
        finally:
            _close_instruction_context(instructions_context)
    return response


@dataclass(frozen=True)
class _Upload:
    """One uploaded file as the request read it: its bytes, or why they could
    not be read."""

    filename: str | None
    data: bytes | None
    read_error: Exception | None = None


@router.post("/{project_name}/files/upload")
async def upload_project_files(
    project_name: str,
    scope: ScopeDep,
    files: list[UploadFile] = File(...),
):
    # The request body is read here, from the upload's own spool; the project
    # lookup and the writes run together on the file store's threads.
    uploads: list[_Upload] = []
    for f in files:
        try:
            uploads.append(_Upload(filename=f.filename, data=await f.read()))
        except Exception as exc:
            uploads.append(_Upload(filename=f.filename, data=None, read_error=exc))
    return await run_efs_unit(partial(_write_uploads, project_name=project_name, uploads=uploads), scope=scope)


def _write_uploads(scoped: ScopedSession, *, project_name: str, uploads: list[_Upload]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with _opened_project_directory_inventory(scoped) as inventory:
        selected_project_name = os.path.basename(project_name)
        if (
            not selected_project_name
            or selected_project_name != project_name
            or selected_project_name in {".", ".."}
            or "\\" in selected_project_name
            or "\x00" in selected_project_name
        ):
            raise HTTPException(status_code=404, detail="Project not found")
        directory = next(
            (
                opened
                for server_name, opened in inventory
                if secrets.compare_digest(server_name, selected_project_name)
            ),
            None,
        )
        if directory is None:
            raise HTTPException(status_code=404, detail="Project not found")

        for upload in uploads:
            if not upload.filename:
                results.append({"name": "", "ok": False, "error": "filename missing"})
                continue
            safe_name = os.path.basename(upload.filename).strip()
            if not safe_name or safe_name.startswith("."):
                results.append(
                    {"name": upload.filename, "ok": False, "error": "invalid filename"}
                )
                continue
            try:
                if upload.read_error is not None:
                    raise upload.read_error
                _write_bytes_at_project_root(
                    directory, _ValidatedProjectPath((safe_name,)), upload.data
                )
                results.append({"name": safe_name, "ok": True, "size": len(upload.data)})
            except Exception as exc:
                logger.error("Failed to write file %s: %s", safe_name, exc)
                results.append(
                    {"name": safe_name, "ok": False, "error": "File write failed"}
                )
    return {"results": results}


_SHAREPOINT_IMPORT_FIELDS = frozenset({"connection", "site", "drive_id", "folder", "item_ids"})
_RESERVED_IMPORT_FOLDERS = frozenset({CONVERSATIONS_DIRNAME, "skills"})
_IMPORTS_RUNNING: set[UUID] = set()
_IMPORTS_LOCK = threading.Lock()
_SHAREPOINT_IMPORT_STATUS = {
    SharePointErrorKind.INVALID: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.TENANT_NOT_FOUND: status.HTTP_409_CONFLICT,
    SharePointErrorKind.APP_NOT_FOUND: status.HTTP_409_CONFLICT,
    SharePointErrorKind.KEY_REJECTED: status.HTTP_409_CONFLICT,
    SharePointErrorKind.SITE_NOT_GRANTED: status.HTTP_404_NOT_FOUND,
    SharePointErrorKind.ITEM_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    SharePointErrorKind.THROTTLED: status.HTTP_429_TOO_MANY_REQUESTS,
    SharePointErrorKind.UNREACHABLE: status.HTTP_502_BAD_GATEWAY,
    SharePointErrorKind.TOO_MANY_FILES: status.HTTP_413_CONTENT_TOO_LARGE,
    SharePointErrorKind.TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    SharePointErrorKind.BAD_NAME: status.HTTP_422_UNPROCESSABLE_CONTENT,
    SharePointErrorKind.PLAN_TOO_SLOW: status.HTTP_504_GATEWAY_TIMEOUT,
    SharePointErrorKind.IMPORT_TOO_SLOW: status.HTTP_504_GATEWAY_TIMEOUT,
}
_IMPORT_WRITE_STATUS: dict[type[_ImportWriteError], int] = {
    _ImportConflict: status.HTTP_409_CONFLICT,
    _ImportFileInUse: status.HTTP_409_CONFLICT,
    _ImportDiskError: status.HTTP_507_INSUFFICIENT_STORAGE,
    _ImportRestoreFailed: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


@dataclass(frozen=True)
class _SharePointImportRequest:
    connection: str
    site: str
    drive_id: str
    folder: str
    item_ids: list[str]


def _sharepoint_import_request(body: Any) -> _SharePointImportRequest:
    """Validate the import body: exactly its fields, strings, and 1 to 100 distinct item ids.

    Every refusal is a fixed message, never an echo of the body.
    """
    invalid = HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Send connection, site, drive_id, folder and item_ids (a list of 1 to 100 ids).",
    )
    if not isinstance(body, dict) or set(body) != _SHAREPOINT_IMPORT_FIELDS:
        raise invalid
    text = {name: body[name] for name in ("connection", "site", "drive_id", "folder")}
    item_ids = body["item_ids"]
    if not all(isinstance(value, str) for value in text.values()):
        raise invalid
    if not isinstance(item_ids, list) or not all(isinstance(value, str) for value in item_ids):
        raise invalid
    distinct = list(dict.fromkeys(item_ids))
    if not 1 <= len(distinct) <= 100:
        raise invalid
    return _SharePointImportRequest(item_ids=distinct, **text)


def _reserved_import_folder(name: str) -> bool:
    """Whether a top-level project folder is Cowork's own and closed to imports."""
    return name.casefold() in _RESERVED_IMPORT_FOLDERS


def _import_folder_parts(folder: str, scoped: ScopedSession) -> tuple[str, ...]:
    """The destination folder as path parts; ``""`` is the project root."""
    if folder == "":
        return ()
    path = _validated_project_path(folder)
    if any(part.startswith(".") for part in path.parts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid path")
    if _reserved_import_folder(path.parts[0]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This folder is reserved for Cowork.")
    _require_workspace_path(path, scoped)
    return path.parts


def _detached_root(directory: PinnedDir) -> PinnedDir:
    """A second handle on a pinned project root that outlives the inventory that opened it."""
    if directory.fd is None:
        return PinnedDir(None, directory.path)
    return PinnedDir(os.dup(directory.fd), directory.path)


def _sharepoint_import_error(exc: SharePointError) -> HTTPException:
    """The import response for a SharePoint failure; a stored key that stopped working is one message."""
    detail = str(exc)
    if exc.kind in {
        SharePointErrorKind.TENANT_NOT_FOUND,
        SharePointErrorKind.APP_NOT_FOUND,
        SharePointErrorKind.KEY_REJECTED,
    }:
        detail = _KEY_NO_LONGER_WORKS
    return HTTPException(status_code=_SHAREPOINT_IMPORT_STATUS[exc.kind], detail=detail)


@contextmanager
def _project_import_lock(project_id: UUID) -> Iterator[None]:
    """Let one import at a time run into a project; a second one is refused, not queued."""
    with _IMPORTS_LOCK:
        if project_id in _IMPORTS_RUNNING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another import into this project is still running.",
            )
        _IMPORTS_RUNNING.add(project_id)
    try:
        yield
    finally:
        with _IMPORTS_LOCK:
            _IMPORTS_RUNNING.discard(project_id)


@router.post(
    "/{project_name}/files/import/sharepoint",
    dependencies=[Depends(require(DesktopOnly))],
    response_model=SharePointImportResponse,
)
def import_sharepoint_files(
    project_name: str,
    body: Annotated[Any, Body()],
    scoped: ScopedSessionDep,
) -> SharePointImportResponse:
    """Copy picked files and folders from a SharePoint site into a project.

    Every check that needs the database runs first, then the session is
    closed: the walk and the downloads can take minutes. Files land together
    or not at all (``_ProjectImport``). Desktop only.
    """
    request = _sharepoint_import_request(body)
    key, sites = _stored_connection(scoped.scope, request.connection)
    site_url = _granted_site(request.site, sites)
    folder = _import_folder_parts(request.folder, scoped)
    project = _project_for_name(project_name, scoped)
    with _project_import_lock(project.id), ExitStack() as held:
        with _opened_selected_project_directory(project_name, scoped) as directory:
            root = _detached_root(directory)
        held.callback(root.close)
        scoped.close()
        try:
            with SharePointClient(key) as client:
                deadline = client.import_deadline()
                plan = client.plan_import(site_url, request.drive_id, request.item_ids)
                if not folder and any(_reserved_import_folder(f.parts[0]) for f in plan.files):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, detail="This folder is reserved for Cowork."
                    )
                budget = DownloadBudget()
                sizes: list[int] = []
                with _ProjectImport(root, [(*folder, *f.parts) for f in plan.files]) as writer:
                    writer.check()
                    for index, planned in enumerate(plan.files):
                        writer.stage(
                            index,
                            lambda sink, item_id=planned.item_id: sizes.append(
                                client.download(plan.site, plan.drive_id, item_id, sink, budget, deadline)
                            ),
                        )
                    writer.commit()
        except SharePointError as exc:
            raise _sharepoint_import_error(exc) from exc
        except _ImportWriteError as exc:
            raise HTTPException(status_code=_IMPORT_WRITE_STATUS[type(exc)], detail=str(exc)) from exc
    return SharePointImportResponse(
        imported=[
            SharePointImportedFile(path="/".join((*folder, *planned.parts)), size=size)
            for planned, size in zip(plan.files, sizes, strict=True)
        ],
        skipped=[SharePointSkippedItem(path=item.path, reason=item.reason) for item in plan.skipped],
    )


@router.delete(
    "/{project_name}/files/{path:path}",
    response_model=ProjectFileDeleteResponse,
    dependencies=[Depends(_authorize_artifact_file_mutation)],
)
async def delete_project_file(
    project_name: str,
    path: ProjectMutationPathDep,
    scope: ScopeDep,
    principal: Principal | None = Depends(get_principal),
):
    return await run_efs_unit(lambda session: _delete_project_file(project_name, path, session, principal), scope=scope)


def _delete_project_file(
    project_name: str,
    path: ProjectMutationPathDep,
    scoped: ScopedSession,
    principal: Principal | None = None,
):
    _require_workspace_path(path, scoped)
    classification_base = (
        _project_dir(project_name, scoped) if scoped.scope.org_mode else None
    )
    _reject_generic_project_memory_mutation(
        path,
        scoped,
        base=classification_base,
    )
    is_instructions_path = _is_instructions_path(path, classification_base)
    instructions_context: _InstructionsDeleteContext | None = None
    if is_instructions_path:
        project, access, coordination_context = _require_instructions_change(
            project_name,
            scoped,
            principal,
        )
        mutation_context = None
        mutation_entered = False
        try:
            base = Path(project.path)
            if not _is_instructions_path(path, base):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Protected project path changed while the request waited",
                )
            key = project_resource_key(project.id)
            previous = _read_project_bytes(base, path)
            access.recover_stale_claim(
                PROJECT_INSTRUCTIONS,
                key,
                resource_exists=lambda: _read_project_bytes(base, path) is not None,
            )
            if access.claim_is_pending(PROJECT_INSTRUCTIONS, key):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Another write is establishing these project instructions",
                )
            mutation_context = access.mutation_lock(
                PROJECT_INSTRUCTIONS,
                key,
                resource_exists=lambda: _read_project_bytes(base, path) is not None,
            )
            mutation_context.__enter__()
            mutation_entered = True
            previous = _read_project_bytes(base, path)
        except Exception:
            if mutation_entered and mutation_context is not None:
                mutation_context.__exit__(None, None, None)
            coordination_context.close()
            raise
        if mutation_context is None:
            coordination_context.close()
            raise RuntimeError("Project-instruction mutation lock was not established")
        instructions_context = _InstructionsDeleteContext(
            project=project,
            access=access,
            previous=previous,
            mutation_context=mutation_context,
            coordination_context=coordination_context,
        )
    try:
        _delete_project_entry(project_name, path, scoped)
    except HTTPException:
        if instructions_context is not None:
            _close_instruction_context(instructions_context)
        raise
    except (OSError, ValueError):
        if instructions_context is not None:
            _close_instruction_context(instructions_context)
        raise HTTPException(status_code=404, detail="File not found")
    except Exception:
        if instructions_context is not None:
            _close_instruction_context(instructions_context)
        raise
    if instructions_context is not None:
        project = instructions_context.project
        access = instructions_context.access
        try:
            access.record_delete(
                PROJECT_INSTRUCTIONS,
                project_resource_key(project.id),
            )
        except Exception:
            # Same ordering as _compensate_instruction_write: the failed audit
            # commit poisons the session the restore has to read through.
            try:
                access.session.rollback()
            except Exception:
                logger.exception(
                    "Could not roll back the session before restoring bytes"
                )
            try:
                _restore_project_bytes(
                    project_name,
                    path,
                    instructions_context.previous,
                    scoped,
                )
            except Exception:
                logger.exception(
                    "Could not restore project instructions after a failed delete audit"
                )
            _close_instruction_context(instructions_context)
            raise
        _close_instruction_context(instructions_context)
    return {"status": "deleted", "path": path.value}


@router.delete(
    "/{project_name}/skill_drafts/{slug}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_skill_draft(
    project_name: str,
    slug: str,
    scope: ScopeDep,
):
    return await run_efs_unit(lambda session: _delete_skill_draft(project_name, slug, session), scope=scope)


def _delete_skill_draft(
    project_name: str,
    slug: str,
    scoped: ScopedSession,
):
    """Remove a staged skill draft once it is Saved (or dismissed).

    Idempotent: a missing draft is a no-op — Save may race the sweep, and a
    lingering draft is the safe default we're clearing, not a hard error. The
    slug is confined to a direct child of the drafts dir (no traversal).
    """
    safe_slug = os.path.basename(slug)
    if (
        safe_slug != slug
        or safe_slug in {"", ".", ".."}
        or "\\" in safe_slug
        or "\0" in safe_slug
    ):
        raise HTTPException(status_code=400, detail="invalid slug")
    try:
        with _opened_selected_project_directory(project_name, scoped) as project:
            with _opened_pinned_descendant(
                project,
                (".anton", "skill_drafts"),
                create=False,
            ) as drafts:
                disk_name = _existing_entry_name(drafts, safe_slug)
                entry = dir_lstat(drafts, disk_name)
                if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                    return
                dir_rmtree(drafts, disk_name)
    except HTTPException:
        raise
    except (OSError, ValueError):
        # Idempotent by contract: missing drafts, non-directories, planted
        # symlinks, and best-effort cleanup failures are all no-ops.
        return


@router.post("/preview-mount-file")
async def preview_mount_file(
    req: _PreviewMountRequest,
    scope: ScopeDep,
):
    return await run_efs_unit(lambda session: _preview_mount_file(req, session), scope=scope)


def _preview_mount_file(
    req: _PreviewMountRequest,
    scoped: ScopedSession,
):
    base = _project_dir(req.name, scoped)
    target = _safe_relpath(req.path, base)
    _require_workspace_access(target, base, scoped)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() not in (".html", ".htm"):
        raise HTTPException(
            status_code=415, detail="Preview mount is only available for HTML files"
        )
    token = _register_preview_mount(target, base, scoped.scope)
    return {
        "token": token,
        "entry": target.name,
        "relUrl": f"/projects/preview-asset/{token}/{target.name}",
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
async def preview_asset(
    token: str, rel_path: str, scope: TenantScope = Depends(get_tenant_scope)
):
    return await run_efs(partial(_preview_asset, token, rel_path, scope), scope=scope)


def _preview_asset(token: str, rel_path: str, scope: TenantScope):
    # An unknown token, an expired one and another member's one all answer the
    # same 404: which of the three it was is itself information about somebody
    # else's files.
    mount = _PROJECT_PREVIEW_MOUNTS.get(token)
    if mount is None or mount.expires_at <= time.time() or not mount.readable_by(scope):
        raise HTTPException(
            status_code=404, detail="Preview mount has expired or is unknown"
        )
    parent = mount.parent
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(
            status_code=403, detail="Asset is outside the mounted directory"
        )
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
        scope=scope,
    )


@router.get("/{project_name}/files-raw/{path:path}")
async def download_project_file(
    project_name: str,
    path: str,
    scope: ScopeDep,
):
    return await run_efs_unit(lambda session: _download_project_file(project_name, path, session), scope=scope)


def _download_project_file(
    project_name: str,
    path: str,
    scoped: ScopedSession,
):
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
        headers={"Content-Disposition": attachment_disposition(target.name)},
        scope=scoped.scope,
    )
