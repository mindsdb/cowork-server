"""Authenticated artifact draft editing, revisions, review access and repair routes."""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import ntpath
import os
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from cowork.api.v1.artifact_preview import wants_comment_layer
from cowork.common.paths import (
    O_NOFOLLOW,
    dir_lstat,
    dir_open,
    dir_scandir,
    open_pinned_child,
)
from cowork.api.v1.artifact_scope import review_artifact_for_request
from cowork.db.scoped import ScopedSessionDep
from cowork.services.product_permissions import has_product_permission, require_product_permission
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    artifact_owner_id,
    require_artifact_owner,
)
from cowork.services.comments_layer import inject_layer
from cowork.services.artifact_identity import opened_artifact_folder
from cowork.services.artifact_revisions import (
    JOURNAL_DIRNAME,
    RepairAlreadyPending,
    RevisionConflict,
    RevisionValidationError,
    active_agent_repair,
    agent_repair_detail,
    cancel_agent_repair,
    create_agent_repair,
    current_source,
    current_workspace,
    finalize_agent_repair,
    list_revisions,
    release_repairs_for_comment,
    revision_with_content,
    save_source,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_DRAFT_RESPONSE_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}
_LIVE_PUBLISH_TIMEOUT_S = 60.0
_LIVE_PUBLISH_LOCK_TTL_S = _LIVE_PUBLISH_TIMEOUT_S * 3
_PRIVATE_DRAFT_ENTRIES = {
    ".revisions",
    ".published.json",
    "metadata.json",
    "README.md",
    "backend.log",
}


def _attachment_disposition(filename: str) -> str:
    """``Content-Disposition`` for saving ``filename``, safe to put on the wire.

    The name is a request-derived path component. ``_relative_file_parts``
    rejects NUL, separators and ``..``, but a quote, CR or LF still pass — and
    any of them interpolated raw into a header is a header-injection vector
    (``project_files.download_project_file`` has exactly that shape; ENG-2044).

    Two spellings so every client gets a usable name: an ASCII ``filename=``
    with quotes and backslashes escaped and non-printables dropped, and an
    RFC 5987 ``filename*=UTF-8''…`` carrying the exact name percent-encoded,
    which is what current browsers read first.
    """
    cleaned = "".join(ch for ch in filename if ch.isprintable())
    ascii_name = cleaned.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("\\", "\\\\").replace('"', '\\"').strip() or "download"
    # A fully non-ASCII stem leaves only the extension(s) behind — "报告.xlsx"
    # -> ".xlsx", "报告.tar.gz" -> ".tar.gz" — a HIDDEN dotfile on macOS/Linux.
    # Prefix rather than rpartition: splitting on the last dot rescued only
    # single-extension names (review pass 2 on #413). Distinct artifacts can
    # still degrade to the same "download.xlsx" — acceptable, since the ASCII
    # spelling is a fallback and `filename*` below carries the exact name.
    if ascii_name.startswith("."):
        ascii_name = "download" + ascii_name
    return (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(cleaned or 'download', safe='')}"
    )


def _relative_file_parts(value: str) -> tuple[str, ...]:
    """Split an untrusted relative path into safe, single components.

    The returned strings are passed to descriptor-relative opens; no
    request-derived string is ever joined onto an absolute filesystem path.
    Treat backslashes as separators too so the validation has the same meaning
    on the Windows desktop and the Linux service.
    """
    if not value or "\x00" in value:
        raise ValueError("invalid path")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ntpath.splitdrive(normalized)[0]:
        raise ValueError("invalid path")
    parts = tuple(normalized.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid path")
    return parts


def _artifact_folder_component(source, folder: Path) -> str:
    """Return the direct child of ``source.base`` selected by resolution.

    Identity lookup indexes a resolved artifacts root, so a normal result can
    be parented by either the declared root or its resolved spelling. Anything
    else must not be translated to ``folder.name``: doing so could turn a grant
    for one resolved folder into a same-named folder under another source.
    """
    base = Path(source.base)
    try:
        parents = {base, base.resolve(strict=False)}
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        ) from exc
    if folder.parent not in parents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        )
    name = folder.name
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        )
    return name


def _existing_draft_entry_name(
    directory, requested: str, *, native_case: bool = False
) -> str:
    """Return the pinned directory's own name for a request selector.

    Validation makes ``requested`` a single component, but it still originated
    in an HTTP path.  Compare it with entries already discovered below the
    pinned directory and return ``DirEntry.name`` so no request-derived string
    is ever supplied to ``openat``.  A replacement after the scan remains safe:
    the subsequent descriptor-relative open uses ``O_NOFOLLOW``.

    ``native_case`` also accepts a spelling the filesystem itself accepts. It
    exists because ``resolve_source`` opens by path and so inherits the
    volume's own case rules, and a boundary stricter than the gate behind it
    makes a reported source unsaveable. The filesystem decides, never the
    platform: the requested spelling has to name the same inode as the entry,
    which on a case-sensitive volume it cannot. The name returned is still the
    one the scan produced.
    """
    insensitive_matches: list[str] = []
    with dir_scandir(directory) as entries:
        for entry in entries:
            if entry.name == requested:
                return entry.name
            if native_case and entry.name.lower() == requested.lower():
                insensitive_matches.append(entry.name)
    # Ambiguous only on a case-sensitive volume, where the exact match above
    # is the only correct answer anyway.
    if len(insensitive_matches) == 1:
        if _names_one_inode(directory, requested, insensitive_matches[0]):
            return insensitive_matches[0]
    raise FileNotFoundError(requested)


def _names_one_inode(directory, requested: str, discovered: str) -> bool:
    """Whether the filesystem resolves both spellings to the same file.

    Neither stat follows a link, so a symlink whose name differs only in case
    from a real entry compares unequal and is refused rather than matched.
    """
    try:
        probe = dir_lstat(directory, requested)
        found = dir_lstat(directory, discovered)
    except OSError:
        return False
    return (probe.st_dev, probe.st_ino) == (found.st_dev, found.st_ino)


def _open_pinned_draft_file(source, folder: Path, parts: tuple[str, ...]):
    """Open one regular draft file without following any writable symlink.

    Authorization chose ``source`` and ``folder`` before this call. The source
    retains its server-owned anchor, and ``opened_artifact_folder`` walks from
    that anchor to the artifact with ``O_NOFOLLOW`` on every component. The
    request path is then walked the same way. Returning the ``ExitStack`` keeps
    every descriptor alive until the response has consumed the final file.
    """
    folder_name = _artifact_folder_component(source, folder)
    resources = ExitStack()
    try:
        current = resources.enter_context(opened_artifact_folder(source, folder_name))
        for requested in parts[:-1]:
            disk_name = _existing_draft_entry_name(current, requested)
            current = open_pinned_child(current, disk_name)
            resources.callback(current.close)
        disk_name = _existing_draft_entry_name(current, parts[-1])
        fd = dir_open(current, disk_name, os.O_RDONLY | O_NOFOLLOW)
        resources.callback(os.close, fd)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("draft target is not a regular file")
    except (OSError, ValueError) as exc:
        resources.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        ) from exc
    return resources, fd, file_stat


def _editable_source_selector(source, folder: Path, requested: str | None) -> str | None:
    """Translate a request-supplied source path into the folder's own spelling.

    ``None`` (or blank) leaves the choice to the service, which falls back to
    ``metadata["primary"]``. That is not a trusted value either: it is read
    from `metadata.json` inside the artifact folder, which is pod-writable on
    shared storage, so what makes the fallback safe is the inner gate's own
    containment, symlink and extension checks rather than where it came from.
    Anything else is the same kind of request-derived string
    ``_open_pinned_draft_file`` refuses to
    hand to the filesystem: it is validated into single components, each one
    is matched against a ``dir_scandir`` pass on a pinned descriptor, and the
    path the revision service receives is joined from the ``DirEntry`` names
    the OS returned — never from the HTTP string. A symlink on any component
    is refused rather than resolved, which the inner gate does not do for an
    intermediate directory: it resolves those and only checks containment, so
    a link inside the folder pointing back into the folder was accepted.
    This leaves ``resolve_source``'s own resolve-then-read window exactly as
    it was, neither narrowed nor closed. That pair is untouched here, and
    closing it means reading from a descriptor rather than a path, in the
    service.
    ``resolve_source`` keeps its own containment and extension checks as the
    inner gate; this is the outer one, at the request boundary, and it is what
    keeps ``?path=`` out of ``pathlib`` in the service layer altogether.
    """
    if requested is None or not requested.strip():
        return None
    try:
        parts = _relative_file_parts(requested.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid artifact source path") from exc
    # The journal only, matching the inner gate, and at any depth rather than
    # just the first component. The private-listing set is a different
    # question: it hides README.md, which is a source the service itself picks.
    if JOURNAL_DIRNAME in parts:
        raise HTTPException(status_code=422, detail="Invalid artifact source path")
    folder_name = _artifact_folder_component(source, folder)
    disk_parts: list[str] = []
    try:
        with ExitStack() as resources:
            current = resources.enter_context(opened_artifact_folder(source, folder_name))
            for part in parts[:-1]:
                disk_name = _existing_draft_entry_name(current, part, native_case=True)
                current = open_pinned_child(current, disk_name)
                resources.callback(current.close)
                disk_parts.append(disk_name)
            disk_name = _existing_draft_entry_name(current, parts[-1], native_case=True)
            if not stat.S_ISREG(dir_lstat(current, disk_name).st_mode):
                raise OSError("draft source is not a regular file")
            disk_parts.append(disk_name)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact source not found",
        ) from exc
    return "/".join(disk_parts)


def _recorded_source_selector(source, folder: Path, metadata: dict) -> str | None:
    """The folder's own spelling of the source `metadata` records.

    Without this the two routes disagree: the request path is translated to
    the disk spelling while an absent one leaves the service reporting
    `metadata["primary"]` verbatim, and the revision journal is keyed by
    whichever string arrived. A read under one spelling and a write under the
    other then answer 409 rather than the 404 a case-exact boundary gave.

    A primary that resolves to nothing yields ``None``, which leaves the
    service to answer for it exactly as it did before this indirection: it
    reads the same recorded value and refuses it. Note that is a refusal, not
    a fallback -- the service only picks a file itself when the primary is
    *empty*, and a set-but-absent one raises. Returning ``None`` here is
    therefore about not adding a second, earlier failure for the same cause,
    not about rescuing stale metadata.
    """
    recorded = metadata.get("primary")
    if not isinstance(recorded, str) or not recorded.strip():
        return None
    try:
        return _editable_source_selector(source, folder, recorded)
    except HTTPException:
        return None


def _comment_layer_from_fd(fd: int) -> HTMLResponse | None:
    """Build the review HTML from the already-authorized file descriptor."""
    payload = bytearray()
    try:
        while chunk := os.read(fd, 1 << 16):
            payload.extend(chunk)
        html = bytes(payload).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        # A non-UTF8 document falls back to the ordinary byte stream, which
        # must start at byte zero. Regular files are seekable; if the descriptor
        # itself failed, the eventual stream will fail rather than reopen a path.
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            pass
    return HTMLResponse(inject_layer(html), headers=_DRAFT_RESPONSE_HEADERS)


def _draft_stream(
    resources: ExitStack,
    fd: int,
    size: int,
    media_type: str,
    *,
    extra_headers: dict[str, str] | None = None,
):
    """Stream bytes from the pinned descriptor and close every held handle."""
    def chunks():
        try:
            while chunk := os.read(fd, 1 << 16):
                yield chunk
        finally:
            resources.close()

    return _PinnedDraftResponse(
        chunks(),
        resources=resources,
        media_type=media_type,
        headers={
            **_DRAFT_RESPONSE_HEADERS,
            **(extra_headers or {}),
            "Content-Length": str(size),
        },
    )


class _PinnedDraftResponse(StreamingResponse):
    """Always release draft descriptors, including before iteration starts."""

    def __init__(self, *args, resources: ExitStack, **kwargs):
        self._resources = resources
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # StreamingResponse does not run a BackgroundTask when the ASGI
            # send callable raises before body iteration. This outer finally
            # covers that disconnect path; ExitStack.close is idempotent with
            # the generator's normal cleanup.
            self._resources.close()


def _artifact_id_from_path(artifact_id: UUID) -> str:
    """The identity gate for every route in this file.

    Declaring the path parameter as a `UUID` makes FastAPI answer 422 before any
    handler body runs, so a request-supplied string never reaches the identity
    resolver, the revision journal or the filesystem. `resolve_artifact_folder`
    normalizes again and the index only ever yields folders it walked itself, so
    this is the outer of several gates rather than the only one — but it is the
    one that keeps unvalidated input out of the service layer entirely.

    Returns the canonical 32-hex spelling, which is what metadata carries and
    what the responses echo, so both the dashed and undashed URL forms resolve
    to one identity.
    """
    return artifact_id.hex


#: Path-validated artifact identity. Routes keep `{artifact_id}` in their path;
#: the dependency claims that parameter and hands the handler a canonical id.
ArtifactIdDep = Annotated[str, Depends(_artifact_id_from_path)]


class _SourceUpdateBody(BaseModel):
    content: str
    expectedRevisionId: str = Field(min_length=1, max_length=80)
    path: str | None = Field(default=None, max_length=1000)
    summary: str = Field(default="Edited artifact", max_length=240)


class _RestoreBody(BaseModel):
    expectedRevisionId: str = Field(min_length=1, max_length=80)


class _AgentRepairAuthor(BaseModel):
    user_id: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=320)


class _AgentRepairThreadEntry(BaseModel):
    author: _AgentRepairAuthor | None = None
    text: str = Field(max_length=10_000)
    createdAt: str | None = Field(default=None, max_length=100)


class _AgentRepairBody(BaseModel):
    expectedRevisionId: str = Field(min_length=1, max_length=80)
    commentThreadId: str = Field(min_length=1, max_length=100)
    selector: str | None = Field(default=None, max_length=2000)
    thread: list[_AgentRepairThreadEntry] = Field(min_length=1, max_length=501)
    conversationId: UUID


class _RepairDecisionBody(BaseModel):
    status: Literal["accepted", "rejected"]
    # The head the user confirmed against. Rejecting restores over it, so a
    # head that moved between the confirm and this request must not be written.
    expectedHeadRevisionId: str | None = Field(default=None, max_length=80)


class _RepairCancelBody(BaseModel):
    # An older client posts `{}`, which must keep the queued-only behaviour.
    discardReady: bool = False


class _RepairReleaseBody(BaseModel):
    commentThreadId: str = Field(min_length=1, max_length=100)


def _owner_workspace(session, project_ref: str, artifact_id: str):
    """Resolve one scoped artifact and enforce its source-mutation boundary.

    Resolution goes through the review path so a reviewer the owner granted
    access to is refused with 403 rather than 404: they are looking at the
    draft, and "not found" would read as deleted. Anything without a grant is
    still invisible — `review_artifact_for_request` raises 404 there.
    """
    source, folder, metadata, _is_own = review_artifact_for_request(
        session, project_ref, artifact_id
    )
    capabilities = require_artifact_owner(session, source)
    return source, folder, metadata, capabilities


async def _sync_live_artifact(session, folder: Path) -> bool | None:
    """Re-publish a live artifact after an editor write.

    ``None`` means the artifact is only a draft, ``True`` means its stable URL
    was updated, and ``False`` means the source was saved but publishing failed.
    A publish failure must not turn a committed source edit into a false save
    failure: retrying that request with the old revision token would only create
    a conflict. Org autopublish can retry on the next turn, while Desktop keeps
    the artifact's modified state visible for a manual retry.
    """
    from cowork.services.publish import (
        desktop_publish_credential,
        publish_artifact,
        published_artifact_access,
    )
    from cowork.services.artifact_locks import release

    artifacts_base = folder.parent
    if not await _acquire_live_publish_lock(folder):
        logger.warning("Could not synchronize live artifact %s: publish lock busy", folder)
        return False

    key = None
    publish_abandoned = False
    publish_started = False
    try:
        try:
            access = await run_in_threadpool(
                published_artifact_access,
                folder,
                artifacts_base=artifacts_base,
            )
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning("Could not read live publish state for %s", folder, exc_info=True)
            return False

        scope = getattr(session, "scope", None)
        if scope is not None and getattr(scope, "org_mode", False):
            artifacts_base, publish_url, key = _owner_publish_context(session, folder)
            api_key = await key.get()
            if not api_key:
                logger.warning("Could not synchronize live artifact %s: no publish key", folder)
                return False
        else:
            api_key, publish_url = await run_in_threadpool(desktop_publish_credential)

        publish_started = True
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    publish_artifact,
                    folder,
                    artifacts_base=artifacts_base,
                    api_key=api_key,
                    publish_url=publish_url,
                    access=access,
                    scope=scope,
                ),
                timeout=_LIVE_PUBLISH_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            publish_abandoned = True
            logger.warning(
                "Could not synchronize live artifact %s: publish timed out", folder
            )
            return False
        return True
    except asyncio.CancelledError:
        publish_abandoned = publish_started
        raise
    except Exception:
        logger.warning("Could not synchronize live artifact %s", folder, exc_info=True)
        return False
    finally:
        if not publish_abandoned:
            await run_in_threadpool(release, folder.parent, folder.name)
        if key is not None and not publish_abandoned:
            await key.revoke()


async def _acquire_live_publish_lock(folder: Path) -> bool:
    from cowork.services.artifact_locks import acquire

    loop = asyncio.get_running_loop()
    lock_deadline = loop.time() + _LIVE_PUBLISH_TIMEOUT_S
    while not await run_in_threadpool(
        acquire,
        folder.parent,
        folder.name,
        ttl_s=_LIVE_PUBLISH_LOCK_TTL_S,
    ):
        if loop.time() >= lock_deadline:
            return False
        await asyncio.sleep(0.1)
    return True


async def _current_capabilities(session, capabilities: dict) -> dict:
    if not capabilities["canEdit"]:
        return capabilities
    can_edit = await has_product_permission(session.scope, "artifact.manage")
    can_execute = await has_product_permission(session.scope, "product.execute")
    return {
        **capabilities,
        "canEdit": can_edit,
        "canAddressWithAgent": capabilities["canAddressWithAgent"] and can_edit and can_execute,
        "canResolveComments": capabilities["canResolveComments"] and can_edit,
    }


@router.get("/workspace/{project_ref}/{artifact_id}")
async def artifact_source(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
    path: str | None = Query(default=None, max_length=1000),
):
    """Authenticated source + revision token for Desktop and Cowork SaaS."""
    source, folder, metadata, capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    selected = await run_in_threadpool(_editable_source_selector, source, folder, path)
    if selected is None:
        selected = await run_in_threadpool(
            _recorded_source_selector, source, folder, metadata
        )
    try:
        result = await run_in_threadpool(
            current_workspace, folder, metadata, artifact_id, selected
        )
        repair = await run_in_threadpool(active_agent_repair, folder, result.get("path"))
        capabilities = await _current_capabilities(session, capabilities)
        return {**result, "capabilities": capabilities, "repair": repair}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/workspace/{project_ref}/{artifact_id}")
async def update_artifact_source(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _SourceUpdateBody,
    session: ScopedSessionDep,
):
    """Optimistic, atomic manual edit. A stale tab receives 409, never overwrite."""
    await require_product_permission(session.scope, "artifact.manage")
    source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    scope = getattr(session, "scope", None)
    actor_id = str(scope.user_id) if scope and scope.user_id else None
    selected = await run_in_threadpool(_editable_source_selector, source, folder, body.path)
    if selected is None:
        selected = await run_in_threadpool(
            _recorded_source_selector, source, folder, metadata
        )
    try:
        saved = await run_in_threadpool(
            save_source,
            folder,
            metadata,
            artifact_id,
            content=body.content,
            expected_revision_id=body.expectedRevisionId,
            rel_path=selected,
            actor_kind="manual",
            actor_id=actor_id,
            summary=body.summary,
        )
        if saved["revision"]["id"] != body.expectedRevisionId:
            await _sync_live_artifact(session, folder)
        return saved
    except RevisionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "currentRevision": exc.current},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (RevisionValidationError, TimeoutError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workspace/{project_ref}/{artifact_id}/revisions")
async def artifact_revisions(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
    path: str | None = Query(default=None, max_length=1000),
):
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    return {"revisions": await run_in_threadpool(list_revisions, folder, rel_path=path)}


@router.get("/workspace/{project_ref}/{artifact_id}/review")
async def artifact_review_entry(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
):
    """What a reviewer needs to comment, and nothing that reveals the source.

    Separate from the `comments-access` POST below because that one mints an
    auth rule: provisioning is the owner's decision, so a reviewer opening the
    artifact must not be what performs it. A reviewer lands here instead and
    gets the revision to anchor comments to; the source itself stays behind
    `require_artifact_owner`.
    """
    from cowork.services.artifact_identity import artifact_key

    source, folder, metadata, _is_own = review_artifact_for_request(
        session, project_ref, artifact_id
    )
    capabilities = await _current_capabilities(session, artifact_capabilities(session, source))
    current_revision = None
    try:
        draft = await run_in_threadpool(current_source, folder, metadata, artifact_id)
        current_revision = draft.get("revision")
    except (FileNotFoundError, OSError, ValueError, TimeoutError):
        # Binary/oversized artifacts still support general review comments.
        pass
    return {
        "artifactKey": artifact_key(artifact_id),
        "capabilities": capabilities,
        "currentRevision": current_revision,
    }


class _AccessBody(BaseModel):
    """The access selection, in `anton.publish_access.resolve_access` shape.

    Passed through rather than re-modelled per mode: the publisher owns the
    schema (`{"mode": "public"}`, `{"mode": "password", "password": ...}`,
    `{"mode": "restricted", "emails": [...], "org_allowed": bool,
    "owner_only": bool}`) and validates it, so a second definition here could
    only drift away from it.
    """

    access: dict


def _owner_publish_context(session, folder: Path):
    """The (artifacts_base, publish_url, key) an org-mode publish needs.

    The same three `autopublish_project_artifacts` resolves, and deliberately
    not `publish.py`'s `_desktop_context`: that one wants an absolute path from
    the request plus a credential out of stored provider settings, neither of
    which exists on an org deployment — which is why the whole `/publish` router
    is local-only.
    """
    from cowork.services.artifact_autopublish import _publish_url
    from cowork.services.artifact_publish_key import PublishKey

    scope = session.scope
    return (
        folder.parent,
        _publish_url(scope),
        PublishKey(str(scope.user_id), str(scope.org_id), min_ttl_s=_LIVE_PUBLISH_TIMEOUT_S + 60.0),
    )


def _artifact_primary(folder: Path, metadata: dict | None):
    from cowork.services.artifacts import _pick_primary, _user_files

    return _pick_primary(folder, _user_files(folder), primary_hint=(metadata or {}).get("primary"))


@router.get("/workspace/{project_ref}/{artifact_id}/access")
async def artifact_access(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
):
    """The owner's full access state for the Share control.

    Owner-only, and that is the point. The artifact CARD drops `accessEmails`
    and `accessPassword` in org mode because one artifacts root is shared by the
    whole organization, so the card cannot tell owner from co-member. This route
    can: `_owner_workspace` refuses anyone else, so the owner gets back what they
    need to pre-fill the dialog without widening what a card exposes.
    """
    from cowork.services.artifacts import _published_access_for

    _source, folder, metadata, _capabilities = _owner_workspace(session, project_ref, artifact_id)
    primary = await run_in_threadpool(_artifact_primary, folder, metadata)
    if primary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This artifact has no publishable file.",
        )
    return await run_in_threadpool(_published_access_for, folder, primary)


@router.put("/workspace/{project_ref}/{artifact_id}/access")
async def set_artifact_access(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _AccessBody,
    session: ScopedSessionDep,
):
    """Re-publish this artifact with a new audience. Owner-only.

    A publish, not a separate access API, because the publish target stores
    access alongside the bundle and reuses the existing `report_id` — so the
    shared URL survives the change. This is the same call autopublish makes on
    every turn, with the owner's selection in place of the first-publish default.
    """
    await require_product_permission(session.scope, "artifact.manage")
    from cowork.services.publish import publish_artifact as _publish_bundle
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    _source, folder, metadata, _capabilities = _owner_workspace(session, project_ref, artifact_id)
    if _artifact_primary(folder, metadata) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This artifact has no publishable file.",
        )
    if not await _acquire_live_publish_lock(folder):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Publishing is busy right now. Try again in a moment.",
        )
    from cowork.services.artifact_locks import release

    key = None
    publish_abandoned = False
    publish_started = False
    try:
        artifacts_base, publish_url, key = _owner_publish_context(session, folder)
        api_key = await key.get()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Publishing is unavailable right now. Try again in a moment.",
            )
        publish_started = True
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _publish_bundle,
                    folder,
                    artifacts_base=artifacts_base,
                    api_key=api_key,
                    publish_url=publish_url,
                    access=dict(body.access or {}),
                    scope=session.scope,
                ),
                timeout=_LIVE_PUBLISH_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            publish_abandoned = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Publishing timed out. Try again in a moment.",
            ) from exc
    except asyncio.CancelledError:
        publish_abandoned = publish_started
        raise
    except ArtifactAccessUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        if not publish_abandoned:
            await run_in_threadpool(release, folder.parent, folder.name)


@router.post("/workspace/{project_ref}/{artifact_id}/comments-access")
async def enable_artifact_comments(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
):
    """Provision same-org draft review without broadening source-edit access.

    Owner-only: this mints an auth rule and reopens a private conversation
    workspace to the organization, which is the owner's call to make. A
    reviewer's client calls the `review` GET above instead.
    """
    await require_product_permission(session.scope, "artifact.manage")
    from cowork.services.artifact_access import (
        ArtifactAccessUnavailable,
        provision_draft_review_access,
    )
    from cowork.services.artifact_draft_review import enable_draft_review
    from cowork.services.artifact_authorization_identity import ensure_authorization_key
    from cowork.services.artifact_identity import artifact_key

    source, folder, metadata, capabilities = _owner_workspace(session, project_ref, artifact_id)
    capabilities = await _current_capabilities(session, capabilities)
    owner_user_id = artifact_owner_id(session, source)
    try:
        canonical_key = await run_in_threadpool(
            ensure_authorization_key,
            artifact_id,
            session.scope,
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        )
        await provision_draft_review_access(
            canonical_key.split("/", 1)[1],
            session.scope,
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        )
    except ArtifactAccessUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    # After the auth rule, never before: the marker is what reopens the folder
    # to co-members on this server, and a rule-less grant would show a draft the
    # comments service then refuses to talk about.
    await run_in_threadpool(
        enable_draft_review,
        folder,
        org_id=str(session.scope.org_id),
        enabled_by=str(session.scope.user_id),
    )
    current_revision = None
    try:
        draft = await run_in_threadpool(current_source, folder, metadata, artifact_id)
        current_revision = draft.get("revision")
    except (FileNotFoundError, OSError, ValueError, TimeoutError):
        # Binary/oversized artifacts still support general review comments.
        pass
    return {
        "enabled": True,
        # Workspace routes/cards keep the local id. The authenticated comments
        # proxy translates it through the same durable alias used to publish.
        "artifactKey": artifact_key(artifact_id),
        "scope": "organization",
        "capabilities": capabilities,
        "currentRevision": current_revision,
    }


@router.get("/workspace/{project_ref}/{artifact_id}/revisions/{revision_id}")
async def artifact_revision(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    revision_id: str,
    session: ScopedSessionDep,
):
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        return await run_in_threadpool(revision_with_content, folder, revision_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/revisions/{revision_id}/restore")
async def restore_artifact_revision(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    revision_id: str,
    body: _RestoreBody,
    session: ScopedSessionDep,
):
    await require_product_permission(session.scope, "artifact.manage")
    _source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        restored = await run_in_threadpool(revision_with_content, folder, revision_id)
        scope = getattr(session, "scope", None)
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        saved = await run_in_threadpool(
            save_source,
            folder,
            metadata,
            artifact_id,
            content=restored["content"],
            expected_revision_id=body.expectedRevisionId,
            rel_path=restored["path"],
            actor_kind="manual",
            actor_id=actor_id,
            summary=f"Restored revision {restored['number']}",
        )
        if saved["revision"]["id"] != body.expectedRevisionId:
            await _sync_live_artifact(session, folder)
        return saved
    except RevisionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "currentRevision": exc.current},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (RevisionValidationError, TimeoutError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs")
async def request_agent_repair(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _AgentRepairBody,
    session: ScopedSessionDep,
):
    await require_product_permission(session.scope, "artifact.manage")
    await require_product_permission(session.scope, "product.execute")
    _source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        return await run_in_threadpool(
            create_agent_repair,
            folder,
            metadata,
            artifact_id,
            expected_revision_id=body.expectedRevisionId,
            comment_thread_id=body.commentThreadId,
            selector=body.selector,
            thread=[entry.model_dump() for entry in body.thread],
            conversation_id=str(body.conversationId),
        )
    except RevisionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "currentRevision": exc.current},
        ) from exc
    except RepairAlreadyPending as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "repairId": exc.repair.get("id"),
                "commentThreadId": exc.repair.get("commentThreadId"),
            },
        ) from exc
    except (RevisionValidationError, TimeoutError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}")
async def get_agent_repair(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    repair_id: str,
    session: ScopedSessionDep,
):
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        return await run_in_threadpool(agent_repair_detail, folder, repair_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs/release")
async def release_agent_repairs_for_comment(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _RepairReleaseBody,
    session: ScopedSessionDep,
):
    """Release the repairs waiting on a comment thread the owner resolved.

    This lives on the workspace router rather than the comments one because
    the comments route forwards to inference in org mode and carries no tenant
    scope, so only here can one call serve both desktop and cloud.
    """
    await require_product_permission(session.scope, "artifact.manage")
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        released = await run_in_threadpool(
            release_repairs_for_comment, folder, body.commentThreadId
        )
        return {"released": released}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (RevisionValidationError, TimeoutError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}/cancel")
async def cancel_queued_agent_repair(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    repair_id: str,
    session: ScopedSessionDep,
    body: _RepairCancelBody | None = None,
):
    """Release a queued repair, or discard a ready one the owner is done with."""
    await require_product_permission(session.scope, "artifact.manage")
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        return await run_in_threadpool(
            cancel_agent_repair,
            folder,
            repair_id,
            discard_ready=bool(body and body.discardReady),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}/decision")
async def decide_agent_repair(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    repair_id: str,
    body: _RepairDecisionBody,
    session: ScopedSessionDep,
):
    await require_product_permission(session.scope, "artifact.manage")
    _source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        scope = getattr(session, "scope", None)
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        return await run_in_threadpool(
            finalize_agent_repair,
            folder,
            metadata,
            artifact_id,
            repair_id,
            body.status,
            actor_id=actor_id,
            expected_head_revision_id=body.expectedHeadRevisionId,
        )
    except RevisionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "currentRevision": exc.current},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/drafts/{project_ref}/{artifact_id}/{rel_path:path}")
async def serve_private_draft(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    rel_path: str,
    request: Request,
    session: ScopedSessionDep,
    download: Annotated[bool, Query()] = False,
):
    """Authenticated draft preview with project/org containment and relative assets.

    Open to a reviewer as well as the owner, but only through
    `review_artifact_for_request`: a co-member's draft is reachable here solely
    because its owner granted same-org review on that one artifact.

    ``?download=1`` returns the same bytes as an attachment (ENG-2044). On an
    org deployment this is the ONLY way to obtain a non-HTML artifact: the
    stateless ``/serve`` route is desktop-only there, and autopublish skips
    anything that is not HTML/Markdown. Authorization is unchanged — a review
    grant already lets its holder read every byte through the preview, so the
    header changes how the response is labelled, not who may read it.
    ``Annotated[..., Query()] = False`` rather than ``= Query(False)`` so a
    direct call (the tests') gets a real ``False``, not the ``Query`` object.
    """
    # Parse before taking basename so a path ending in a valid UUID is rejected,
    # never silently accepted. Keep the recognized sanitizer at this filesystem
    # boundary; SAST does not model the UUID dependency or catalog lookup.
    if project_ref == "local":
        project_selector = "local"
    else:
        try:
            project_selector = os.path.basename(str(UUID(project_ref)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid project",
            ) from exc
    try:
        artifact_selector = os.path.basename(UUID(artifact_id).hex)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identity",
        ) from exc
    source, folder, metadata, _is_own = review_artifact_for_request(
        session, project_selector, artifact_selector
    )
    try:
        parts = _relative_file_parts(rel_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path",
        ) from exc
    if parts[0] in _PRIVATE_DRAFT_ENTRIES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    if str(metadata.get("type") or "").startswith("fullstack-"):
        try:
            primary_parts = _relative_file_parts(
                str(metadata.get("primary") or "static/index.html")
            )
        except ValueError:
            primary_parts = ()
        public_parts = primary_parts[:-1]
        if not public_parts:
            # A full-stack artifact needs a distinct public subtree. Serving its
            # root would let a reviewer guess backend.py or credential-bearing
            # runtime files. The dedicated runtime can handle legacy root-level
            # apps; the private source preview must stay fail-closed.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact file not found",
            )
        if parts[:len(public_parts)] != public_parts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact file not found",
            )

    media_type = mimetypes.guess_type(parts[-1])[0] or "application/octet-stream"
    resources, fd, file_stat = _open_pinned_draft_file(source, folder, parts)
    try:
        if not download and wants_comment_layer(media_type, request):
            resp = await run_in_threadpool(_comment_layer_from_fd, fd)
            if resp is not None:
                resources.close()
                return resp
        extra = (
            {"Content-Disposition": _attachment_disposition(parts[-1])}
            if download else None
        )
        return _draft_stream(
            resources, fd, file_stat.st_size, media_type, extra_headers=extra,
        )
    except BaseException:
        resources.close()
        raise
