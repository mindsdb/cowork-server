"""Authenticated artifact draft editing, revisions, review access and repair routes."""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import stat
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from functools import partial
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from anton.core.artifacts.internal_files import NON_CONTENT_NAMES

from cowork.api.v1.artifact_preview import wants_comment_layer, wants_download
from cowork.api.v1.content_disposition import attachment_disposition
from cowork.api.v1.pinned_response import PinnedStreamingResponse
from cowork.common.paths import (
    dir_lstat,
    open_pinned_child,
    read_bounded,
)
from cowork.api.v1.artifact_scope import review_artifact_for_request
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import default_draft_env_label, get_app_settings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import FilesystemBusy, run_db, run_efs, run_on_unit_threads
from cowork.draft_host.token import DRAFT_TTL_SECONDS, issue_draft_token
from cowork.services.product_permissions import has_product_permission, require_product_permission
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    require_artifact_owner,
    with_export_formats,
)
from cowork.services import artifact_files
from cowork.services.preview_html import prepare_preview_html
from cowork.services.artifact_identity import opened_artifact_folder
from cowork.schemas.artifact_export import ExportRequest, ExportResult
from cowork.schemas.base import CamelResponse
from cowork.services.artifact_export import (
    ExportRefusal,
    ExportRefused,
    ExportSource,
    export_primary_file,
    export_slot,
    org_export_refused,
    primary_export_source,
)
from cowork.services.render_pdf_client import RenderPdfFailed
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

# AuthenticatedInOrgMode, declared explicitly: a ScopedSession already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

_DRAFT_RESPONSE_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}
_LIVE_PUBLISH_TIMEOUT_S = 60.0
_LIVE_PUBLISH_LOCK_TTL_S = _LIVE_PUBLISH_TIMEOUT_S * 3
# Matched against the first path component of a draft request. The file
# walk refuses them there casefolded too, and the store-owned ones at any
# depth (`artifact_files.refuse_housekeeping`).
_PRIVATE_DRAFT_ENTRIES = NON_CONTENT_NAMES


def _artifact_folder_component(source, folder: Path) -> str:
    """Return the direct child of ``source.base`` selected by resolution, or 404.

    ``artifact_files.artifact_folder_name`` holds the rule: a folder whose
    parent is not the source's base is refused, so a grant for one resolved
    folder can't select a same-named folder under another source.
    """
    try:
        return artifact_files.artifact_folder_name(source=source, folder=folder)
    except artifact_files.ArtifactFileRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        ) from exc


def _editable_source_selector(source, folder: Path, requested: str | None) -> str | None:
    """Translate a request-supplied source path into the folder's own spelling.

    ``None`` (or blank) leaves the choice to the service, which falls back to
    ``metadata["primary"]``. That is not a trusted value either: it is read
    from `metadata.json` inside the artifact folder, which is pod-writable on
    shared storage, so what makes the fallback safe is the inner gate's own
    containment, symlink and extension checks rather than where it came from.
    Anything else is the same kind of request-derived string
    ``open_pinned_draft_file`` refuses to
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
        parts = artifact_files.relative_file_parts(requested.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid artifact source path") from exc
    # Only the journal (any depth), like the inner gate: the private-listing
    # set is not rejected because `metadata.primary` or an explicit path may
    # legitimately name README.md or prd.md as the editable source.
    if JOURNAL_DIRNAME in parts:
        raise HTTPException(status_code=422, detail="Invalid artifact source path")
    folder_name = _artifact_folder_component(source, folder)
    disk_parts: list[str] = []
    try:
        with ExitStack() as resources:
            current = resources.enter_context(opened_artifact_folder(source, folder_name))
            for part in parts[:-1]:
                disk_name = artifact_files.existing_draft_entry_name(
                    current, part, native_case=True
                )
                current = open_pinned_child(current, disk_name)
                resources.callback(current.close)
                disk_parts.append(disk_name)
            disk_name = artifact_files.existing_draft_entry_name(
                current, parts[-1], native_case=True
            )
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


def _preview_html_from_fd(fd: int, *, comments: bool) -> HTMLResponse | None:
    """Build the preview HTML from the already-authorized file descriptor."""
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
    return HTMLResponse(
        prepare_preview_html(html, comments=comments), headers=_DRAFT_RESPONSE_HEADERS
    )


def _draft_stream(
    resources: ExitStack,
    fd: int,
    size: int,
    media_type: str,
    *,
    extra_headers: dict[str, str] | None = None,
):
    """Stream at most *size* bytes from the pinned descriptor and close every
    held handle.

    No ``Content-Length``: the agent may still be writing the file, and a
    length taken from ``fstat`` would then disagree with the bytes read, which
    the browser reports as ``ERR_CONTENT_LENGTH_MISMATCH`` (ENG-2950). Chunked
    transfer sends exactly what the descriptor yields. *size* (the ``fstat``
    size at authorization) still bounds the body, so a file that keeps growing
    cannot keep the response open.
    """
    def chunks():
        try:
            yield from read_bounded(fd, size)
        finally:
            resources.close()

    return PinnedStreamingResponse(
        chunks(),
        resources=resources,
        media_type=media_type,
        headers={**_DRAFT_RESPONSE_HEADERS, **(extra_headers or {})},
    )


def _draft_selectors(project_ref: str, artifact_id: str) -> tuple[str, str]:
    """The project and artifact selectors a draft route resolves with.

    Parse before taking basename so a path ending in a valid UUID is rejected,
    never silently accepted. Keep the recognized sanitizer at this filesystem
    boundary; SAST does not model the UUID dependency or catalog lookup.
    """
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
    return project_selector, artifact_selector


def _draft_file_parts(rel_path: str, metadata: dict) -> tuple[str, ...]:
    """Validate a draft file path against the artifact it is read from.

    Shared by the authenticated ``/drafts`` route, the draft-link route and the
    draft host (cowork/draft_host/app.py), so all three refuse the same paths:
    400 for a path that is not a plain relative one, 404 for the private
    entries and, in a full-stack artifact, anything outside its public subtree.
    """
    try:
        parts = artifact_files.relative_file_parts(rel_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path",
        ) from exc
    if parts[0] in _PRIVATE_DRAFT_ENTRIES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    if str(metadata.get("type") or "").startswith("fullstack-"):
        try:
            primary_parts = artifact_files.relative_file_parts(
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
    return parts


def _draft_file_response(
    source,
    folder: Path,
    parts: tuple[str, ...],
    *,
    comments: bool,
    download: bool = False,
):
    """Open the draft file at ``parts`` and answer with it.

    Shared by the authenticated ``/drafts`` route and the draft host. HTML goes
    out as the preview document unless ``download`` asks for the bytes; every
    other file, and HTML that is not UTF-8, streams from the pinned descriptor.
    Blocking, so callers run it in a worker thread.
    """
    media_type = mimetypes.guess_type(parts[-1])[0] or "application/octet-stream"
    try:
        resources, fd, file_stat = artifact_files.open_pinned_draft_file(
            source, folder, parts
        )
    except artifact_files.ArtifactFileRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        ) from exc
    try:
        if not download and media_type == "text/html":
            response = _preview_html_from_fd(fd, comments=comments)
            if response is not None:
                resources.close()
                return response
        extra = (
            {"Content-Disposition": attachment_disposition(parts[-1])}
            if download else None
        )
        return _draft_stream(
            resources, fd, file_stat.st_size, media_type, extra_headers=extra,
        )
    except BaseException:
        resources.close()
        raise


async def _artifact_id_from_path(artifact_id: UUID) -> str:
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
    # Deliberately untyped. A typed model would answer 422 on the first odd
    # entry, which would fail the repair over a diagnostic — the opposite of
    # what diagnostics are for. create_agent_repair drops anything malformed.
    previewErrors: Any = None


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


async def _owner_workspace(scope: TenantScope, project_ref: str, artifact_id: str):
    """Resolve one scoped artifact and enforce its source-mutation boundary.

    Resolution goes through the review path so a reviewer the owner granted
    access to is refused with 403 rather than 404: they are looking at the
    draft, and "not found" would read as deleted. Anything without a grant is
    still invisible — `review_artifact_for_request` raises 404 there.
    """
    source, folder, metadata, _is_own = await review_artifact_for_request(
        scope, project_ref, artifact_id
    )
    capabilities = await run_db(
        lambda unit: require_artifact_owner(unit, source, folder.name), scope=scope
    )
    return source, folder, metadata, capabilities


async def _sync_live_artifact(
    scope: TenantScope, folder: Path, *, project_id: str | None = None
) -> bool | None:
    """Re-publish a live artifact after an editor write.

    ``None`` means the artifact is only a draft, ``True`` means its stable URL
    was updated, and ``False`` means the source was saved but publishing failed.
    A publish failure must not turn a committed source edit into a false save
    failure: retrying that request with the old revision token would only create
    a conflict. Org autopublish can retry on the next turn, while Desktop keeps
    the artifact's modified state visible for a manual retry.
    """
    from cowork.services.publish import (
        PublishOwnedByOtherUser,
        PublishTooLarge,
        desktop_publish_credential,
        publish_artifact,
        published_artifact_access,
    )
    from cowork.services.artifact_locks import release

    artifacts_base = folder.parent
    if not await _acquire_live_publish_lock(folder, scope):
        logger.warning("Could not synchronize live artifact %s: publish lock busy", folder)
        return False

    key = None
    publish_abandoned = False
    publish_started = False
    try:
        try:
            access = await run_efs(
                partial(
                    published_artifact_access,
                    folder,
                    artifacts_base=artifacts_base,
                ),
                scope=scope,
            )
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning("Could not read live publish state for %s", folder, exc_info=True)
            return False

        if scope.org_mode:
            artifacts_base, publish_url, key = await run_on_unit_threads(partial(_owner_publish_context, scope, folder))
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
                    project_id=project_id,
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
    except PublishOwnedByOtherUser:
        # Repeats on every save of this artifact, so no traceback.
        logger.warning("Could not synchronize live artifact %s: another account owns its report", folder)
        return False
    except PublishTooLarge:
        # Repeats on every save until the artifact shrinks, so no traceback.
        logger.warning("Could not synchronize live artifact %s: it is too large to publish", folder)
        return False
    except Exception:
        logger.warning("Could not synchronize live artifact %s", folder, exc_info=True)
        return False
    finally:
        if not publish_abandoned:
            # Waits for a slot: a lock left behind would block the next publish.
            await run_efs(
                partial(release, folder.parent, folder.name), scope=scope, deadline=False
            )
        if key is not None and not publish_abandoned:
            await key.revoke()


async def _acquire_live_publish_lock(folder: Path, scope: TenantScope) -> bool:
    from cowork.services.artifact_locks import acquire

    loop = asyncio.get_running_loop()
    lock_deadline = loop.time() + _LIVE_PUBLISH_TIMEOUT_S
    while True:
        try:
            if await run_efs(
                partial(acquire, folder.parent, folder.name, ttl_s=_LIVE_PUBLISH_LOCK_TTL_S),
                scope=scope,
            ):
                return True
        except FilesystemBusy:
            pass  # a busy file store is a busy lock: wait and try again
        if loop.time() >= lock_deadline:
            return False
        await asyncio.sleep(0.1)


async def _current_capabilities(scope: TenantScope, capabilities: dict) -> dict:
    if not capabilities["canEdit"]:
        return capabilities
    can_edit = await has_product_permission(scope, "artifact.manage")
    can_execute = await has_product_permission(scope, "product.execute")
    current = {
        **capabilities,
        "canEdit": can_edit,
        "canAddressWithAgent": capabilities["canAddressWithAgent"] and can_edit and can_execute,
        "canResolveComments": capabilities["canResolveComments"] and can_edit,
    }
    # The export route needs `artifact.manage` too, so it offers nothing here.
    if not can_edit and "exportFormats" in capabilities:
        current["exportFormats"] = []
    return current


def _selected_source(source, folder: Path, metadata: dict, requested: str | None) -> str | None:
    """The source file a workspace read or write addresses: the request's own
    path when it names an editable file, else the one the folder records."""
    selected = _editable_source_selector(source, folder, requested)
    if selected is None:
        selected = _recorded_source_selector(source, folder, metadata)
    return selected


def _workspace_state(
    folder: Path, metadata: dict, artifact_id: str, selected: str | None, *, with_export: bool
) -> tuple[dict, dict | None, ExportSource]:
    """What the workspace GET returns from the folder, in one file store hop:
    the source and its revision, the repair in progress, and the export
    source."""
    result = current_workspace(folder, metadata, artifact_id, selected)
    repair = active_agent_repair(folder, result.get("path"))
    export_source = (
        primary_export_source(folder=folder, metadata=metadata)
        if with_export
        else ExportSource(source_name="", is_backend=False)
    )
    return result, repair, export_source


@router.get("/workspace/{project_ref}/{artifact_id}")
async def artifact_source(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    # Spell out the dependency so SAST does not treat the server-derived
    # scope (and the artifact roots it discovers) as an HTTP parameter.
    scope: TenantScope = Depends(get_tenant_scope),
    path: str | None = Query(default=None, max_length=1000),
):
    """Authenticated source + revision token for Desktop and Cowork SaaS.

    ``_owner_workspace`` is the owner gate. The capabilities it returns gain
    ``exportFormats``, which read the artifact's primary file, once the folder
    is resolved, through the helper ``artifact_capabilities`` adds it with.
    """
    source, folder, metadata, owner_capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    selected = await run_efs(
        partial(_selected_source, source, folder, metadata, path), scope=scope
    )
    # No format is offered whatever the folder holds while org export is off,
    # so the walk that finds the primary is skipped, and the source names none.
    with_export = not org_export_refused(org_mode=bool(scope.org_mode))
    try:
        result, repair, export_source = await run_efs(
            partial(_workspace_state, folder, metadata, artifact_id, selected, with_export=with_export),
            scope=scope,
        )
        capabilities = await _current_capabilities(
            scope, with_export_formats(scope, owner_capabilities, export_source)
        )
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
    scope: ScopeDep,
):
    """Optimistic, atomic manual edit. A stale tab receives 409, never overwrite."""
    await require_product_permission(scope, "artifact.manage")
    source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    actor_id = str(scope.user_id) if scope and scope.user_id else None
    selected = await run_efs(
        partial(_selected_source, source, folder, metadata, body.path), scope=scope
    )
    try:
        saved = await run_efs(
            partial(
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
            ),
            scope=scope,
        )
        if saved["revision"]["id"] != body.expectedRevisionId:
            await _sync_live_artifact(scope, folder, project_id=source.project_id)
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


_EXPORT_REFUSAL_STATUS = {
    ExportRefusal.NOT_OFFERED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ExportRefusal.TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    ExportRefusal.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ExportRefusal.BUSY: status.HTTP_429_TOO_MANY_REQUESTS,
}
# What the route says for each status a failed PDF export answers with.
_PDF_FAILURE_DETAIL = {
    413: "This file can't be exported.",
    422: "This file can't be exported.",
    502: "Couldn't export this file.",
    503: "PDF export is busy; try again.",
    504: "PDF export took too long.",
}


@router.post("/workspace/{project_ref}/{artifact_id}/export")
async def export_workspace_artifact(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: ExportRequest,
    scope: ScopeDep,
) -> ExportResult:
    """Export the artifact's primary file as a new HTML, Word or PDF file beside it.

    One route for every tenancy, checked the way a source edit is: the caller
    needs ``artifact.manage`` and must own the artifact, and both checks run
    before the format is read, so a caller who may not export gets 403 or 404
    whatever they ask for. While org export is off, the owner gets 403 too,
    after those checks, so a caller with no grant still gets 404. A busy
    server answers 429 at once. The conversion runs in a worker thread, a PDF
    that doesn't come back answers the render client's status, and only a
    200 leaves a file behind.
    """
    await require_product_permission(scope, "artifact.manage")
    source, folder, metadata, capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    if org_export_refused(org_mode=bool(scope.org_mode)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Export isn't available on this deployment.",
        )
    folder_name = _artifact_folder_component(source, folder)
    try:
        async with export_slot(scope=scope, export_format=body.format):
            return await run_efs(
                partial(
                    export_primary_file,
                    source=source,
                    folder_name=folder_name,
                    metadata=metadata,
                    artifact_id=artifact_id,
                    export_format=body.format,
                    can_edit=bool(capabilities.get("canEdit")),
                ),
                scope=scope,
            )
    except ExportRefused as exc:
        raise HTTPException(
            status_code=_EXPORT_REFUSAL_STATUS[exc.refusal], detail=str(exc)
        ) from exc
    except RenderPdfFailed as exc:
        # The reason never quotes the render service, so it is safe to log.
        logger.info("PDF export failed for %s: %s", artifact_id, exc)
        raise HTTPException(
            status_code=exc.status, detail=_PDF_FAILURE_DETAIL[exc.status]
        ) from exc
    except FilesystemBusy:
        raise  # 503 with Retry-After (cowork.server), not a failed export
    except Exception as exc:
        logger.warning(
            "Artifact export to %s failed for %s", body.format, artifact_id, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Export failed"
        ) from exc


@router.get("/workspace/{project_ref}/{artifact_id}/revisions")
async def artifact_revisions(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: ScopeDep,
    path: str | None = Query(default=None, max_length=1000),
):
    _source, folder, _metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    return {"revisions": await run_efs(partial(list_revisions, folder, rel_path=path), scope=scope)}


@router.get("/workspace/{project_ref}/{artifact_id}/review")
async def artifact_review_entry(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    # Spell out the dependency so SAST does not treat the server-derived
    # scope (and the artifact roots it discovers) as an HTTP parameter.
    scope: TenantScope = Depends(get_tenant_scope),
):
    """What a reviewer needs to comment, and nothing that reveals the source.

    Separate from the `comments-access` POST below because that one mints an
    auth rule: provisioning is the owner's decision, so a reviewer opening the
    artifact must not be what performs it. A reviewer lands here instead and
    gets the revision to anchor comments to; the source itself stays behind
    `require_artifact_owner`.
    """
    from cowork.services.artifact_identity import artifact_key

    source, folder, metadata, _is_own = await review_artifact_for_request(
        scope, project_ref, artifact_id
    )
    capabilities = await _current_capabilities(
        scope,
        await run_db(
            lambda unit: artifact_capabilities(unit, source, folder.name), scope=scope
        ),
    )
    current_revision = None
    try:
        draft = await run_efs(partial(current_source, folder, metadata, artifact_id), scope=scope)
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


def _owner_publish_context(scope: TenantScope, folder: Path):
    """The (artifacts_base, publish_url, key) an org-mode publish needs.

    The same three `autopublish_project_artifacts` resolves, and deliberately
    not `publish.py`'s `_desktop_context`: that one wants an absolute path from
    the request plus a credential out of stored provider settings, neither of
    which exists on an org deployment — which is why the whole `/publish` router
    is local-only.
    """
    from cowork.services.artifact_autopublish import _active_workspace_id, _publish_url
    from cowork.services.artifact_publish_key import PublishKey

    return (
        folder.parent,
        _publish_url(scope),
        PublishKey(
            str(scope.user_id), str(scope.org_id),
            min_ttl_s=_LIVE_PUBLISH_TIMEOUT_S + 60.0,
            workspace_id=_active_workspace_id(scope),
        ),
    )


def _artifact_primary(folder: Path, metadata: dict | None):
    from cowork.services.artifacts import _pick_primary, _user_files

    return _pick_primary(folder, _user_files(folder), primary_hint=(metadata or {}).get("primary"))


@router.get("/workspace/{project_ref}/{artifact_id}/access")
async def artifact_access(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: ScopeDep,
):
    """The owner's full access state for the Share control.

    Owner-only, and that is the point. The artifact CARD drops `accessEmails`
    and `accessPassword` in org mode because one artifacts root is shared by the
    whole organization, so the card cannot tell owner from co-member. This route
    can: `_owner_workspace` refuses anyone else, so the owner gets back what they
    need to pre-fill the dialog without widening what a card exposes.
    """
    from cowork.services.artifacts import _published_access_for

    _source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    primary = await run_efs(partial(_artifact_primary, folder, metadata), scope=scope)
    if primary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This artifact has no publishable file.",
        )
    return await run_efs(partial(_published_access_for, folder, primary), scope=scope)


@router.put("/workspace/{project_ref}/{artifact_id}/access")
async def set_artifact_access(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _AccessBody,
    scope: ScopeDep,
):
    """Re-publish this artifact with a new audience. Owner-only.

    A publish, not a separate access API, because the publish target stores
    access alongside the bundle and reuses the existing `report_id` — so the
    shared URL survives the change. This is the same call autopublish makes on
    every turn, with the owner's selection in place of the first-publish default.
    """
    await require_product_permission(scope, "artifact.manage")
    from cowork.services.publish import PublishRejected, PublisherUnavailable
    from cowork.services.publish import publish_artifact as _publish_bundle
    from cowork.services.artifact_access import ArtifactAccessUnavailable

    source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    if await run_efs(partial(_artifact_primary, folder, metadata), scope=scope) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This artifact has no publishable file.",
        )
    if not await _acquire_live_publish_lock(folder, scope):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Publishing is busy right now. Try again in a moment.",
        )
    from cowork.services.artifact_locks import release

    key = None
    publish_abandoned = False
    publish_started = False
    try:
        artifacts_base, publish_url, key = await run_on_unit_threads(partial(_owner_publish_context, scope, folder))
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
                    scope=scope,
                    project_id=source.project_id,
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
    # PublisherUnavailable and PublishRejected are RuntimeErrors, so they come first.
    except PublisherUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except PublishRejected as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        # A local failure's text can name shared-storage paths, so only the log
        # gets it. Not every such error passed publish_artifact's own log.
        logger.warning("Could not change access for artifact %s", artifact_id, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Publishing failed. Try again in a moment.",
        ) from exc
    finally:
        if not publish_abandoned:
            # Waits for a slot: a lock left behind would block the next publish.
            await run_efs(
                partial(release, folder.parent, folder.name), scope=scope, deadline=False
            )
        if key is not None and not publish_abandoned:
            await key.revoke()


class CommentsAccessResponse(CamelResponse):
    """The owner's draft comments, turned on."""

    enabled: Literal[True] = True
    artifact_key: str
    # "organization": same-org reviewers may comment; "owner": only the owner.
    scope: Literal["organization", "owner"]
    capabilities: dict[str, Any]
    current_revision: dict[str, Any] | None


@router.post("/workspace/{project_ref}/{artifact_id}/comments-access")
async def enable_artifact_comments(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: ScopeDep,
) -> CommentsAccessResponse:
    """Turn on draft comments for the artifact's owner.

    Owner-only. In org mode this provisions same-org draft review without
    broadening source-edit access: it mints an auth rule and reopens a private
    conversation workspace to the organization, which is the owner's call to
    make. A reviewer's client calls the `review` GET above instead.

    Local tenancy provisions nothing. Its comment routes keep the threads in the
    artifact's own folder (`LocalArtifactComments`), which finds the artifact
    only among the caller's artifacts roots, so the owner is the one person who
    reaches them.
    """
    await require_product_permission(scope, "artifact.manage")
    from cowork.services.artifact_access import (
        ArtifactAccessUnavailable,
        provision_draft_review_access,
    )
    from cowork.services.artifact_draft_review import enable_draft_review
    from cowork.services.artifact_authorization_identity import ensure_authorization_key
    from cowork.services.artifact_identity import artifact_key

    source, folder, metadata, capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    capabilities = await _current_capabilities(scope, capabilities)
    if scope.org_mode:
        # `_owner_workspace` already refused anyone but the owner, so the owner is
        # the caller; resolving it again would only repeat that query.
        owner_user_id = str(scope.user_id) if scope.user_id else None
        try:
            canonical_key = await run_in_threadpool(
                ensure_authorization_key,
                artifact_id,
                scope,
                owner_user_id=owner_user_id,
            )
            await provision_draft_review_access(
                canonical_key.split("/", 1)[1],
                scope,
                owner_user_id=owner_user_id,
            )
        except ArtifactAccessUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        # After the auth rule, never before: the marker is what reopens the folder
        # to co-members on this server, and a rule-less grant would show a draft the
        # comments service then refuses to talk about.
        await run_efs(
            partial(
                enable_draft_review,
                folder,
                org_id=str(scope.org_id),
                enabled_by=str(scope.user_id),
            ),
            scope=scope,
        )
        comments_scope = "organization"
    else:
        comments_scope = "owner"
    current_revision = None
    try:
        draft = await run_efs(partial(current_source, folder, metadata, artifact_id), scope=scope)
        current_revision = draft.get("revision")
    except (FileNotFoundError, OSError, ValueError, TimeoutError):
        # Binary/oversized artifacts still support general review comments.
        pass
    return CommentsAccessResponse(
        # Workspace routes/cards keep the local id. The authenticated comments
        # proxy translates it through the same durable alias used to publish.
        artifact_key=artifact_key(artifact_id),
        scope=comments_scope,
        capabilities=capabilities,
        current_revision=current_revision,
    )


@router.get("/workspace/{project_ref}/{artifact_id}/revisions/{revision_id}")
async def artifact_revision(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    revision_id: str,
    scope: ScopeDep,
):
    _source, folder, _metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        return await run_efs(partial(revision_with_content, folder, revision_id), scope=scope)
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
    scope: ScopeDep,
):
    await require_product_permission(scope, "artifact.manage")
    source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        restored = await run_efs(partial(revision_with_content, folder, revision_id), scope=scope)
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        saved = await run_efs(
            partial(
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
            ),
            scope=scope,
        )
        if saved["revision"]["id"] != body.expectedRevisionId:
            await _sync_live_artifact(scope, folder, project_id=source.project_id)
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
    scope: ScopeDep,
):
    await require_product_permission(scope, "artifact.manage")
    await require_product_permission(scope, "product.execute")
    _source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    # Before the repair is queued, so the end of the conversation's turn
    # checks this folder; a failed write queues nothing.
    from cowork.services.queued_repairs import record_queued_repair

    await run_db(
        partial(record_queued_repair, conversation_id=body.conversationId, slug=folder.name), scope=scope
    )
    try:
        return await run_efs(
            partial(
                create_agent_repair,
                folder,
                metadata,
                artifact_id,
                expected_revision_id=body.expectedRevisionId,
                comment_thread_id=body.commentThreadId,
                selector=body.selector,
                thread=[entry.model_dump() for entry in body.thread],
                conversation_id=str(body.conversationId),
                preview_errors=body.previewErrors,
            ),
            scope=scope,
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
    scope: ScopeDep,
):
    _source, folder, _metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        return await run_efs(partial(agent_repair_detail, folder, repair_id), scope=scope)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs/release")
async def release_agent_repairs_for_comment(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _RepairReleaseBody,
    scope: ScopeDep,
):
    """Release the repairs waiting on a comment thread the owner resolved.

    This lives on the workspace router rather than the comments one because
    the comments route forwards to inference in org mode and carries no tenant
    scope, so only here can one call serve both desktop and cloud.
    """
    await require_product_permission(scope, "artifact.manage")
    _source, folder, _metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        released = await run_efs(
            partial(
                release_repairs_for_comment, folder, body.commentThreadId,
            ),
            scope=scope,
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
    scope: ScopeDep,
    body: _RepairCancelBody | None = None,
):
    """Release a queued repair, or discard a ready one the owner is done with."""
    await require_product_permission(scope, "artifact.manage")
    _source, folder, _metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        return await run_efs(
            partial(
                cancel_agent_repair,
                folder,
                repair_id,
                discard_ready=bool(body and body.discardReady),
            ),
            scope=scope,
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
    scope: ScopeDep,
):
    await require_product_permission(scope, "artifact.manage")
    _source, folder, metadata, _capabilities = await _owner_workspace(
        scope, project_ref, artifact_id
    )
    try:
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        return await run_efs(
            partial(
                finalize_agent_repair,
                folder,
                metadata,
                artifact_id,
                repair_id,
                body.status,
                actor_id=actor_id,
                expected_head_revision_id=body.expectedHeadRevisionId,
            ),
            scope=scope,
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


def _primary_rel_path(folder: Path, primary: Path | None) -> str:
    """The primary file's path inside ``folder``, as the draft link names it."""
    if primary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This artifact has no previewable file.",
        )
    try:
        return primary.resolve(strict=False).relative_to(folder.resolve(strict=False)).as_posix()
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file not found",
        ) from exc


def _draft_link_file_parts(folder: Path, metadata: dict) -> tuple[str, ...]:
    """Name the draft's primary file for the link.

    Blocking (path resolution on the file store), so the endpoint runs it in
    a file store thread.
    """
    from cowork.services.artifacts import _pick_primary

    hint = metadata.get("primary")
    # A recorded primary that resolves needs no walk of the artifact tree.
    primary = _pick_primary(folder, [], primary_hint=hint) if hint else None
    if primary is None:
        primary = _artifact_primary(folder, metadata)
    return _draft_file_parts(_primary_rel_path(folder, primary), metadata)


@router.post("/drafts/{project_ref}/{artifact_id}/link")
async def create_draft_link(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    # Spell out the dependency so SAST does not treat the server-derived
    # scope (and the artifact roots it discovers) as an HTTP parameter.
    scope: TenantScope = Depends(get_tenant_scope),
):
    """A 30-minute signed link to this draft on the draft host.

    The same access check as ``serve_private_draft``: the owner, or a same-org
    reviewer the owner granted. The link carries the caller's identity and
    points at the primary file; its relative assets reuse the token. The web
    app asks for a new link on every open, reload and "open in a browser tab",
    so nothing caches or reuses one.
    """
    settings = get_app_settings()
    if not settings.draft_links_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft links are not available",
        )
    if not scope.user_id or not scope.org_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    project_selector, artifact_selector = _draft_selectors(project_ref, artifact_id)
    _source, folder, metadata, _is_own = await review_artifact_for_request(
        scope, project_selector, artifact_selector
    )
    parts = await run_efs(partial(_draft_link_file_parts, folder, metadata), scope=scope)
    now = int(time.time())
    token = issue_draft_token(
        env=default_draft_env_label(),
        sub=str(scope.user_id),
        org=str(scope.org_id),
        proj=project_selector,
        art=artifact_selector,
        keys=settings.draft_link_keyring(),
        kid=settings.draft_link_kid,
        now=now,
    )
    # Counted at WARNING so LOG_LEVEL=WARNING deployments keep it during the
    # rollout. The link itself is never logged.
    logger.warning("Draft link issued", extra={"event": "draft_link_issued"})
    url = f"{settings.draft_link_origin()}/_t/{token}/{quote('/'.join(parts), safe='/')}"
    return JSONResponse(
        {"url": url, "expiresAt": now + DRAFT_TTL_SECONDS},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/drafts/{project_ref}/{artifact_id}/{rel_path:path}")
async def serve_private_draft(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    rel_path: str,
    request: Request,
    # Spell out the dependency so SAST does not treat the server-derived
    # scope (and the artifact roots it discovers) as an HTTP parameter.
    scope: TenantScope = Depends(get_tenant_scope),
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
    The effective flag below also folds in ``wants_download(request)``, the
    same predicate `/serve` and `/preview-asset` use, so all three routes
    agree on what a raw query string like ``?download=0`` means; a direct
    call still controls the outcome through the keyword argument, since its
    bare ``request`` carries no query string of its own.
    """
    project_selector, artifact_selector = _draft_selectors(project_ref, artifact_id)
    source, folder, metadata, _is_own = await review_artifact_for_request(
        scope, project_selector, artifact_selector
    )
    parts = _draft_file_parts(rel_path, metadata)

    return await run_efs(
        partial(
            _draft_file_response,
            source,
            folder,
            parts,
            comments=wants_comment_layer(request),
            download=download or wants_download(request),
        ),
        scope=scope,
    )
