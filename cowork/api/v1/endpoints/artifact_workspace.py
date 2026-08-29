"""Authenticated artifact draft editing, revisions, review access and repair routes."""
from __future__ import annotations

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
    dir_open,
    dir_scandir,
    open_pinned_child,
)
from cowork.api.v1.artifact_scope import review_artifact_for_request
from cowork.db.scoped import ScopedSessionDep
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    artifact_owner_id,
    require_artifact_owner,
)
from cowork.services.comments_layer import inject_layer
from cowork.services.artifact_revisions import (
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
    revision_with_content,
    save_source,
)

router = APIRouter()

_DRAFT_RESPONSE_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}
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


def _existing_draft_entry_name(directory, requested: str) -> str:
    """Return the pinned directory's own name for a request selector.

    Validation makes ``requested`` a single component, but it still originated
    in an HTTP path.  Compare it with entries already discovered below the
    pinned directory and return ``DirEntry.name`` so no request-derived string
    is ever supplied to ``openat``.  A replacement after the scan remains safe:
    the subsequent descriptor-relative open uses ``O_NOFOLLOW``.
    """
    with dir_scandir(directory) as entries:
        for entry in entries:
            if entry.name == requested:
                return entry.name
    raise FileNotFoundError(requested)


def _open_pinned_draft_file(source, folder: Path, parts: tuple[str, ...]):
    """Open one regular draft file without following any writable symlink.

    Authorization chose ``source`` and ``folder`` before this call. The source
    retains its server-owned anchor, and ``opened_artifact_folder`` walks from
    that anchor to the artifact with ``O_NOFOLLOW`` on every component. The
    request path is then walked the same way. Returning the ``ExitStack`` keeps
    every descriptor alive until the response has consumed the final file.
    """
    from cowork.services.artifact_identity import opened_artifact_folder

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


@router.get("/workspace/{project_ref}/{artifact_id}")
async def artifact_source(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    session: ScopedSessionDep,
    path: str | None = Query(default=None),
):
    """Authenticated source + revision token for Desktop and Cowork SaaS."""
    _source, folder, metadata, capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        result = await run_in_threadpool(current_workspace, folder, metadata, artifact_id, path)
        repair = await run_in_threadpool(active_agent_repair, folder)
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
    _source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    scope = getattr(session, "scope", None)
    actor_id = str(scope.user_id) if scope and scope.user_id else None
    try:
        return await run_in_threadpool(
            save_source,
            folder,
            metadata,
            artifact_id,
            content=body.content,
            expected_revision_id=body.expectedRevisionId,
            rel_path=body.path,
            actor_kind="manual",
            actor_id=actor_id,
            summary=body.summary,
        )
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
    path: str | None = Query(default=None),
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
    capabilities = artifact_capabilities(session, source)
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
    from cowork.services.artifact_access import (
        ArtifactAccessUnavailable,
        provision_draft_review_access,
    )
    from cowork.services.artifact_draft_review import enable_draft_review

    source, folder, metadata, capabilities = _owner_workspace(session, project_ref, artifact_id)
    owner_user_id = artifact_owner_id(session, source)
    try:
        key = await provision_draft_review_access(
            artifact_id,
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
        "artifactKey": key,
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
    _source, folder, metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        restored = await run_in_threadpool(revision_with_content, folder, revision_id)
        scope = getattr(session, "scope", None)
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        return await run_in_threadpool(
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


@router.post("/workspace/{project_ref}/{artifact_id}/agent-repairs/{repair_id}/cancel")
async def cancel_queued_agent_repair(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    repair_id: str,
    session: ScopedSessionDep,
):
    """Release a queued repair when its agent turn could not be started."""
    _source, folder, _metadata, _capabilities = _owner_workspace(
        session, project_ref, artifact_id
    )
    try:
        return await run_in_threadpool(cancel_agent_repair, folder, repair_id)
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
    source, folder, metadata, _is_own = review_artifact_for_request(
        session, project_ref, artifact_id
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
