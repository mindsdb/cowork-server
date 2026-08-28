"""Authenticated artifact draft editing, revisions, review access and repair routes."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cowork.api.v1.artifact_preview import (
    html_with_comment_layer,
    wants_comment_layer,
)
from cowork.api.v1.artifact_scope import (
    review_artifact_for_request,
    workspace_artifact_for_request,
)
from cowork.db.scoped import ScopedSessionDep
from cowork.services.artifact_permissions import (
    artifact_capabilities,
    artifact_owner_id,
    require_artifact_owner,
)
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
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
    artifact_id: str,
    rel_path: str,
    request: Request,
    session: ScopedSessionDep,
):
    """Authenticated draft preview with project/org containment and relative assets.

    Open to a reviewer as well as the owner, but only through
    `review_artifact_for_request`: a co-member's draft is reachable here solely
    because its owner granted same-org review on that one artifact.
    """
    _source, folder, metadata, _is_own = review_artifact_for_request(
        session, project_ref, artifact_id
    )
    if not rel_path or "\x00" in rel_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path")
    try:
        target = (folder / rel_path).resolve(strict=False)
        target.relative_to(folder.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc
    try:
        relative = target.relative_to(folder.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Artifact path is outside the draft") from exc
    if not relative.parts or relative.parts[0] in {
        ".revisions", ".published.json", "metadata.json", "README.md", "backend.log",
    }:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    if str(metadata.get("type") or "").startswith("fullstack-"):
        primary = Path(str(metadata.get("primary") or "static/index.html"))
        primary_parent = primary.parent
        if primary.is_absolute() or ".." in primary.parts or primary_parent == Path("."):
            # A full-stack artifact needs a distinct public subtree. Serving its
            # root would let a reviewer guess backend.py or credential-bearing
            # runtime files. The dedicated runtime can handle legacy root-level
            # apps; the private source preview must stay fail-closed.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact file not found",
            )
        allowed_root = (folder / primary_parent).resolve(strict=False)
        try:
            allowed_root.relative_to(folder.resolve())
            target.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact file not found",
            ) from exc
    if not target.is_file() or target.is_symlink():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if wants_comment_layer(media_type, request):
        resp = await run_in_threadpool(html_with_comment_layer, target)
        if resp is not None:
            resp.headers["Cache-Control"] = "private, no-store"
            resp.headers["X-Content-Type-Options"] = "nosniff"
            return resp
    return FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
