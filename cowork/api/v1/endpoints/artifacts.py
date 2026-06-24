"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, Field
from sqlmodel import Session, select

from cowork.db.session import get_session
from cowork.models.artifact import Artifact, ArtifactActivityEvent, ArtifactComment, ArtifactVersion
from cowork.services.artifacts import (
    _artifact_root_for,
    _load_metadata,
    _pick_primary,
    _project_artifacts_base,
    get_preview_mount,
    list_artifacts_page,
    mount_preview,
    preview_artifact as _preview_artifact,
    resolve_artifact_path,
    reveal_in_file_manager,
    serve_url_for,
    _unpublish_folder,
    _user_files,
    verify_serve_url_token,
)
from cowork.models.project import Project
from cowork.services.projects import ProjectService
from cowork.services.request_identity import (
    AuthenticationError,
    RequestPrincipal,
    principal_from_authorization_header,
)
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned
from cowork.services.project_collaboration import delivery_to_dict, dispatch_project_notification
from cowork.services.artifact_versions import (
    ArtifactVersionService,
    _artifact_from_identifier_or_path,
    apply_comment_patch as _apply_comment_patch,
    create_comment as _create_comment,
    diff_versions as _diff_versions,
    fork_version as _fork_version,
    get_or_create_artifact_for_path,
    list_comments as _list_comments,
    list_versions as _list_versions,
    mark_comments_read as _mark_comments_read,
    preview_comment_patch as _preview_comment_patch,
    record_deployment,
    restore_artifact as _restore_artifact,
    set_comment_status as _set_comment_status,
    snapshot_artifact as _snapshot_artifact,
    version_to_dict,
)
from cowork.services.artifact_handoff import handoff_artifact_to_conversation

logger = logging.getLogger(__name__)

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def get_request_principal(request: Request) -> RequestPrincipal | None:
    try:
        return principal_from_authorization_header(request.headers.get("authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


PrincipalDep = Annotated[RequestPrincipal | None, Depends(get_request_principal)]


# Soft-deleted artifacts get their unique keys (path, slug) suffixed with this prefix
# + a uuid, freeing the originals for a re-create while preserving them in the delete
# event for recovery. Stripped again for display (see _undeleted).
_DELETED_TOMBSTONE_PREFIX = "#deleted-"


def _tombstone(value: str, token: str, max_length: int) -> str:
    """Suffix ``value`` with the delete tombstone, truncating the base so the result
    still fits ``max_length`` (slug 255, path 2048). Without the truncation a long
    slug/path overflows the column under validate_assignment and turns delete into a 500."""
    suffix = f"{_DELETED_TOMBSTONE_PREFIX}{token}"
    return f"{value[: max_length - len(suffix)]}{suffix}"


def _actor_kwargs(principal: RequestPrincipal | None) -> dict[str, str | None]:
    return {
        "actor_name": principal.name if principal is not None else None,
        "actor_email": principal.email if principal is not None else None,
        "actor_subject": principal.subject if principal is not None else None,
    }


def _browser_file_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        **(extra or {}),
    }


def _materialized_version_preview_path(
    session: Session,
    artifact_path: Path,
    version_id: str | None,
) -> Path | None:
    if not version_id:
        return None
    try:
        clean_id = UUID(str(version_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version_id") from exc

    version = session.get(ArtifactVersion, clean_id)
    source_artifact = session.get(Artifact, version.artifact_id) if version is not None else None
    if version is None or source_artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact version not found")
    requested_root = _artifact_root_for(artifact_path)
    if requested_root.resolve(strict=False) != Path(source_artifact.path).resolve(strict=False):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact version not found")

    service = ArtifactVersionService(session)
    target = service.store_root / "previews" / "workspace" / str(source_artifact.id) / str(version.id)
    service.materialize_version(version.id, target, clean=True)
    service.write_version_housekeeping(version.id, target)
    metadata = _load_metadata(target) if (target / "metadata.json").is_file() else {}
    files = _user_files(target)
    primary = _pick_primary(target, files, primary_hint=metadata.get("primary"))
    if primary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact version has no previewable files")
    return primary


def _live_preview_url(payload: dict) -> str | None:
    for key in ("proxyUrl", "serveUrl", "relUrl", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _mark_live_preview_ready(session: Session, artifact_path: Path, payload: dict) -> None:
    try:
        artifact = get_or_create_artifact_for_path(session, str(artifact_path))
        folder = Path(artifact.path)
        service = ArtifactVersionService(session)
        current = session.get(ArtifactVersion, artifact.current_version_id) if artifact.current_version_id else None
        manifest = service.scan_manifest(folder)
        if current is not None and current.files_hash == manifest.files_hash:
            version = current
        else:
            version = service.snapshot_artifact(
                folder,
                artifact_id=artifact.id,
                project_id=artifact.project_id,
                slug=artifact.slug,
                title=artifact.title,
                description=artifact.description,
                artifact_type=artifact.artifact_type,
                operation_type="preview",
                label="Preview passed",
                preview_status="ready",
            )
        record_deployment(
            session,
            version,
            target="preview",
            status="ready",
            url=_live_preview_url(payload),
            details={"kind": payload.get("kind") or "", "path": str(artifact_path)},
        )
    except Exception:
        logger.warning("Failed to mark live preview ready for %s", artifact_path, exc_info=True)
        session.rollback()


def _record_live_preview_failure(session: Session, artifact_path: Path, detail: str) -> None:
    try:
        artifact = get_or_create_artifact_for_path(session, str(artifact_path))
        service = ArtifactVersionService(session)
        failed = service.snapshot_artifact(
            artifact.path,
            artifact_id=artifact.id,
            project_id=artifact.project_id,
            slug=artifact.slug,
            title=artifact.title,
            description=artifact.description,
            artifact_type=artifact.artifact_type,
            operation_type="preview_failure",
            label="Failed preview",
            preview_status="failed",
        )
        failed_deployment = record_deployment(
            session,
            failed,
            target="preview",
            status="failed",
            url=None,
            details={"error": detail, "path": str(artifact_path)},
        )
        artifact = session.get(Artifact, failed.artifact_id)
        if artifact is None:
            return
        rollback_version_id = artifact.last_known_good_version_id or failed.parent_version_id
        rollback_error = None
        if rollback_version_id is not None:
            try:
                service.replace_with_version(
                    rollback_version_id,
                    artifact.path,
                    preserve_published=True,
                )
            except Exception as exc:
                rollback_error = str(exc) or exc.__class__.__name__
        if rollback_error:
            failed_details = dict(failed_deployment.details or {})
            failed_details["rollbackError"] = rollback_error
            failed_deployment.details = failed_details
            session.add(failed_deployment)
        elif rollback_version_id is not None:
            artifact.current_version_id = rollback_version_id
        session.add(artifact)
        session.commit()
    except Exception:
        logger.warning("Failed to record live preview failure for %s", artifact_path, exc_info=True)
        session.rollback()


class _PathBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    version_id: str | None = Field(default=None, validation_alias=AliasChoices("version_id", "versionId"))


class _CheckpointBody(BaseModel):
    path: str
    label: str | None = None
    operation_type: str = "checkpoint"
    prompt: str | None = None


class _RestoreVersionBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str | None = None
    artifact_id: str | None = Field(default=None, validation_alias=AliasChoices("artifact_id", "artifactId"))
    version_id: str = Field(validation_alias=AliasChoices("version_id", "versionId"))
    label: str | None = None
    create_checkpoint: bool = Field(
        default=False,
        validation_alias=AliasChoices("create_checkpoint", "createCheckpoint"),
    )


class _ForkVersionBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    version_id: str = Field(validation_alias=AliasChoices("version_id", "versionId"))
    name: str | None = None
    slug: str | None = None
    target_project_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("target_project_id", "targetProjectId", "project_id", "projectId"),
    )
    project: str | None = Field(default=None, validation_alias=AliasChoices("project", "targetProject"))


class _CommentBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    body: str | None = None
    text: str | None = None
    kind: str = "comment"
    anchor: dict | None = None
    proposed_patch: dict | None = Field(
        default=None,
        validation_alias=AliasChoices("proposed_patch", "proposedPatch"),
    )
    parent_comment_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("parent_comment_id", "parentCommentId"),
    )
    actor_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("actor_name", "actorName"),
    )


class _CommentReadBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    comment_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("comment_id", "commentId"),
    )
    activity_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("activity_id", "activityId"),
    )


class _HandoffBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str | None = None
    artifact_id: str | None = Field(default=None, validation_alias=AliasChoices("artifact_id", "artifactId"))
    version_id: str | None = Field(default=None, validation_alias=AliasChoices("version_id", "versionId"))
    comment_id: str | None = Field(default=None, validation_alias=AliasChoices("comment_id", "commentId"))
    prompt: str | None = None
    title: str | None = None
    project_id: UUID | None = Field(default=None, validation_alias=AliasChoices("project_id", "projectId"))
    project: str | None = None
    model: str | None = None


def _require_project_capability_if_owned(
    session: Session,
    project_id: UUID | None,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    if project_id is None:
        return
    try:
        require_project_permission_if_owned(
            session,
            project_id,
            principal.email if principal is not None else None,
            capability,
        )
    except ProjectPermissionError as exc:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required for this shared project",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to work with this artifact",
        ) from exc


def _require_artifact_capability(
    session: Session,
    artifact: Artifact,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    _require_project_capability_if_owned(session, artifact.project_id, principal, capability)


def _project_for_path(session: Session, path: str | Path | None) -> Project | None:
    if not path:
        return None
    try:
        requested = Path(path).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return None
    for project in ProjectService(session).list_projects():
        try:
            requested.relative_to(Path(project.path).resolve(strict=False))
            return project
        except (OSError, ValueError):
            continue
    return None


def _principal_can_project(
    session: Session,
    project: Project | None,
    principal: RequestPrincipal | None,
    capability: str,
) -> bool:
    if project is None:
        return True
    try:
        require_project_permission_if_owned(
            session,
            project.id,
            principal.email if principal is not None else None,
            capability,
        )
        return True
    except ProjectPermissionError:
        return False


def _redact_owner_publish_state(payload: dict) -> dict:
    redacted = dict(payload)
    if "accessPassword" in redacted:
        redacted["accessPassword"] = ""
    if "accessEmails" in redacted:
        redacted["accessEmails"] = []
    if "orgAllowed" in redacted:
        redacted["orgAllowed"] = False
    return redacted


def _sanitize_artifact_payload(
    session: Session,
    payload: dict,
    principal: RequestPrincipal | None,
    path: str | Path | None = None,
) -> dict:
    project = _project_for_path(session, path or payload.get("folder") or payload.get("path") or payload.get("artifactDir"))
    if _principal_can_project(session, project, principal, "edit"):
        return payload
    return _redact_owner_publish_state(payload)


def _deleted_artifact_card(
    session: Session,
    event: ArtifactActivityEvent,
    artifact: Artifact,
) -> dict | None:
    details = event.details or {}
    version_id = str(details.get("preDeleteVersionId") or event.version_id or "")
    version = None
    if version_id:
        try:
            version = session.get(ArtifactVersion, UUID(version_id))
        except ValueError:
            version = None
    external_id = str(details.get("externalArtifactId") or details.get("artifactId") or artifact.slug or artifact.id)
    # On delete the row's slug/path were tombstoned ("…#deleted-<uuid>"). Show
    # the originals: prefer the values captured in the event details, else strip
    # the tombstone suffix off the row's current values.
    def _undeleted(value: str | None) -> str | None:
        if not value:
            return value
        return value.split(_DELETED_TOMBSTONE_PREFIX, 1)[0]

    display_slug = details.get("slug") or _undeleted(artifact.slug)
    display_path = details.get("path") or _undeleted(artifact.path)
    return {
        "id": external_id,
        "artifactId": external_id,
        "internalArtifactId": str(artifact.id),
        "title": artifact.title or display_slug,
        "description": artifact.description or "",
        "type": artifact.artifact_type or "mixed",
        "kind": artifact.artifact_type or "Artifact",
        "slug": display_slug,
        "path": display_path,
        "folder": display_path,
        "projectId": str(artifact.project_id) if artifact.project_id else None,
        "deletedAt": event.created_at.isoformat() if event.created_at else None,
        "actorName": event.actor_name,
        "preDeleteVersionId": version_id or None,
        "versionId": version_id or None,
        "fileCount": version.file_count if version is not None else None,
        "totalBytes": version.total_bytes if version is not None else None,
        "restoreEligible": bool(version_id and version is not None),
    }


@router.get("/deleted")
async def list_deleted_artifacts(session: SessionDep, principal: PrincipalDep):
    events = session.exec(
        select(ArtifactActivityEvent)
        .where(ArtifactActivityEvent.event_type == "deleted")
        .order_by(ArtifactActivityEvent.created_at.desc())
    ).all()
    seen: set[UUID] = set()
    deleted: list[dict] = []
    for event in events:
        if event.artifact_id in seen:
            continue
        artifact = session.get(Artifact, event.artifact_id)
        if artifact is None:
            continue
        seen.add(artifact.id)
        if Path(artifact.path).exists():
            continue
        project = session.get(Project, artifact.project_id) if artifact.project_id else _project_for_path(session, artifact.path)
        if not _principal_can_project(session, project, principal, "view"):
            continue
        card = _deleted_artifact_card(session, event, artifact)
        if card is not None:
            deleted.append(card)
    return {"artifacts": deleted, "deleted": deleted}


def _require_path_project_capability(
    session: Session,
    path: str | Path | None,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    project = _project_for_path(session, path)
    if project is not None:
        _require_project_capability_if_owned(session, project.id, principal, capability)


def _artifact_for_comment(session: Session, comment_id: UUID) -> Artifact:
    comment = session.get(ArtifactComment, comment_id)
    if comment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")
    artifact = session.get(Artifact, comment.artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return artifact


def _handoff_target_project_id(session: Session, req: _HandoffBody) -> UUID | None:
    if req.project_id is not None:
        if session.get(Project, req.project_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")
        return req.project_id
    if req.project:
        project = session.exec(select(Project).where(Project.name == req.project)).first()
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")
        return project.id
    return None


def _target_project_for_fork(session: Session, req: _ForkVersionBody) -> Project | None:
    if req.target_project_id is not None:
        project = session.get(Project, req.target_project_id)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")
        return project
    project_ref = (req.project or "").strip()
    if not project_ref:
        return None
    project = session.exec(select(Project).where(Project.name == project_ref)).first()
    if project is not None:
        return project
    try:
        project = session.get(Project, UUID(project_ref))
        if project is not None:
            return project
    except ValueError:
        pass
    requested = Path(project_ref).expanduser().resolve(strict=False)
    for candidate in ProjectService(session).list_projects():
        if Path(candidate.path).expanduser().resolve(strict=False) == requested:
            return candidate
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")


@router.get("/")
async def list_artifacts(
    response: Response,
    session: SessionDep,
    principal: PrincipalDep,
    project_path: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Paginated artifact listing (newest-first).

    The body stays a bare JSON array of artifact cards (the page window) so
    existing consumers keep working unchanged. Pagination metadata rides on
    response headers — ``X-Total-Count`` (the honest total of artifacts the
    viewer can see), ``X-Offset``, ``X-Limit``, ``X-Has-More`` — so the library
    UI can render "showing N of M" + load-more instead of the old silent
    newest-80 truncation that dropped the rest with no signal.

    Per-artifact project visibility is enforced as a folder-level filter so the
    total and the page window only ever count what the viewer is allowed to see.
    """
    viewer_email = principal.email if principal is not None else None
    if project_path is not None:
        # Single project-level capability check covers the whole scoped tree;
        # no per-folder filtering needed.
        _require_path_project_capability(session, project_path, principal, "view")
        page = list_artifacts_page(
            project_path, viewer_email=viewer_email, limit=limit, offset=offset
        )
    else:
        # Global listing — filter by per-folder project visibility BEFORE
        # paginating so the total stays honest. Cache the folder→allowed decision
        # so a folder the pager touches twice (count pass + window build) only
        # resolves once.
        visibility_cache: dict[str, bool] = {}

        def _folder_visible(folder: Path) -> bool:
            key = str(folder)
            cached = visibility_cache.get(key)
            if cached is not None:
                return cached
            project = _project_for_path(session, folder)
            allowed = _principal_can_project(session, project, principal, "view")
            visibility_cache[key] = allowed
            return allowed

        page = list_artifacts_page(
            None,
            viewer_email=viewer_email,
            limit=limit,
            offset=offset,
            folder_filter=_folder_visible,
        )

    response.headers["X-Total-Count"] = str(page["total"])
    response.headers["X-Offset"] = str(page["offset"])
    response.headers["X-Limit"] = str(page["limit"])
    response.headers["X-Has-More"] = "true" if page["hasMore"] else "false"
    # Let the browser fetch() read the pagination headers cross-origin (web build
    # talks to the API over a different origin behind the auth proxy).
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Total-Count, X-Offset, X-Limit, X-Has-More"
    )
    return [_sanitize_artifact_payload(session, card, principal) for card in page["artifacts"]]


@router.post("/handoff", status_code=status.HTTP_201_CREATED)
async def handoff_artifact(req: _HandoffBody, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = _artifact_from_identifier_or_path(session, artifact_id=req.artifact_id, path=req.path)
        _require_artifact_capability(session, artifact, principal, "view")
        target_project_id = _handoff_target_project_id(session, req)
        if target_project_id is not None:
            _require_project_capability_if_owned(session, target_project_id, principal, "edit")
        return await handoff_artifact_to_conversation(
            session,
            path=req.path,
            artifact_id=req.artifact_id,
            version_id=req.version_id,
            comment_id=req.comment_id,
            prompt=req.prompt,
            title=req.title,
            project_id=target_project_id,
            project=req.project,
            model=req.model,
            **_actor_kwargs(principal),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(e))


@router.get("/versions")
def list_artifact_versions(session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    try:
        artifact = get_or_create_artifact_for_path(session, path)
        _require_artifact_capability(session, artifact, principal, "view")
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    versions = _list_versions(session, artifact)
    last_known_good_version_id = getattr(artifact, "last_known_good_version_id", None)
    return {
        "artifactId": str(artifact.id),
        "currentVersionId": str(artifact.current_version_id) if artifact.current_version_id else None,
        "lastKnownGoodVersionId": (
            str(last_known_good_version_id) if last_known_good_version_id else None
        ),
        "versions": [version_to_dict(version) for version in versions],
        "checkpoints": [version_to_dict(version) for version in versions],
    }


@router.post("/versions")
def create_artifact_checkpoint(req: _CheckpointBody, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = get_or_create_artifact_for_path(session, req.path)
        _require_artifact_capability(session, artifact, principal, "edit")
        version = _snapshot_artifact(
            session,
            req.path,
            operation_type=req.operation_type,
            label=req.label,
            prompt=req.prompt,
            **_actor_kwargs(principal),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"version": version_to_dict(version), "checkpoint": version_to_dict(version)}


@router.post("/versions/restore")
def restore_artifact_version(req: _RestoreVersionBody, session: SessionDep, principal: PrincipalDep):
    try:
        if req.path is None and req.artifact_id is None:
            raise ValueError("path or artifact_id is required")
        artifact = _artifact_from_identifier_or_path(session, artifact_id=req.artifact_id, path=req.path)
        _require_artifact_capability(session, artifact, principal, "edit")
        payload = _restore_artifact(
            session,
            path=req.path,
            artifact_id=req.artifact_id,
            version_id=req.version_id,
            body=req.model_dump(),
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if "checkpoint" not in payload and payload.get("version"):
        payload["checkpoint"] = payload["version"]
    return payload


@router.post("/versions/fork", status_code=status.HTTP_201_CREATED)
def fork_artifact_version(req: _ForkVersionBody, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = get_or_create_artifact_for_path(session, req.path)
        _require_artifact_capability(session, artifact, principal, "view")
        target_project = _target_project_for_fork(session, req)
        if target_project is None:
            _require_artifact_capability(session, artifact, principal, "edit")
        else:
            _require_project_capability_if_owned(session, target_project.id, principal, "edit")
        return _fork_version(
            session,
            req.version_id,
            path=req.path,
            body={
                "name": req.name,
                "slug": req.slug,
                "target_project_id": str(target_project.id) if target_project is not None else None,
                "project": req.project,
            },
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/diff")
def diff_artifact_versions(
    session: SessionDep,
    principal: PrincipalDep,
    request: Request,
    path: str = Query(...),
    from_version: str | None = Query(default=None, alias="from"),
    to_version: str | None = Query(default=None, alias="to"),
):
    try:
        artifact = get_or_create_artifact_for_path(session, path)
        _require_artifact_capability(session, artifact, principal, "view")
        versions = _list_versions(session, artifact)
        if not versions:
            raise ValueError("At least one checkpoint is required to diff")
        return {
            "available": True,
            **_diff_versions(
                session,
                path=path,
                base=from_version,
                compare=to_version or "current",
                request_base_url=str(request.base_url),
            ),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/comments")
def list_artifact_comments(session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    try:
        artifact = get_or_create_artifact_for_path(session, path)
        _require_artifact_capability(session, artifact, principal, "view")
        return _list_comments(
            session,
            artifact_id=artifact.id,
            viewer_email=principal.email if principal is not None else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/comments", status_code=status.HTTP_201_CREATED)
def create_artifact_comment(req: _CommentBody, session: SessionDep, principal: PrincipalDep):
    text = (req.body or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Comment body is required")
    try:
        artifact = get_or_create_artifact_for_path(session, req.path)
        capability = "review" if req.kind in {"suggestion", "review"} else "comment"
        _require_artifact_capability(session, artifact, principal, capability)
        return _create_comment(
            session,
            path=req.path,
            body=text,
            kind=req.kind,
            anchor=req.anchor or {},
            proposed_patch=req.proposed_patch,
            parent_comment_id=req.parent_comment_id,
            actor_name=(principal.name if principal is not None and principal.name else req.actor_name),
            actor_email=principal.email if principal is not None else None,
            actor_subject=principal.subject if principal is not None else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/comments/read")
def mark_artifact_comments_read(req: _CommentReadBody, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = get_or_create_artifact_for_path(session, req.path)
        _require_artifact_capability(session, artifact, principal, "view")
        return _mark_comments_read(
            session,
            path=req.path,
            comment_id=req.comment_id,
            activity_id=req.activity_id,
            viewer_email=principal.email if principal is not None else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/comments/{comment_id}/resolve")
def resolve_artifact_comment(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "review")
        return _set_comment_status(
            session,
            comment_id,
            status="resolved",
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/comments/{comment_id}/reopen")
def reopen_artifact_comment(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "review")
        return _set_comment_status(
            session,
            comment_id,
            status="open",
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/comments/{comment_id}/accept")
def accept_artifact_suggestion(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "edit")
        return _set_comment_status(
            session,
            comment_id,
            status="accepted",
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        detail = str(e)
        error_status = status.HTTP_400_BAD_REQUEST if "older artifact version" in detail else status.HTTP_404_NOT_FOUND
        raise HTTPException(status_code=error_status, detail=detail)


@router.post("/comments/{comment_id}/reject")
def reject_artifact_suggestion(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "review")
        return _set_comment_status(
            session,
            comment_id,
            status="rejected",
            **_actor_kwargs(principal),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/comments/{comment_id}/preview")
def preview_artifact_suggestion_patch(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "review")
        return _preview_comment_patch(session, comment_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/comments/{comment_id}/apply")
def apply_artifact_suggestion_patch(comment_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, _artifact_for_comment(session, comment_id), principal, "edit")
        return _apply_comment_patch(
            session,
            comment_id,
            **_actor_kwargs(principal),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/preview")
async def preview_artifact(
    session: SessionDep,
    principal: PrincipalDep,
    path: str = Query(...),
    version_id: str | None = Query(default=None),
):
    try:
        artifact = resolve_artifact_path(path)
        _require_path_project_capability(session, artifact, principal, "view")
        preview_path = _materialized_version_preview_path(session, artifact, version_id) or artifact
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        payload = _preview_artifact(preview_path)
        if not version_id:
            _mark_live_preview_ready(session, artifact, payload)
        return payload
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        if not version_id:
            _record_live_preview_failure(session, artifact, str(e) or "Could not read artifact")
        raise HTTPException(status_code=500, detail="Could not read artifact") from e


class _ExportBody(BaseModel):
    path: str
    format: str  # 'pdf' | 'docx' | 'html'


@router.post("/export")
async def export_artifact_endpoint(req: _ExportBody):
    """Convert a document artifact (markdown/HTML) to PDF/Word/HTML, writing
    the result into the same artifact folder. Returns the new file's path so
    the client can open it (desktop) plus a signed origin-relative ``serveUrl``
    so the client can download it (web, where the file isn't on this machine)."""
    from fastapi.concurrency import run_in_threadpool

    from cowork.services.artifact_export import ExportError, export_artifact

    try:
        source = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        out = await run_in_threadpool(export_artifact, source, req.format)
    except ExportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Export failed") from e
    return {"path": str(out), "filename": out.name, "serveUrl": serve_url_for(out)}


@router.post("/preview-mount")
async def preview_mount_endpoint(req: _PathBody, request: Request, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = resolve_artifact_path(req.path)
        _require_path_project_capability(session, artifact, principal, "view")
        preview_path = _materialized_version_preview_path(session, artifact, req.version_id) or artifact
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        payload = await mount_preview(preview_path)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        if not req.version_id:
            _record_live_preview_failure(session, artifact, str(e) or "Could not preview artifact")
        raise HTTPException(status_code=500, detail="Could not preview artifact") from e
    payload = _sanitize_artifact_payload(session, payload, principal, path=artifact)

    if payload.get("kind") == "proxy":
        # Build the absolute proxy URL from the incoming request. Using
        # scheme+netloc means the iframe loads through the same host
        # the client used to reach us — works equally for desktop
        # (127.0.0.1:port) and cloud (reverse-proxy origin).
        token = payload["token"]
        payload["proxyUrl"] = (
            f"{request.url.scheme}://{request.url.netloc}"
            f"/api/v1/artifacts/proxy/{token}/"
        )
        if payload.get("backendRunning") is False:
            if not req.version_id:
                _record_live_preview_failure(
                    session,
                    artifact,
                    payload.get("launchError") or "Backend failed to start",
                )
            return payload
    if not req.version_id:
        _mark_live_preview_ready(session, artifact, payload)
    return payload


@router.get("/preview-asset/{token}/{rel_path:path}")
async def preview_asset(token: str, rel_path: str):
    parent = get_preview_mount(token)
    if parent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview mount has expired or is unknown")
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Asset is outside the artifact directory")
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, headers=_browser_file_headers())


@router.get("/serve/{project_name}/{file_path:path}")
def serve_artifact_file(
    request: Request,
    session: SessionDep,
    project_name: str,
    file_path: str,
    token: str | None = Query(default=None),
):
    """Serve a file from `<project>/.anton/artifacts/<file_path>` over
    HTTP. Stateless, origin-relative, frame-able so the in-app iframe
    and new-tab open both work in web deployments."""
    base = _project_artifacts_base(project_name)
    if base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    try:
        target = (base / file_path).resolve()
        target.relative_to(base.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    if not verify_serve_url_token(target, token):
        principal = get_request_principal(request)
        project = ProjectService(session).get_project_by_name_or_none(project_name)
        _require_project_capability_if_owned(
            session,
            project.id if project is not None else None,
            principal,
            "view",
        )
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, headers=_browser_file_headers())


@router.post("/open")
async def open_artifact(req: _PathBody, session: SessionDep, principal: PrincipalDep):
    try:
        artifact = resolve_artifact_path(req.path)
        _require_path_project_capability(session, artifact, principal, "view")
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        subprocess.run(["open", str(artifact)], check=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not open artifact") from exc
    return {"status": "ok", "path": str(artifact)}


def _resolve_reveal_path(path: str, session: Session) -> Path:
    try:
        return resolve_artifact_path(path)
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    try:
        requested = Path(path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path") from exc
    for project in ProjectService(session).list_projects():
        project_dir = Path(project.path).resolve()
        try:
            requested.relative_to(project_dir)
        except ValueError:
            continue
        if requested.exists():
            return requested
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path is not in a known project or artifact directory")


@router.post("/reveal")
async def reveal_artifact(req: _PathBody, session: SessionDep, principal: PrincipalDep):
    target = _resolve_reveal_path(req.path, session)
    _require_path_project_capability(session, target, principal, "view")
    try:
        reveal_in_file_manager(target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(target)}


@router.api_route(
    "/proxy/{token}/{rel_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(token: str, rel_path: str, request: Request):
    """HTTP forwarder for fullstack-artifact previews.

    Streams the request to the artifact's backend running on
    `127.0.0.1:<metadata.json port>`, injects CORS, strips hop-by-hop
    headers. See `cowork.services.preview_proxy` for the body.
    """
    from cowork.services.preview_proxy import proxy_artifact_request
    return await proxy_artifact_request(token, rel_path, request)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
def delete_artifact_endpoint(session: SessionDep, principal: PrincipalDep, path: str = Query(...)):
    pending_delete: Path | None = None
    artifact_folder: Path | None = None
    committed = False
    try:
        artifact = get_or_create_artifact_for_path(session, path)
        _require_artifact_capability(session, artifact, principal, "manage")
        metadata = _load_metadata(Path(artifact.path)) or {}
        external_artifact_id = str(metadata.get("id") or metadata.get("slug") or artifact.slug or artifact.id)
        pre_delete = _snapshot_artifact(
            session,
            path,
            operation_type="pre_delete",
            label="Before delete",
            **_actor_kwargs(principal),
        )
        artifact_folder = Path(artifact.path).expanduser().resolve(strict=False)
        if not artifact_folder.is_dir():
            raise FileNotFoundError("Artifact not found")
        _unpublish_folder(artifact_folder)
        pending_delete = artifact_folder.parent / f".{artifact_folder.name}.delete-{uuid4().hex}"
        os.replace(artifact_folder, pending_delete)
        session.add(
            ArtifactActivityEvent(
                artifact_id=artifact.id,
                version_id=pre_delete.id,
                event_type="deleted",
                actor_name=principal.name if principal is not None else None,
                details={
                    "artifactId": external_artifact_id,
                    "externalArtifactId": external_artifact_id,
                    "slug": artifact.slug,
                    "path": artifact.path,
                    "preDeleteVersionId": str(pre_delete.id),
                    "notificationDeliveries": [
                        delivery_to_dict(delivery)
                        for delivery in dispatch_project_notification(
                            session,
                            artifact,
                            "deleted",
                            details={
                                "artifactId": external_artifact_id,
                                "externalArtifactId": external_artifact_id,
                                "versionId": str(pre_delete.id),
                                "preDeleteVersionId": str(pre_delete.id),
                                "slug": artifact.slug,
                                "path": artifact.path,
                            },
                        )
                    ],
                },
            )
        )
        # Release the folder path AND slug so a NEW artifact created at the same
        # path/name starts fresh, instead of re-attaching to this (now-deleted)
        # record and inheriting its history. BOTH unique keys must be freed —
        # uq_artifacts_path AND uq_artifacts_project_slug — or the re-create 500s on
        # an integrity error. The originals are preserved in the "deleted" event
        # details above, so recovery can restore them (see restore_artifact).
        tombstone = uuid4().hex
        artifact.path = _tombstone(artifact.path, tombstone, 2048)
        if artifact.slug:
            artifact.slug = _tombstone(artifact.slug, tombstone, 255)
        session.add(artifact)
        session.commit()
        committed = True
    except FileNotFoundError as e:
        session.rollback()
        _restore_pending_delete(artifact_folder, pending_delete)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        session.rollback()
        _restore_pending_delete(artifact_folder, pending_delete)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        session.rollback()
        _restore_pending_delete(artifact_folder, pending_delete)
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e
    finally:
        if committed and pending_delete is not None:
            shutil.rmtree(pending_delete, ignore_errors=True)


def _restore_pending_delete(artifact_folder: Path | None, pending_delete: Path | None) -> None:
    if artifact_folder is None or pending_delete is None or not pending_delete.exists():
        return
    if artifact_folder.exists():
        return
    os.replace(pending_delete, artifact_folder)
