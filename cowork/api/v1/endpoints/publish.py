"""Publish API endpoints — publish/unpublish HTML artifacts to 4nton.ai.

Ported from cowork/server/routes/utilities.py (publish section).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlmodel import Session, select

from cowork.db.session import get_session
from cowork.models.artifact import Artifact, ArtifactVersion
from cowork.models.project import Project
from cowork.services.artifact_versions import ArtifactVersionService, record_deployment, snapshot_artifact
from cowork.services.artifacts import _artifact_root_for, resolve_artifact_path
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned
from cowork.services.publish import (
    list_publishable,
    publish_artifact as _publish,
    unpublish_artifact as _unpublish,
)
from cowork.services.request_identity import (
    AuthenticationError,
    RequestPrincipal,
    principal_from_authorization_header,
)

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def get_request_principal(request: Request) -> RequestPrincipal | None:
    try:
        return principal_from_authorization_header(request.headers.get("authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


PrincipalDep = Annotated[RequestPrincipal | None, Depends(get_request_principal)]


def _actor_kwargs(principal: RequestPrincipal | None) -> dict[str, str | None]:
    return {
        "actor_name": principal.name if principal is not None else None,
        "actor_email": principal.email if principal is not None else None,
        "actor_subject": principal.subject if principal is not None else None,
    }


class _AccessBody(BaseModel):
    # Mutually exclusive publish modes (ENG-322):
    #   public     — anyone with the link
    #   password   — visitors must enter `password`
    #   restricted — only `emails` and/or everyone in the owner's org
    mode: Literal["public", "password", "restricted"] = "public"
    password: str | None = None
    emails: list[str] = []
    org_allowed: bool = False


class _PublishBody(BaseModel):
    path: str
    versionId: str | None = None
    version_id: str | None = None
    # Back-compat: a bare top-level password still publishes password-protected.
    # New clients send the structured `access` object instead. Only a hash (and,
    # for restricted, the email list) leaves this machine; plaintext stays in
    # .published.json for the in-app reveal.
    password: str | None = None
    access: _AccessBody | None = None


@router.get("/")
async def list_publishable_endpoint(session: SessionDep, principal: PrincipalDep):
    payload = list_publishable()
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if isinstance(artifacts, list):
        payload["artifacts"] = [
            _sanitize_publish_payload(session, item, principal)
            for item in artifacts
            if isinstance(item, dict) and _principal_can_path(session, _path_from_payload(item), principal, "view")
        ]
    history = payload.get("history") if isinstance(payload, dict) else None
    if isinstance(history, list):
        payload["history"] = [
            item
            for item in history
            if not isinstance(item, dict) or _principal_can_path(session, _path_from_payload(item), principal, "view")
        ]
    return payload


@router.post("/")
async def publish_artifact(req: _PublishBody, session: SessionDep, principal: PrincipalDep):
    _require_publish_capability(session, req.path, principal)
    version = None
    published_payload = None
    requested_version_id = req.version_id or req.versionId
    publishing_selected_version = bool(requested_version_id)
    try:
        if requested_version_id:
            version = _resolve_publish_version(session, req.path, requested_version_id)
        else:
            version = snapshot_artifact(
                session,
                req.path,
                operation_type="publish",
                label="Published version",
                **_actor_kwargs(principal),
            )
        version_metadata = _version_publish_metadata(version)
        if version is not None and not isinstance(version, dict):
            with TemporaryDirectory(prefix="cowork-publish-version-") as tmp:
                publish_source = str(Path(tmp) / "artifact")
                ArtifactVersionService(session).materialize_version(version.id, publish_source, clean=True)
                _copy_publish_housekeeping(session, version, Path(publish_source))
                published_payload = _publish_with_version_metadata(
                    req.path,
                    req.password,
                    access=req.access.model_dump() if req.access else None,
                    version_metadata=version_metadata,
                    publish_source_path=publish_source,
                )
        else:
            published_payload = _publish_with_version_metadata(
                req.path,
                req.password,
                access=req.access.model_dump() if req.access else None,
                version_metadata=version_metadata,
                publish_source_path=None,
            )
        if version_metadata and isinstance(published_payload, dict):
            published_payload.setdefault("publishedVersionId", version_metadata["id"])
            published_payload.setdefault("publishedFilesHash", version_metadata["filesHash"])
            published_payload.setdefault("publishedManifestHash", version_metadata["manifestHash"])
            published_payload.setdefault("publishedVersionNumber", version_metadata["versionNumber"])
        if version is not None and not isinstance(version, dict):
            record_deployment(
                session,
                version,
                target="publish",
                status="published",
                url=published_payload.get("url") or published_payload.get("publishedUrl"),
                details={"access": published_payload.get("access") or {}},
                **_actor_kwargs(principal),
            )
        return published_payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if version is not None and not isinstance(version, dict):
            try:
                failed_deployment = record_deployment(
                    session,
                    version,
                    target="publish",
                    status="failed",
                    url=(published_payload or {}).get("url") if isinstance(published_payload, dict) else None,
                    details={"error": detail},
                    **_actor_kwargs(principal),
                )
                artifact = session.get(Artifact, version.artifact_id)
                if artifact is not None:
                    rollback_version_id = artifact.last_known_good_version_id or version.parent_version_id
                    rollback_error = None
                    if rollback_version_id is not None and not publishing_selected_version:
                        try:
                            ArtifactVersionService(session).replace_with_version(
                                rollback_version_id,
                                artifact.path,
                                preserve_published=True,
                            )
                        except Exception as rollback_exc:
                            rollback_error = str(rollback_exc) or rollback_exc.__class__.__name__
                    if rollback_error:
                        details = dict(failed_deployment.details or {})
                        details["rollbackError"] = rollback_error
                        failed_deployment.details = details
                        session.add(failed_deployment)
                        session.commit()
                    elif rollback_version_id is not None and not publishing_selected_version:
                        artifact.current_version_id = rollback_version_id
                        session.add(artifact)
                        session.commit()
            except Exception:
                session.rollback()
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)


@router.delete("/")
async def unpublish_artifact(
    session: SessionDep,
    principal: PrincipalDep,
    path: str = Query(..., description="Absolute path to the published HTML artifact"),
):
    _require_publish_capability(session, path, principal)
    try:
        payload = _unpublish(path)
        _record_unpublish_deployment(session, path, payload, principal)
        return payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)


def _record_unpublish_deployment(
    session: Session,
    path: str,
    payload: dict[str, Any],
    principal: RequestPrincipal | None,
) -> None:
    version_ref = payload.get("publishedVersionId") if isinstance(payload, dict) else None
    if not version_ref:
        return
    try:
        version = _resolve_publish_version(session, path, str(version_ref))
    except (FileNotFoundError, ValueError):
        return
    record_deployment(
        session,
        version,
        target="publish",
        status="unpublished",
        url=payload.get("publishedUrl") or None,
        details={
            "previousPublishedVersionId": str(version.id),
            "filesHash": payload.get("publishedFilesHash") or version.files_hash,
            "manifestHash": payload.get("publishedManifestHash") or version.manifest_hash,
            "versionNumber": payload.get("publishedVersionNumber") or version.version_number,
        },
        **_actor_kwargs(principal),
    )


def _require_publish_capability(
    session: Session,
    path: str,
    principal: RequestPrincipal | None,
) -> None:
    project = _project_for_path(session, path)
    if project is None:
        return
    try:
        require_project_permission_if_owned(
            session,
            project.id,
            principal.email if principal is not None else None,
            "edit",
        )
    except ProjectPermissionError as exc:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required for this shared project",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to publish this artifact",
        ) from exc


def _project_for_path(session: Session, path: str | Path | None) -> Project | None:
    if path is None:
        return None
    try:
        requested = Path(path).expanduser().resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return None
    for project in session.exec(select(Project)).all():
        try:
            requested.relative_to(Path(project.path).expanduser().resolve(strict=False))
            return project
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def _principal_can_path(
    session: Session,
    path: str | Path | None,
    principal: RequestPrincipal | None,
    capability: str,
) -> bool:
    project = _project_for_path(session, path)
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


def _path_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("path", "artifact", "file", "artifactPath"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _sanitize_publish_payload(
    session: Session,
    payload: dict[str, Any],
    principal: RequestPrincipal | None,
) -> dict[str, Any]:
    if _principal_can_path(session, _path_from_payload(payload), principal, "edit"):
        return payload
    redacted = dict(payload)
    if "accessPassword" in redacted:
        redacted["accessPassword"] = ""
    if "accessEmails" in redacted:
        redacted["accessEmails"] = []
    if "orgAllowed" in redacted:
        redacted["orgAllowed"] = False
    return redacted


def _version_publish_metadata(version: Any) -> dict[str, Any] | None:
    if version is None or isinstance(version, dict):
        return None
    return {
        "id": str(version.id),
        "artifactId": str(version.artifact_id),
        "filesHash": version.files_hash,
        "manifestHash": version.manifest_hash,
        "versionNumber": version.version_number,
    }


def _resolve_publish_version(session: Session, path: str, version_ref: str) -> ArtifactVersion:
    try:
        version_id = UUID(str(version_ref))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid versionId") from exc
    version = session.get(ArtifactVersion, version_id)
    if version is None:
        raise FileNotFoundError("Artifact version not found")
    artifact = session.get(Artifact, version.artifact_id)
    if artifact is None:
        raise FileNotFoundError("Artifact not found for version")

    requested = resolve_artifact_path(path, allow_dir=True)
    if requested is None:
        raise FileNotFoundError("Artifact not found")
    requested_root = requested if requested.is_dir() else _artifact_root_for(requested)
    if requested_root.resolve(strict=False) != Path(artifact.path).resolve(strict=False):
        raise FileNotFoundError("Artifact version not found for this artifact")
    return version


def _publish_with_version_metadata(
    path: str,
    password: str | None,
    *,
    access: dict[str, Any] | None,
    version_metadata: dict[str, Any] | None,
    publish_source_path: str | None,
) -> dict[str, Any]:
    if version_metadata is None and publish_source_path is None:
        return _publish(path, password, access=access)
    optional_kwargs: dict[str, Any] = {}
    if version_metadata is not None:
        optional_kwargs["version_metadata"] = version_metadata
    if publish_source_path is not None:
        optional_kwargs["publish_source_path"] = publish_source_path
    try:
        signature = inspect.signature(_publish)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        call_kwargs = {
            key: value
            for key, value in optional_kwargs.items()
            if key in signature.parameters or accepts_kwargs
        }
        if call_kwargs:
            return _publish(path, password, access=access, **call_kwargs)
    except (TypeError, ValueError):
        pass
    return _publish(path, password, access=access)


def _copy_publish_housekeeping(session: Session, version: Any, publish_source: Path) -> None:
    ArtifactVersionService(session).write_version_housekeeping(version.id, publish_source)
    artifact = session.get(Artifact, version.artifact_id)
    if artifact is None:
        return
    source_root = Path(artifact.path)
    for name in ("metadata.json", "README.md"):
        if (publish_source / name).is_file():
            continue
        source = source_root / name
        if not source.is_file():
            continue
        target = publish_source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
