"""Artifact version/checkpoint API endpoints.

The artifact-version worker owns the long-term foundation service. This
endpoint layer is intentionally shaped around that service name, while keeping
an endpoint-local filesystem implementation so the v1 contract is useful in
tests and local development before the foundation lands.
"""
from __future__ import annotations

import difflib
import hashlib
import importlib
import inspect
import json
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import AliasChoices, BaseModel, Field
from sqlmodel import Session, select

from cowork.db.session import get_session
from cowork.models.artifact import Artifact, ArtifactActivityEvent
from cowork.models.project import Project
from cowork.services.artifacts import (
    KIND_BY_EXT,
    KIND_BY_TYPE,
    TEXT_EXTENSIONS,
    _HOUSEKEEPING_FILES,
    _load_metadata,
    _pick_primary,
    _published_access_for,
    _published_url_for,
    _scan_artifact_dirs,
    _user_files,
    resolve_artifact_path,
    serve_url_for,
)
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned
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


class _ArtifactPathBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str | None = None
    title: str | None = None
    description: str | None = None
    artifact_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("artifact_type", "artifactType", "type"),
    )
    primary: str | None = None


class _CheckpointBody(_ArtifactPathBody):
    label: str | None = None
    summary: str | None = None
    prompt: str | None = None
    source_conversation_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("source_conversation_id", "sourceConversationId", "conversation_id", "conversationId"),
    )
    source_message_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("source_message_id", "sourceMessageId", "message_id", "messageId"),
    )
    kind: Literal["manual", "auto", "preview", "publish", "restore"] = "manual"


class _RestoreBody(_ArtifactPathBody):
    version_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("version_id", "versionId", "checkpoint_id", "checkpointId"),
    )
    create_checkpoint: bool = Field(
        default=False,
        validation_alias=AliasChoices("create_checkpoint", "createCheckpoint"),
    )


class _ForkBody(_ArtifactPathBody):
    version_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("version_id", "versionId", "checkpoint_id", "checkpointId"),
    )
    slug: str | None = None
    target_project_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("target_project_id", "targetProjectId", "project_id", "projectId"),
    )
    project: str | None = Field(default=None, validation_alias=AliasChoices("project", "targetProject"))


_VERSION_INDEX = "versions.json"
_SNAPSHOT_DIR = "snapshots"
_SNAPSHOT_EXCLUDED_TOPS = {
    *_HOUSEKEEPING_FILES,
    ".cowork_versions",
    ".DS_Store",
}
_TEXT_DIFF_LIMIT = 200_000


@router.get("/{artifact_id}/versions")
async def list_versions(
    artifact_id: str,
    session: SessionDep,
    principal: PrincipalDep,
    path: str | None = Query(default=None, description="Optional artifact folder/file path fallback"),
):
    _require_artifact_capability(session, artifact_id=artifact_id, path=path, principal=principal, capability="view")
    foundation = await _call_foundation(
        "list",
        session=session,
        artifact_id=artifact_id,
        path=path,
    )
    if foundation is not None:
        try:
            folder = _resolve_artifact_folder(artifact_id, path=path)
        except HTTPException:
            return _sanitize_artifact_payload(
                session,
                artifact_id=artifact_id,
                path=path,
                principal=principal,
                payload=foundation,
            )
        return _sanitize_artifact_payload(
            session,
            artifact_id=artifact_id,
            path=path,
            principal=principal,
            payload=_merge_integration(folder, foundation),
        )

    folder = _resolve_artifact_folder(artifact_id, path=path)
    return _sanitize_artifact_payload(
        session,
        artifact_id=artifact_id,
        path=path,
        principal=principal,
        payload=_list_versions_fallback(folder),
    )


@router.post("/{artifact_id}/checkpoints", status_code=status.HTTP_201_CREATED)
async def create_checkpoint(
    artifact_id: str,
    body: _CheckpointBody,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_artifact_capability(
        session,
        artifact_id=artifact_id,
        path=body.path,
        principal=principal,
        capability="edit",
        create=bool(body.path),
    )
    foundation = await _call_foundation(
        "checkpoint",
        session=session,
        artifact_id=artifact_id,
        path=body.path,
        body=body.model_dump(),
    )
    if foundation is not None:
        folder = _resolve_artifact_folder(artifact_id, path=body.path)
        return _merge_integration(folder, foundation)

    folder = _resolve_artifact_folder(
        artifact_id,
        path=body.path,
        create=True,
        title=body.title,
        description=body.description,
        artifact_type=body.artifact_type,
        primary=body.primary,
    )
    return _create_checkpoint_fallback(folder, label=body.label, summary=body.summary, kind=body.kind)


@router.post("/{artifact_id}/restore")
async def restore_version(
    artifact_id: str,
    body: _RestoreBody,
    session: SessionDep,
    principal: PrincipalDep,
):
    version_id = (body.version_id or "").strip()
    if not version_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="version_id is required")
    _require_artifact_capability(session, artifact_id=artifact_id, path=body.path, principal=principal, capability="edit")

    foundation = await _call_foundation(
        "restore",
        session=session,
        artifact_id=artifact_id,
        path=body.path,
        version_id=version_id,
        body=body.model_dump(),
    )
    if foundation is not None:
        folder = _resolve_artifact_folder(artifact_id, path=body.path)
        return _merge_integration(folder, foundation)

    folder = _resolve_artifact_folder(artifact_id, path=body.path)
    try:
        created = None
        if body.create_checkpoint:
            created = _create_checkpoint_fallback(
                folder,
                label=f"Before restore to {version_id}",
                summary="Automatic checkpoint created before restore",
                kind="restore",
            )
        restored = _restore_version_fallback(folder, version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    payload = {
        "status": "ok",
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
        "restoredVersion": restored,
        **_artifact_integration(folder),
    }
    if created is not None:
        payload["createdCheckpoint"] = created["version"]
    return payload


@router.post("/{artifact_id}/fork", status_code=status.HTTP_201_CREATED)
async def fork_version(
    artifact_id: str,
    body: _ForkBody,
    session: SessionDep,
    principal: PrincipalDep,
):
    version_id = (body.version_id or "").strip()
    if not version_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="version_id is required")
    _require_artifact_capability(session, artifact_id=artifact_id, path=body.path, principal=principal, capability="view")
    target_project = _target_project_for_fork(session, body)
    if target_project is None:
        _require_artifact_capability(session, artifact_id=artifact_id, path=body.path, principal=principal, capability="edit")
    else:
        _require_project_capability(session, target_project.id, principal, "edit")

    foundation = await _call_foundation(
        "fork",
        session=session,
        artifact_id=artifact_id,
        path=body.path,
        version_id=version_id,
        body={
            **body.model_dump(),
            "target_project_id": str(target_project.id) if target_project is not None else None,
        },
    )
    if foundation is not None:
        folder = _resolve_artifact_folder(artifact_id, path=body.path)
        return _merge_integration(folder, foundation)

    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Forking requires artifact version service")


@router.get("/{artifact_id}/diff")
async def diff_versions(
    artifact_id: str,
    request: Request,
    session: SessionDep,
    principal: PrincipalDep,
    base: str | None = Query(default=None, description="Base checkpoint id; defaults to latest checkpoint"),
    compare: str | None = Query(default="current", description="Compare checkpoint id, latest, or current"),
    kind: Literal["manifest", "text"] = Query(default="manifest"),
    path: str | None = Query(default=None, description="Optional artifact folder/file path fallback"),
):
    _require_artifact_capability(session, artifact_id=artifact_id, path=path, principal=principal, capability="view")
    foundation = await _call_foundation(
        "diff",
            session=session,
            artifact_id=artifact_id,
            path=path,
            base=base,
            compare=compare,
            kind=kind,
            request_base_url=str(request.base_url),
        )
    if foundation is not None:
        folder = _resolve_artifact_folder(artifact_id, path=path)
        return _sanitize_artifact_payload(
            session,
            artifact_id=artifact_id,
            path=path,
            principal=principal,
            payload=_merge_integration(folder, foundation),
        )

    folder = _resolve_artifact_folder(artifact_id, path=path)
    try:
        return _sanitize_artifact_payload(
            session,
            artifact_id=artifact_id,
            path=path,
            principal=principal,
            payload=_diff_versions_fallback(folder, base=base, compare=compare or "current", kind=kind),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


async def _call_foundation(operation: str, **kwargs) -> Any | None:
    """Call the foundation service if present.

    The foundation branch is deliberately conservative: it supports the likely
    module-function names and an ``ArtifactVersionService(session)`` class, and
    falls back only when the service/module is absent or does not expose that
    operation. Runtime errors from a present service propagate to the endpoint.
    """
    try:
        service_module = importlib.import_module("cowork.services.artifact_versions")
    except ImportError:
        return None

    candidates = {
        "list": ("list_versions", "list_artifact_versions"),
        "checkpoint": ("create_checkpoint", "checkpoint_artifact", "snapshot_artifact"),
        "restore": ("restore_checkpoint", "restore_artifact", "restore_version"),
        "fork": ("fork_version", "fork_artifact_version"),
        "diff": ("diff_versions", "diff_artifact_versions", "diff_artifact"),
    }[operation]

    service_obj = None
    service_cls = getattr(service_module, "ArtifactVersionService", None)
    if service_cls is not None:
        try:
            service_obj = service_cls(kwargs.get("session"))
        except TypeError:
            service_obj = service_cls()

    targets = [service_module, service_obj] if service_obj is not None else [service_module]
    for target in targets:
        for name in candidates:
            fn = getattr(target, name, None)
            if fn is None:
                continue
            try:
                result = _invoke_compatible(fn, kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except FileNotFoundError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except ValueError as exc:
                code = status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST
                raise HTTPException(status_code=code, detail=str(exc)) from exc
            return result
    return None


def _require_artifact_capability(
    session: Session,
    *,
    artifact_id: str,
    path: str | None,
    principal: RequestPrincipal | None,
    capability: str,
    create: bool = False,
) -> None:
    project_id = _project_id_for_artifact_request(session, artifact_id=artifact_id, path=path, create=create)
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


def _require_project_capability(
    session: Session,
    project_id: UUID,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
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
            detail="You do not have permission to work with this project",
        ) from exc


def _target_project_for_fork(session: Session, body: _ForkBody) -> Project | None:
    if body.target_project_id is not None:
        project = session.get(Project, body.target_project_id)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")
        return project

    project_ref = (body.project or "").strip()
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
    for candidate in session.exec(select(Project)).all():
        if Path(candidate.path).expanduser().resolve(strict=False) == requested:
            return candidate
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target project not found")


def _artifact_capability_allowed(
    session: Session,
    *,
    artifact_id: str,
    path: str | None,
    principal: RequestPrincipal | None,
    capability: str,
) -> bool:
    try:
        project_id = _project_id_for_artifact_request(session, artifact_id=artifact_id, path=path, create=False)
    except HTTPException:
        return False
    if project_id is None:
        return True
    try:
        require_project_permission_if_owned(
            session,
            project_id,
            principal.email if principal is not None else None,
            capability,
        )
        return True
    except ProjectPermissionError:
        return False


def _redact_owner_publish_state(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    redacted = dict(payload)
    for container in (redacted, redacted.get("publish"), redacted.get("artifact")):
        if not isinstance(container, dict):
            continue
        if "accessPassword" in container:
            container["accessPassword"] = ""
        if "accessEmails" in container:
            container["accessEmails"] = []
        if "orgAllowed" in container:
            container["orgAllowed"] = False
    return redacted


def _sanitize_artifact_payload(
    session: Session,
    *,
    artifact_id: str,
    path: str | None,
    principal: RequestPrincipal | None,
    payload: Any,
) -> Any:
    if _artifact_capability_allowed(
        session,
        artifact_id=artifact_id,
        path=path,
        principal=principal,
        capability="edit",
    ):
        return payload
    return _redact_owner_publish_state(payload)


def _project_id_for_artifact_request(
    session: Session,
    *,
    artifact_id: str,
    path: str | None,
    create: bool,
) -> UUID | None:
    if path:
        project = _project_for_path(session, path)
        if project is not None:
            return project.id
        if create:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Artifact path must be inside a known project artifacts folder",
            )

    artifact = _artifact_by_identifier(session, artifact_id)
    if artifact is not None:
        if artifact.project_id is not None:
            return artifact.project_id
        project = _project_for_path(session, artifact.path)
        if path and project is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Artifact path must be inside a known project artifacts folder",
            )
        return project.id if project is not None else None

    folder = _resolve_artifact_folder(artifact_id, path=path)
    project = _project_for_path(session, folder)
    if path and project is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Artifact path must be inside a known project artifacts folder",
        )
    return project.id if project is not None else None


def _artifact_by_identifier(session: Session, artifact_id: str | UUID | None) -> Artifact | None:
    if artifact_id is None:
        return None
    if isinstance(artifact_id, UUID):
        return session.get(Artifact, artifact_id)
    clean = str(artifact_id)
    try:
        found = session.get(Artifact, UUID(clean))
        if found is not None:
            return found
    except ValueError:
        pass
    artifact = session.exec(select(Artifact).where(Artifact.slug == clean)).first()
    if artifact is not None:
        return artifact
    events = session.exec(
        select(ArtifactActivityEvent)
        .where(ArtifactActivityEvent.event_type == "deleted")
        .order_by(ArtifactActivityEvent.created_at.desc())
    ).all()
    for event in events:
        details = event.details or {}
        identifiers = {
            str(details.get("externalArtifactId") or ""),
            str(details.get("artifactId") or ""),
            str(details.get("slug") or ""),
        }
        if clean not in identifiers:
            continue
        artifact = session.get(Artifact, event.artifact_id)
        if artifact is not None:
            return artifact
    return None


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


def _invoke_compatible(fn, values: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    params = signature.parameters
    call_kwargs = {
        name: value
        for name, value in values.items()
        if name in params and value is not None
    }
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        call_kwargs = {name: value for name, value in values.items() if value is not None}
    try:
        return fn(**call_kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "cowork.services.artifact_versions is present but its endpoint "
            f"adapter for {getattr(fn, '__name__', 'operation')} is incompatible"
        ) from exc


def _resolve_artifact_folder(
    artifact_id: str,
    *,
    path: str | None = None,
    create: bool = False,
    title: str | None = None,
    description: str | None = None,
    artifact_type: str | None = None,
    primary: str | None = None,
) -> Path:
    if path:
        return _resolve_artifact_folder_from_path(
            artifact_id,
            path,
            create=create,
            title=title,
            description=description,
            artifact_type=artifact_type,
            primary=primary,
        )

    for root in _scan_artifact_dirs():
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for folder in children:
            if not folder.is_dir():
                continue
            meta = _load_metadata(folder) or {}
            identifiers = {
                folder.name,
                str(meta.get("id") or ""),
                str(meta.get("slug") or ""),
            }
            if artifact_id in identifiers:
                return folder.resolve()
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")


def _resolve_artifact_folder_from_path(
    artifact_id: str,
    raw_path: str,
    *,
    create: bool,
    title: str | None,
    description: str | None,
    artifact_type: str | None,
    primary: str | None,
) -> Path:
    try:
        resolved = resolve_artifact_path(raw_path, allow_dir=True)
        if resolved is None:
            raise FileNotFoundError("Artifact not found")
        folder = resolved if resolved.is_dir() else _artifact_root_for_file(resolved)
    except (FileNotFoundError, ValueError):
        try:
            target = Path(raw_path).expanduser()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc

        if create and not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        try:
            resolved_target = target.resolve(strict=False)
        except OSError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc

        if resolved_target.is_file():
            folder = _artifact_root_for_file(resolved_target)
        else:
            folder = resolved_target

    if not folder.exists():
        if not create:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
        folder.mkdir(parents=True, exist_ok=True)
    if not folder.is_dir():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Artifact path must resolve to a folder")

    metadata = folder / "metadata.json"
    if not metadata.is_file():
        if not create:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact metadata not found")
        _write_minimal_metadata(
            folder,
            artifact_id=artifact_id,
            title=title,
            description=description,
            artifact_type=artifact_type,
            primary=primary,
        )

    return folder.resolve()


def _artifact_root_for_file(path: Path) -> Path:
    current = path.parent.resolve()
    while current.parent != current:
        if (current / "metadata.json").is_file():
            return current
        if current.name == "artifacts" and current.parent.name == ".anton":
            break
        current = current.parent
    return path.parent.resolve()


def _write_minimal_metadata(
    folder: Path,
    *,
    artifact_id: str,
    title: str | None,
    description: str | None,
    artifact_type: str | None,
    primary: str | None,
) -> None:
    primary_name = primary or _detect_primary(folder)
    inferred_type = artifact_type or _type_for_primary(primary_name)
    metadata = {
        "id": artifact_id,
        "slug": folder.name,
        "name": title or folder.name.replace("-", " ").strip().title() or folder.name,
        "description": description or "",
        "type": inferred_type,
        "primary": primary_name,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _detect_primary(folder: Path) -> str:
    for name in ("index.html", "dashboard.html", "report.md", "README.md"):
        if (folder / name).is_file() and name != "README.md":
            return name
    files = _iter_snapshot_files(folder)
    return files[0].relative_to(folder).as_posix() if files else ""


def _type_for_primary(primary: str) -> str:
    suffix = Path(primary).suffix.lower()
    if suffix == ".html":
        return "html-app"
    if suffix in {".md", ".txt", ".pdf"}:
        return "document"
    if suffix in {".csv", ".json"}:
        return "dataset"
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "image"
    return "mixed"


def _versions_root(folder: Path) -> Path:
    if folder.parent.name == "artifacts" and folder.parent.parent.name == ".anton":
        return folder.parent.parent / "artifact_versions" / folder.name
    return folder / ".cowork_versions"


def _load_index(folder: Path) -> dict[str, Any]:
    path = _versions_root(folder) / _VERSION_INDEX
    if not path.is_file():
        return {"schemaVersion": 1, "versions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schemaVersion": 1, "versions": []}
    versions = data.get("versions") if isinstance(data, dict) else None
    return {"schemaVersion": 1, "versions": versions if isinstance(versions, list) else []}


def _save_index(folder: Path, index: dict[str, Any]) -> None:
    root = _versions_root(folder)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _VERSION_INDEX
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _create_checkpoint_fallback(
    folder: Path,
    *,
    label: str | None,
    summary: str | None,
    kind: str,
) -> dict[str, Any]:
    index = _load_index(folder)
    manifest = _build_manifest(folder)
    created_at = _now_iso()
    version_id = _new_version_id(manifest)
    snapshot_dir = _versions_root(folder) / _SNAPSHOT_DIR / version_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    for file_entry in manifest["files"]:
        src = folder / file_entry["path"]
        dst = snapshot_dir / file_entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    number = len(index["versions"]) + 1
    clean_label = (label or "").strip()
    human_label = clean_label or _default_checkpoint_label(kind, number)
    record = {
        "id": version_id,
        "versionId": version_id,
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
        "kind": kind,
        "label": human_label,
        "humanLabel": human_label,
        "summary": (summary or "").strip(),
        "createdAt": created_at,
        "fileCount": manifest["fileCount"],
        "files": manifest["files"],
        "manifest": manifest,
    }
    index["versions"].append(record)
    _save_index(folder, index)

    return {
        "status": "ok",
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
        "version": _decorate_version(folder, record),
        **_artifact_integration(folder),
    }


def _default_checkpoint_label(kind: str, number: int) -> str:
    labels = {
        "preview": "Preview checkpoint",
        "publish": "Published checkpoint",
        "restore": "Restore safety checkpoint",
        "auto": "Automatic checkpoint",
    }
    if kind in labels:
        return labels[kind]
    return f"Checkpoint {number}"


def _list_versions_fallback(folder: Path) -> dict[str, Any]:
    index = _load_index(folder)
    versions = [_decorate_version(folder, record) for record in reversed(index["versions"])]
    return {
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
        "versions": versions,
        "latest": versions[0] if versions else None,
        **_artifact_integration(folder),
    }


def _restore_version_fallback(folder: Path, version_id: str) -> dict[str, Any]:
    root = _versions_root(folder)
    index = _load_index(folder)
    record = _find_version(index, version_id)
    snapshot_dir = root / _SNAPSHOT_DIR / version_id
    if not snapshot_dir.is_dir():
        raise FileNotFoundError("Checkpoint snapshot is missing")

    for file_path in _iter_snapshot_files(folder):
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass
    _prune_empty_dirs(folder)

    for src in _iter_files(snapshot_dir):
        rel = src.relative_to(snapshot_dir)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return _decorate_version(folder, record)


def _diff_versions_fallback(
    folder: Path,
    *,
    base: str | None,
    compare: str,
    kind: str,
) -> dict[str, Any]:
    index = _load_index(folder)
    if not index["versions"] and not base:
        raise ValueError("At least one checkpoint is required to diff against the current artifact")

    base_ref = base or index["versions"][-1]["id"]
    base_info = _resolve_ref(folder, index, base_ref)
    compare_info = _resolve_ref(folder, index, compare or "current")
    base_files = {entry["path"]: entry for entry in base_info["manifest"]["files"]}
    compare_files = {entry["path"]: entry for entry in compare_info["manifest"]["files"]}

    changes: list[dict[str, Any]] = []
    for rel_path in sorted(set(base_files) | set(compare_files)):
        before = base_files.get(rel_path)
        after = compare_files.get(rel_path)
        if before is None:
            change_status = "added"
        elif after is None:
            change_status = "removed"
        elif before["sha256"] != after["sha256"]:
            change_status = "modified"
        else:
            continue

        change = {
            "path": rel_path,
            "status": change_status,
            "kind": (after or before or {}).get("kind", "File"),
            "label": _change_label(change_status, rel_path),
            "humanLabel": _change_label(change_status, rel_path),
            "before": before,
            "after": after,
            "sizeDelta": (after or {}).get("size", 0) - (before or {}).get("size", 0),
        }
        if kind == "text":
            text_diff = _text_diff_for_change(base_info, compare_info, rel_path, before, after)
            if text_diff is not None:
                change["textDiff"] = text_diff["text"]
                change["textDiffTruncated"] = text_diff["truncated"]
        changes.append(change)

    summary = {
        "added": sum(1 for c in changes if c["status"] == "added"),
        "modified": sum(1 for c in changes if c["status"] == "modified"),
        "removed": sum(1 for c in changes if c["status"] == "removed"),
        "unchanged": len(set(base_files) & set(compare_files)) - sum(1 for c in changes if c["status"] == "modified"),
        "totalChanged": len(changes),
    }
    return {
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
        "kind": kind,
        "base": _ref_payload(base_info),
        "compare": _ref_payload(compare_info),
        "summary": summary,
        "changes": changes,
        "changedFiles": changes,
        **_artifact_integration(folder),
    }


def _find_version(index: dict[str, Any], version_id: str) -> dict[str, Any]:
    for record in index["versions"]:
        if record.get("id") == version_id or record.get("versionId") == version_id:
            return record
    raise FileNotFoundError("Checkpoint not found")


def _resolve_ref(folder: Path, index: dict[str, Any], ref: str) -> dict[str, Any]:
    clean = (ref or "current").strip()
    if clean in {"current", "working", "workspace"}:
        return {
            "id": "current",
            "label": "Current artifact",
            "humanLabel": "Current artifact",
            "createdAt": _now_iso(),
            "root": folder,
            "manifest": _build_manifest(folder),
        }
    if clean in {"latest", "head"}:
        if not index["versions"]:
            raise FileNotFoundError("No checkpoints found")
        clean = index["versions"][-1]["id"]

    record = _find_version(index, clean)
    return {
        "id": record["id"],
        "label": record.get("label") or record["id"],
        "humanLabel": record.get("humanLabel") or record.get("label") or record["id"],
        "createdAt": record.get("createdAt", ""),
        "root": _versions_root(folder) / _SNAPSHOT_DIR / record["id"],
        "manifest": record.get("manifest") or {"fileCount": record.get("fileCount", 0), "files": record.get("files", [])},
    }


def _ref_payload(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": info["id"],
        "label": info.get("label", info["id"]),
        "humanLabel": info.get("humanLabel", info.get("label", info["id"])),
        "createdAt": info.get("createdAt", ""),
        "fileCount": info["manifest"].get("fileCount", 0),
    }


def _text_diff_for_change(
    base_info: dict[str, Any],
    compare_info: dict[str, Any],
    rel_path: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    file_entry = after or before
    if file_entry is None or not _is_text_entry(file_entry):
        return None

    old_text = _read_ref_text(base_info["root"], rel_path) if before is not None else ""
    new_text = _read_ref_text(compare_info["root"], rel_path) if after is not None else ""
    diff_lines = list(difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"{base_info['id']}/{rel_path}",
        tofile=f"{compare_info['id']}/{rel_path}",
        lineterm="",
    ))
    truncated = len("\n".join(diff_lines)) > _TEXT_DIFF_LIMIT
    text = "\n".join(diff_lines)
    if truncated:
        text = text[:_TEXT_DIFF_LIMIT]
    return {"text": text, "truncated": truncated}


def _read_ref_text(root: Path, rel_path: str) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace")


def _build_manifest(folder: Path) -> dict[str, Any]:
    entries = [_file_manifest(folder, path) for path in _iter_snapshot_files(folder)]
    digest_source = json.dumps(
        [{"path": e["path"], "sha256": e["sha256"], "size": e["size"]} for e in entries],
        sort_keys=True,
    ).encode("utf-8")
    return {
        "fileCount": len(entries),
        "digest": hashlib.sha256(digest_source).hexdigest(),
        "files": entries,
    }


def _file_manifest(folder: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(folder).as_posix()
    stat = path.stat()
    suffix = path.suffix.lower()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return {
        "path": rel,
        "name": path.name,
        "kind": _kind_for_file(path),
        "mime": mime,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "sha256": _sha256(path),
        "text": suffix in TEXT_EXTENSIONS or mime.startswith("text/"),
    }


def _iter_snapshot_files(folder: Path) -> list[Path]:
    files = [
        path
        for path in _iter_files(folder)
        if not _is_excluded_snapshot_path(path.relative_to(folder))
    ]
    files.sort(key=lambda p: p.relative_to(folder).as_posix())
    return files


def _iter_files(root: Path) -> list[Path]:
    try:
        return [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    except OSError:
        return []


def _is_excluded_snapshot_path(rel_path: Path) -> bool:
    if not rel_path.parts:
        return True
    return rel_path.parts[0] in _SNAPSHOT_EXCLUDED_TOPS


def _prune_empty_dirs(folder: Path) -> None:
    for path in sorted((p for p in folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if path == folder:
            continue
        try:
            path.relative_to(_versions_root(folder))
            continue
        except ValueError:
            pass
        try:
            path.rmdir()
        except OSError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_version_id(manifest: dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = manifest.get("digest", "")[:10] or uuid4().hex[:10]
    return f"cp_{ts}_{digest}_{uuid4().hex[:6]}"


def _decorate_version(folder: Path, record: dict[str, Any]) -> dict[str, Any]:
    out = {
        **record,
        "artifactId": _artifact_id(folder),
        "artifactPath": str(folder),
    }
    out.setdefault("versionId", out.get("id"))
    out.setdefault("humanLabel", out.get("label") or out.get("id"))
    out.setdefault("fileCount", len(out.get("files", [])))
    out.setdefault("manifest", {"fileCount": out["fileCount"], "files": out.get("files", [])})
    return out


def _artifact_id(folder: Path) -> str:
    meta = _load_metadata(folder) or {}
    return str(meta.get("id") or meta.get("slug") or folder.name)


def _artifact_integration(folder: Path) -> dict[str, Any]:
    meta = _load_metadata(folder) or {}
    files = _user_files(folder)
    primary = _pick_primary(folder, files, primary_hint=meta.get("primary"))
    primary_path = str(primary) if primary is not None else str(folder)
    artifact_type = meta.get("type") or "mixed"
    primary_ext = primary.suffix.lower() if primary is not None else ""
    access = _published_access_for(folder, primary)
    return {
        "artifact": {
            "id": _artifact_id(folder),
            "slug": meta.get("slug") or folder.name,
            "title": meta.get("name") or folder.name,
            "description": meta.get("description") or "",
            "type": artifact_type,
            "kind": KIND_BY_TYPE.get(artifact_type) or KIND_BY_EXT.get(primary_ext, "File"),
            "path": primary_path,
            "folder": str(folder),
            "primary": meta.get("primary") or None,
        },
        "preview": {
            "path": primary_path,
            "serveUrl": serve_url_for(primary_path),
        },
        "publish": {
            "publishedUrl": _published_url_for(folder, primary),
            **access,
        },
    }


def _merge_integration(folder: Path, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    merged = dict(payload)
    integration = _artifact_integration(folder)
    for key, value in integration.items():
        merged.setdefault(key, value)
    return merged


def _kind_for_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".py", ".js", ".ts", ".tsx", ".css", ".sql", ".sh"}:
        return "Code"
    return KIND_BY_EXT.get(suffix, "File")


def _change_label(change_status: str, rel_path: str) -> str:
    verbs = {
        "added": "Added",
        "modified": "Updated",
        "removed": "Removed",
    }
    return f"{verbs.get(change_status, 'Changed')} {rel_path}"


def _is_text_entry(entry: dict[str, Any]) -> bool:
    if not entry.get("text"):
        return False
    return int(entry.get("size") or 0) <= _TEXT_DIFF_LIMIT


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
