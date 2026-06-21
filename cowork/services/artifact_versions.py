"""Content-addressed artifact snapshots.

This service snapshots the existing Anton/Hermes artifact folder convention
without changing the scanner or publish paths. Version rows are immutable:
restore materializes an older version, then snapshots that materialized content
as a new version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import difflib
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import quote
from uuid import UUID, uuid4

from sqlmodel import Session, func, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.artifact import (
    Artifact,
    ArtifactActivityEvent,
    ArtifactComment,
    ArtifactDeployment,
    ArtifactVersion,
    ArtifactVersionFile,
)
from cowork.models.project import Project
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.project_collaboration import delivery_to_dict, dispatch_project_notification, normalize_email
from cowork.services.project_permissions import role_allows
from cowork.services.artifact_version_backends import ShadowBackendSnapshot, run_shadow_snapshots
from cowork.services.artifacts import (
    KIND_BY_EXT,
    KIND_BY_TYPE,
    TEXT_EXTENSIONS,
    _load_metadata,
    _pick_primary,
    _published_access_for,
    _published_url_for,
    _scan_artifact_dirs,
    _user_files,
    register_preview_mount,
    resolve_artifact_path,
    serve_url_for,
)
from cowork.services.screenshot_diff import (
    ScreenshotDiffUnavailable,
    render_static_html_screenshot_diff,
    render_url_screenshot_diff,
)

try:
    from cowork.services.artifacts import _HOUSEKEEPING_FILES as SCANNER_HOUSEKEEPING_FILES
except Exception:  # pragma: no cover - defensive import fallback
    SCANNER_HOUSEKEEPING_FILES = {"metadata.json", "README.md", ".published.json"}


MANIFEST_SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
MAX_HOUSEKEEPING_SNAPSHOT_BYTES = 256 * 1024
HOUSEKEEPING_TOP_LEVEL = frozenset(
    {
        *SCANNER_HOUSEKEEPING_FILES,
        ".artifact-store",
        ".artifact_versions",
        ".versions",
    }
)


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    content_hash: str
    size: int


@dataclass(frozen=True)
class SnapshotManifest:
    files: tuple[SnapshotFile, ...]
    files_hash: str
    manifest_hash: str
    total_bytes: int

    @property
    def file_count(self) -> int:
        return len(self.files)


def _canonical_json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_blob_path(content_hash: str) -> Path:
    return Path("blobs") / HASH_ALGORITHM / content_hash[:2] / content_hash[2:4] / content_hash


def _safe_relative_path(path: str) -> PurePosixPath:
    rel = PurePosixPath(path)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Unsafe artifact version path: {path!r}")
    return rel


class ArtifactVersionService:
    def __init__(self, session: Session, store_root: str | Path | None = None) -> None:
        self.session = session
        self.store_root = Path(store_root) if store_root is not None else self._default_store_root()

    def _default_store_root(self) -> Path:
        return Path(get_app_settings().project.root_dir) / ".cowork-artifact-store"

    def scan_manifest(self, artifact_dir: str | Path) -> SnapshotManifest:
        """Build the deterministic manifest for an artifact folder."""
        root = Path(artifact_dir).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise FileNotFoundError(f"Artifact folder does not exist: {root}")

        files: list[SnapshotFile] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            rel_posix = rel.as_posix()
            top = rel.parts[0] if rel.parts else ""
            if top in HOUSEKEEPING_TOP_LEVEL:
                continue
            stat = path.stat()
            files.append(
                SnapshotFile(
                    path=rel_posix,
                    content_hash=_sha256_file(path),
                    size=stat.st_size,
                )
            )

        files.sort(key=lambda entry: entry.path)
        file_payload = [
            {"path": entry.path, "sha256": entry.content_hash, "size": entry.size}
            for entry in files
        ]
        files_hash = _sha256_bytes(_canonical_json(file_payload))
        manifest_hash = _sha256_bytes(
            _canonical_json(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "hash_algorithm": HASH_ALGORITHM,
                    "files": file_payload,
                }
            )
        )
        return SnapshotManifest(
            files=tuple(files),
            files_hash=files_hash,
            manifest_hash=manifest_hash,
            total_bytes=sum(entry.size for entry in files),
        )

    def snapshot_artifact(
        self,
        artifact_dir: str | Path,
        *,
        artifact_id: UUID | None = None,
        project_id: UUID | None = None,
        slug: str | None = None,
        title: str | None = None,
        description: str | None = None,
        artifact_type: str | None = None,
        source_conversation_id: UUID | None = None,
        source_message_id: UUID | None = None,
        prompt: str | None = None,
        label: str | None = None,
        operation_type: str = "snapshot",
        preview_status: str = "pending",
        publish_status: str = "unpublished",
        restored_from_version_id: UUID | None = None,
        branch_name: str | None = None,
        forked_from_version_id: UUID | None = None,
        actor_name: str | None = None,
        actor_email: str | None = None,
        actor_subject: str | None = None,
    ) -> ArtifactVersion:
        """Persist a content-addressed snapshot and return the new version row."""
        root = Path(artifact_dir).expanduser().resolve(strict=False)
        manifest = self.scan_manifest(root)
        metadata = self._read_metadata(root)
        if project_id is None:
            project_id = _project_id_for_folder(self.session, root)
        artifact = self._get_or_create_artifact(
            root,
            artifact_id=artifact_id,
            project_id=project_id,
            slug=slug or self._metadata_text(metadata, "slug") or root.name,
            title=title or self._metadata_text(metadata, "name") or root.name,
            description=description if description is not None else self._metadata_text(metadata, "description"),
            artifact_type=artifact_type if artifact_type is not None else self._metadata_text(metadata, "type"),
        )
        version_number = self._next_version_number(artifact.id)
        version = ArtifactVersion(
            artifact_id=artifact.id,
            parent_version_id=artifact.current_version_id,
            version_number=version_number,
            label=label,
            manifest_hash=manifest.manifest_hash,
            files_hash=manifest.files_hash,
            file_count=manifest.file_count,
            total_bytes=manifest.total_bytes,
            store_path=str(Path("versions") / f"{artifact.id}" / f"{version_number:06d}-{manifest.manifest_hash}.json"),
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            prompt=prompt,
            operation_type=operation_type,
            preview_status=preview_status,
            publish_status=publish_status,
            restored_from_version_id=restored_from_version_id,
            branch_name=branch_name,
            forked_from_version_id=forked_from_version_id,
        )
        self.session.add(version)
        self.session.flush()

        self._store_manifest(version, artifact, manifest, metadata=metadata, root=root)
        for entry in manifest.files:
            blob_rel = self._store_blob(root / Path(entry.path), entry.content_hash)
            self.session.add(
                ArtifactVersionFile(
                    version_id=version.id,
                    path=entry.path,
                    content_hash=entry.content_hash,
                    size=entry.size,
                    blob_path=str(blob_rel),
                )
            )

        artifact.current_version_id = version.id
        if _is_good_status(preview_status, publish_status):
            artifact.last_known_good_version_id = version.id
        self.session.add(artifact)
        shadow_backends = run_shadow_snapshots(
            ShadowBackendSnapshot(
                artifact_id=artifact.id,
                version_id=version.id,
                version_number=version.version_number,
                artifact_dir=root,
                store_root=self.store_root,
                manifest_path=self.store_root / version.store_path,
                files_hash=version.files_hash,
                manifest_hash=version.manifest_hash,
            )
        )
        activity_details = {
            "label": label or "",
            "prompt": prompt or "",
            **_actor_details(actor_name=actor_name, actor_email=actor_email, actor_subject=actor_subject),
            **({"shadowBackends": shadow_backends} if shadow_backends else {}),
        }
        notification_event_type = _snapshot_notification_event(operation_type)
        if notification_event_type:
            deliveries = dispatch_project_notification(
                self.session,
                artifact,
                notification_event_type,
                details={
                    "versionId": str(version.id),
                    "versionNumber": version.version_number,
                    "label": label or "",
                    "prompt": prompt or "",
                    "operationType": operation_type,
                    **_actor_details(actor_name=actor_name, actor_email=actor_email, actor_subject=actor_subject),
                },
            )
            activity_details["notificationDeliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
        self.session.add(
            ArtifactActivityEvent(
                artifact_id=artifact.id,
                version_id=version.id,
                event_type=operation_type,
                actor_name=actor_name,
                details=activity_details,
            )
        )
        self.session.commit()
        self.session.refresh(version)
        return version

    def materialize_version(self, version_id: UUID, target_dir: str | Path, *, clean: bool = True) -> Path:
        """Write a stored version's files to target_dir and return that path."""
        version = self._get_version(version_id)
        target = Path(target_dir).expanduser().resolve(strict=False)
        target.mkdir(parents=True, exist_ok=True)

        files = self.session.exec(
            select(ArtifactVersionFile)
            .where(ArtifactVersionFile.version_id == version.id)
            .order_by(ArtifactVersionFile.path)
        ).all()
        materialized_entries = []
        for entry in files:
            rel = _safe_relative_path(entry.path)
            blob = self.store_root / entry.blob_path
            if not blob.is_file():
                raise FileNotFoundError(f"Missing artifact blob: {blob}")
            if _sha256_file(blob) != entry.content_hash:
                raise IOError(f"Stored artifact blob failed hash check: {entry.path}")
            materialized_entries.append((entry, rel, blob))
        if clean:
            self._clean_materialized_files(target)

        for entry, rel, blob in materialized_entries:
            destination = target.joinpath(*rel.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blob, destination)
            if _sha256_file(destination) != entry.content_hash:
                raise IOError(f"Materialized artifact file failed hash check: {entry.path}")
        return target

    def replace_with_version(
        self,
        version_id: UUID,
        target_dir: str | Path,
        *,
        metadata_overrides: dict | None = None,
        include_readme: bool = True,
        clear_published: bool = False,
        preserve_published: bool = False,
    ) -> Path:
        """Atomically replace a live artifact folder with a stored version."""
        target = Path(target_dir).expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        published_snapshot: bytes | None = None
        published_path = target / ".published.json"
        if preserve_published and not clear_published and published_path.is_file():
            published_snapshot = published_path.read_bytes()
        with TemporaryDirectory(prefix=f".{target.name}.version-", dir=target.parent) as tmp:
            staged = Path(tmp) / target.name
            self.materialize_version(version_id, staged, clean=True)
            self.write_version_housekeeping(
                version_id,
                staged,
                metadata_overrides=metadata_overrides,
                include_readme=include_readme,
                clear_published=clear_published,
            )
            if published_snapshot is not None:
                (staged / ".published.json").write_bytes(published_snapshot)
            self._replace_artifact_folder(target, staged)
        return target

    def version_metadata(self, version_id: UUID) -> dict:
        """Return metadata.json captured with a version, when available."""
        payload = self._read_version_manifest(self._get_version(version_id))
        metadata = payload.get("artifact_metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
        housekeeping = payload.get("housekeeping")
        if isinstance(housekeeping, dict):
            metadata_entry = housekeeping.get("metadata.json")
            if isinstance(metadata_entry, dict) and isinstance(metadata_entry.get("data"), dict):
                return dict(metadata_entry["data"])
        return {}

    def write_version_housekeeping(
        self,
        version_id: UUID,
        target_dir: str | Path,
        *,
        metadata_overrides: dict | None = None,
        include_readme: bool = True,
        clear_published: bool = False,
    ) -> bool:
        """Write captured non-hashed support files for a materialized version."""
        version = self._get_version(version_id)
        payload = self._read_version_manifest(version)
        target = Path(target_dir).expanduser().resolve(strict=False)
        target.mkdir(parents=True, exist_ok=True)

        wrote_metadata = False
        metadata = self.version_metadata(version.id)
        if metadata or metadata_overrides:
            metadata = {**metadata, **(metadata_overrides or {})}
            (target / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            wrote_metadata = True

        housekeeping = payload.get("housekeeping")
        if include_readme and isinstance(housekeeping, dict):
            readme_entry = housekeeping.get("README.md")
            readme_text = readme_entry.get("text") if isinstance(readme_entry, dict) else None
            if isinstance(readme_text, str):
                (target / "README.md").write_text(readme_text, encoding="utf-8")

        if clear_published:
            (target / ".published.json").unlink(missing_ok=True)
        return wrote_metadata

    def restore_version(
        self,
        source_version_id: UUID,
        target_dir: str | Path | None = None,
        *,
        source_conversation_id: UUID | None = None,
        source_message_id: UUID | None = None,
        prompt: str | None = None,
        operation_type: str = "restore",
        preview_status: str = "pending",
        publish_status: str = "unpublished",
        label: str | None = None,
        external_artifact_id: str | None = None,
        actor_name: str | None = None,
        actor_email: str | None = None,
        actor_subject: str | None = None,
    ) -> ArtifactVersion:
        """Materialize an older version, then record that restored content as new history."""
        source = self._get_version(source_version_id)
        artifact = self.session.get(Artifact, source.artifact_id)
        if artifact is None:
            raise ValueError("Artifact not found for version")
        target = Path(target_dir if target_dir is not None else artifact.path).expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        try:
            with TemporaryDirectory(prefix=f".{target.name}.restore-", dir=target.parent) as tmp:
                staged = Path(tmp) / target.name
                self.materialize_version(source.id, staged, clean=True)
                wrote_metadata = self.write_version_housekeeping(
                    source.id,
                    staged,
                    metadata_overrides={"id": external_artifact_id} if external_artifact_id else None,
                    clear_published=True,
                )
                if not wrote_metadata:
                    self._write_restored_metadata(staged, artifact, external_artifact_id=external_artifact_id)
                backup = self._replace_artifact_folder(target, staged, keep_backup=True)
            restored = self.snapshot_artifact(
                target,
                artifact_id=artifact.id,
                project_id=artifact.project_id,
                slug=artifact.slug,
                title=artifact.title,
                description=artifact.description,
                artifact_type=artifact.artifact_type,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
                prompt=prompt,
                label=label,
                operation_type=operation_type,
                preview_status=preview_status,
                publish_status=publish_status,
                restored_from_version_id=source.id,
                actor_name=actor_name,
                actor_email=actor_email,
                actor_subject=actor_subject,
            )
        except Exception:
            self.session.rollback()
            if backup is not None and backup.exists():
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                os.replace(backup, target)
            raise
        finally:
            if backup is not None and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        return restored

    def _get_version(self, version_id: UUID) -> ArtifactVersion:
        version = self.session.get(ArtifactVersion, version_id)
        if version is None:
            raise ValueError("Artifact version not found")
        return version

    def _get_or_create_artifact(
        self,
        root: Path,
        *,
        artifact_id: UUID | None,
        project_id: UUID | None,
        slug: str,
        title: str,
        description: str | None,
        artifact_type: str | None,
    ) -> Artifact:
        artifact = self.session.get(Artifact, artifact_id) if artifact_id is not None else None
        root_text = str(root)
        if artifact is None:
            artifact = self.session.exec(select(Artifact).where(Artifact.path == root_text)).first()
        if artifact is None:
            artifact = Artifact(
                project_id=project_id,
                slug=slug,
                title=title,
                description=description,
                artifact_type=artifact_type,
                path=root_text,
            )
        else:
            if project_id is not None:
                artifact.project_id = project_id
            artifact.slug = slug
            artifact.title = title
            artifact.description = description
            artifact.artifact_type = artifact_type
            artifact.path = root_text
        self.session.add(artifact)
        self.session.flush()
        return artifact

    def _next_version_number(self, artifact_id: UUID) -> int:
        current = self.session.exec(
            select(func.max(ArtifactVersion.version_number)).where(ArtifactVersion.artifact_id == artifact_id)
        ).one()
        return int(current or 0) + 1

    def _store_blob(self, source: Path, content_hash: str) -> Path:
        rel = _relative_blob_path(content_hash)
        dest = self.store_root / rel
        if dest.is_file():
            if _sha256_file(dest) == content_hash:
                return rel
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(prefix=f".{content_hash}.", dir=dest.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, tmp)
        try:
            if _sha256_file(tmp_path) != content_hash:
                raise IOError(f"Artifact file changed while snapshotting: {source}")
            os.replace(tmp_path, dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return rel

    def _store_manifest(
        self,
        version: ArtifactVersion,
        artifact: Artifact,
        manifest: SnapshotManifest,
        *,
        metadata: dict,
        root: Path,
    ) -> None:
        path = self.store_root / version.store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "hash_algorithm": HASH_ALGORITHM,
            "artifact_id": str(artifact.id),
            "artifact_slug": artifact.slug,
            "version_id": str(version.id),
            "version_number": version.version_number,
            "manifest_hash": manifest.manifest_hash,
            "files_hash": manifest.files_hash,
            "artifact_metadata": metadata,
            "housekeeping": self._snapshot_housekeeping(root, metadata),
            "files": [
                {"path": entry.path, "sha256": entry.content_hash, "size": entry.size}
                for entry in manifest.files
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _read_version_manifest(self, version: ArtifactVersion) -> dict:
        path = self.store_root / version.store_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _snapshot_housekeeping(self, root: Path, metadata: dict) -> dict:
        housekeeping: dict[str, dict] = {}
        if (root / "metadata.json").is_file():
            housekeeping["metadata.json"] = {"encoding": "json", "data": metadata}
        readme = root / "README.md"
        try:
            if readme.is_file() and readme.stat().st_size <= MAX_HOUSEKEEPING_SNAPSHOT_BYTES:
                housekeeping["README.md"] = {
                    "encoding": "utf-8",
                    "text": readme.read_text(encoding="utf-8"),
                }
        except (OSError, UnicodeDecodeError):
            pass
        return housekeeping

    def _clean_materialized_files(self, target: Path) -> None:
        if not target.exists():
            return
        paths = sorted(target.rglob("*"), key=lambda item: len(item.relative_to(target).parts), reverse=True)
        for path in paths:
            rel = path.relative_to(target)
            top = rel.parts[0] if rel.parts else ""
            if top in HOUSEKEEPING_TOP_LEVEL:
                continue
            if path.is_dir() and not path.is_symlink():
                try:
                    path.rmdir()
                except OSError:
                    pass
            else:
                path.unlink(missing_ok=True)

    def _replace_artifact_folder(self, target: Path, staged: Path, *, keep_backup: bool = False) -> Path | None:
        target = target.expanduser().resolve(strict=False)
        staged = staged.expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(staged, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None and not keep_backup:
            shutil.rmtree(backup, ignore_errors=True)
            return None
        return backup

    def _read_metadata(self, root: Path) -> dict:
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _metadata_text(self, metadata: dict, key: str) -> str | None:
        value = metadata.get(key)
        return value if isinstance(value, str) and value else None

    def _write_restored_metadata(
        self,
        target: Path,
        artifact: Artifact,
        *,
        external_artifact_id: str | None,
    ) -> None:
        files = _user_files(target)
        primary = _pick_primary(target, files, primary_hint=None)
        metadata = {
            "id": external_artifact_id or _deleted_external_artifact_id(self.session, artifact) or artifact.slug or str(artifact.id),
            "slug": target.name,
            "name": artifact.title or target.name,
            "description": artifact.description or "",
            "type": artifact.artifact_type or _type_for_primary(primary.name if primary is not None else ""),
        }
        if primary is not None:
            metadata["primary"] = primary.relative_to(target).as_posix()
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_or_create_artifact_for_path(
    session: Session,
    raw_path: str,
    *,
    artifact_id: str | UUID | None = None,
    title: str | None = None,
    description: str | None = None,
    artifact_type: str | None = None,
) -> Artifact:
    folder = _artifact_folder_from_path(session, raw_path)
    metadata = _load_metadata(folder) or {}
    project_id = _project_id_for_folder(session, folder)
    internal_id = artifact_id if isinstance(artifact_id, UUID) else None
    if internal_id is None and isinstance(artifact_id, str):
        try:
            internal_id = UUID(artifact_id)
        except ValueError:
            internal_id = None
    service = ArtifactVersionService(session)
    return service._get_or_create_artifact(
        folder,
        artifact_id=internal_id,
        project_id=project_id,
        slug=str(metadata.get("slug") or folder.name),
        title=title or _metadata_str(metadata, "name") or folder.name,
        description=description if description is not None else _metadata_str(metadata, "description"),
        artifact_type=artifact_type if artifact_type is not None else _metadata_str(metadata, "type"),
    )


def list_versions(
    session: Session,
    artifact: Artifact | None = None,
    *,
    artifact_id: str | UUID | None = None,
    path: str | None = None,
    **_: object,
) -> list[ArtifactVersion] | dict:
    explicit_artifact = artifact is not None
    artifact = artifact or _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
    versions = session.exec(
        select(ArtifactVersion)
        .where(ArtifactVersion.artifact_id == artifact.id)
        .order_by(ArtifactVersion.version_number.desc())
    ).all()
    if explicit_artifact:
        return list(versions)
    artifact_external_id = _external_artifact_id(artifact, session)
    return {
        "artifactId": artifact_external_id,
        "artifactPath": artifact.path,
        "currentVersionId": str(artifact.current_version_id) if artifact.current_version_id else None,
        "lastKnownGoodVersionId": (
            str(artifact.last_known_good_version_id) if artifact.last_known_good_version_id else None
        ),
        "versions": [version_to_dict(version, session=session, artifact_external_id=artifact_external_id) for version in versions],
        "latest": version_to_dict(versions[0], session=session, artifact_external_id=artifact_external_id) if versions else None,
    }


def snapshot_artifact(
    session: Session,
    path: str | None = None,
    *,
    artifact_id: str | UUID | None = None,
    body: dict | None = None,
    operation_type: str = "checkpoint",
    label: str | None = None,
    prompt: str | None = None,
    preview_status: str = "pending",
    publish_status: str = "unpublished",
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
    **_: object,
) -> ArtifactVersion | dict:
    body = body or {}
    requested_path = path or _body_str(body, "path")
    folder = _artifact_folder_from_request(session, artifact_id=artifact_id, path=requested_path, create=bool(body))
    if body and artifact_id is not None:
        _ensure_metadata(folder, artifact_id, body=body)
    metadata = _load_metadata(folder) or {}
    checkpoint_label = label or _body_str(body, "label") or _body_str(body, "summary")
    version = ArtifactVersionService(session).snapshot_artifact(
        folder,
        artifact_id=artifact_id if isinstance(artifact_id, UUID) else None,
        project_id=_project_id_for_folder(session, folder),
        slug=str(metadata.get("slug") or folder.name),
        title=_body_str(body, "title") or _metadata_str(metadata, "name") or folder.name,
        description=_body_str(body, "description") if "description" in body else _metadata_str(metadata, "description"),
        artifact_type=_body_str(body, "artifact_type") or _body_str(body, "artifactType") or _metadata_str(metadata, "type"),
        source_conversation_id=_uuid_or_none(
            body.get("source_conversation_id")
            or body.get("sourceConversationId")
            or body.get("conversation_id")
            or body.get("conversationId")
        ),
        source_message_id=_uuid_or_none(
            body.get("source_message_id")
            or body.get("sourceMessageId")
            or body.get("message_id")
            or body.get("messageId")
        ),
        prompt=prompt or _body_str(body, "prompt"),
        label=checkpoint_label,
        operation_type=operation_type if operation_type != "checkpoint" else _body_str(body, "kind") or operation_type,
        preview_status=preview_status,
        publish_status=publish_status,
        actor_name=actor_name,
        actor_email=actor_email,
        actor_subject=actor_subject,
    )
    if artifact_id is None and not body:
        return version
    artifact = session.get(Artifact, version.artifact_id)
    return {
        "status": "ok",
        "artifactId": _external_artifact_id(artifact) if artifact else str(version.artifact_id),
        "artifactPath": artifact.path if artifact else str(folder),
        "version": version_to_dict(version, session=session, artifact_external_id=_external_artifact_id(artifact) if artifact else None),
        **_artifact_integration(folder),
    }


def restore_version(
    session: Session,
    path: str,
    version_id: UUID,
    *,
    label: str | None = None,
    prompt: str | None = None,
    **_: object,
) -> ArtifactVersion:
    folder = _artifact_folder_from_path(session, path)
    return ArtifactVersionService(session).restore_version(version_id, folder, label=label, prompt=prompt)


def restore_artifact(
    session: Session,
    artifact_id: str | UUID | None = None,
    *,
    path: str | None = None,
    version_id: str | UUID | None = None,
    body: dict | None = None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
    **_: object,
) -> dict:
    body = body or {}
    version_ref = version_id or _body_str(body, "version_id") or _body_str(body, "versionId")
    if version_ref is None:
        raise ValueError("version_id is required")
    source_id = version_ref if isinstance(version_ref, UUID) else UUID(str(version_ref))
    source = session.get(ArtifactVersion, source_id)
    if source is None:
        raise FileNotFoundError("Checkpoint not found")
    requested_artifact = _artifact_from_identifier_or_path(
        session,
        artifact_id=artifact_id,
        path=path or _body_str(body, "path"),
    )
    if requested_artifact.id != source.artifact_id:
        raise FileNotFoundError("Checkpoint not found")
    folder = Path(requested_artifact.path)
    service = ArtifactVersionService(session)
    created_checkpoint = None
    # A deleted artifact's path was tombstoned on delete (released so a new artifact
    # could reuse the original path). Restore it to its ORIGINAL folder — or a free
    # sibling if that path is now occupied by a different artifact — and heal the
    # record's path so it's live again.
    if not folder.is_dir():
        original = _deleted_original_path(session, requested_artifact)
        if original:
            target = Path(original)
            if target.exists():
                base, n = target, 2
                while target.exists():
                    target = base.parent / f"{base.name}-restored{'' if n == 2 else f'-{n}'}"
                    n += 1
            if str(target) != requested_artifact.path:
                folder = target
                requested_artifact.path = str(folder)
                session.add(requested_artifact)
    folder_exists = folder.is_dir()
    if folder_exists and bool(body.get("create_checkpoint") or body.get("createCheckpoint")):
        created_checkpoint = service.snapshot_artifact(
            folder,
            artifact_id=source.artifact_id if source is not None else None,
            label="Before restore",
            operation_type="restore_safety",
            actor_name=actor_name,
            actor_email=actor_email,
            actor_subject=actor_subject,
        )
    requested_artifact_id = (
        _body_str(body, "artifact_id")
        or _body_str(body, "artifactId")
        or (str(artifact_id) if artifact_id is not None else None)
    )
    deleted_external_id = _deleted_external_artifact_id(session, requested_artifact)
    external_artifact_id = deleted_external_id if not folder_exists and deleted_external_id else requested_artifact_id
    restored = service.restore_version(
        source_id,
        folder,
        label=_body_str(body, "label") or "Restored version",
        operation_type="restore" if folder_exists else "restore_deleted",
        external_artifact_id=external_artifact_id,
        actor_name=actor_name,
        actor_email=actor_email,
        actor_subject=actor_subject,
    )
    artifact = session.get(Artifact, restored.artifact_id)
    artifact_external_id = _external_artifact_id(artifact, session) if artifact else str(restored.artifact_id)
    payload = {
        "status": "ok",
        "artifactId": artifact_external_id,
        "artifactPath": artifact.path if artifact else str(folder),
        "restoredVersion": version_to_dict(source, session=session, artifact_external_id=artifact_external_id) if source else None,
        "version": version_to_dict(restored, session=session, artifact_external_id=artifact_external_id),
        **_artifact_integration(folder),
    }
    if created_checkpoint is not None:
        payload["createdCheckpoint"] = version_to_dict(
            created_checkpoint,
            session=session,
            artifact_external_id=artifact_external_id,
        )
    return payload


def restore_checkpoint(*args, **kwargs) -> dict:
    return restore_artifact(*args, **kwargs)


def fork_version(
    session: Session,
    version_id: str | UUID | None = None,
    *,
    artifact_id: str | UUID | None = None,
    path: str | None = None,
    body: dict | None = None,
    name: str | None = None,
    slug: str | None = None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
    **_: object,
) -> dict:
    body = body or {}
    source_ref = version_id or _body_str(body, "version_id") or _body_str(body, "versionId")
    if source_ref is None:
        raise ValueError("version_id is required")
    source = _get_version(session, source_ref)
    source_artifact = session.get(Artifact, source.artifact_id)
    if source_artifact is None:
        raise ValueError("Artifact not found for version")
    if artifact_id is not None or path is not None:
        requested_artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
        if requested_artifact.id != source.artifact_id:
            raise FileNotFoundError("Checkpoint not found")

    source_folder = Path(source_artifact.path)
    target_project = _target_project_from_body(session, body)
    parent = Path(target_project.path) / ".anton" / "artifacts" if target_project is not None else source_folder.parent
    parent.mkdir(parents=True, exist_ok=True)
    target_project_id = target_project.id if target_project is not None else source_artifact.project_id
    requested_name = name or _body_str(body, "name") or _body_str(body, "title") or f"{source_artifact.title} copy"
    requested_slug = slug or _body_str(body, "slug") or _safe_slug(requested_name)
    target = _unique_artifact_folder(parent, requested_slug)

    service = ArtifactVersionService(session)
    service.materialize_version(source.id, target, clean=True)
    service.write_version_housekeeping(source.id, target)
    metadata = service.version_metadata(source.id) or service._read_metadata(source_folder)
    fork_external_id = _body_str(body, "id") or f"{metadata.get('id') or source_artifact.slug}-{target.name}"
    metadata = {
        **metadata,
        "id": fork_external_id,
        "slug": target.name,
        "name": requested_name,
        "forked_from_artifact_id": str(source_artifact.id),
        "forked_from_version_id": str(source.id),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    version = service.snapshot_artifact(
        target,
        project_id=target_project_id,
        slug=target.name,
        title=requested_name,
        description=_metadata_str(metadata, "description") or source_artifact.description,
        artifact_type=_metadata_str(metadata, "type") or source_artifact.artifact_type,
        operation_type="fork",
        label=f"Forked from version {source.version_number}",
        branch_name=requested_name,
        forked_from_version_id=source.id,
        actor_name=actor_name,
        actor_email=actor_email,
        actor_subject=actor_subject,
    )
    artifact = session.get(Artifact, version.artifact_id)
    return {
        "status": "ok",
        "artifactId": _external_artifact_id(artifact) if artifact else fork_external_id,
        "artifactPath": str(target),
        "sourceVersion": version_to_dict(source, session=session, artifact_external_id=_external_artifact_id(source_artifact), include_files=False),
        "version": version_to_dict(version, session=session, artifact_external_id=_external_artifact_id(artifact) if artifact else None),
        **_artifact_integration(target),
    }


def fork_artifact_version(*args, **kwargs) -> dict:
    return fork_version(*args, **kwargs)


def diff_versions(
    session: Session,
    base_id: UUID | str | None = None,
    compare_id: UUID | str | None = None,
    *,
    artifact_id: str | UUID | None = None,
    path: str | None = None,
    base: str | None = None,
    compare: str | None = None,
    kind: str = "text",
    request_base_url: str | None = None,
    **_: object,
) -> dict:
    service = ArtifactVersionService(session)
    artifact = None
    base_version = None
    compare_version = None
    base_root = None
    compare_root = None
    base_manifest = None
    compare_manifest = None
    if artifact_id is not None or path is not None:
        artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
        versions = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .order_by(ArtifactVersion.version_number.desc())
        ).all()
        if not versions:
            raise ValueError("At least one checkpoint is required to diff")
        compare_ref = compare or "latest"
        if _is_current_ref(compare_ref):
            compare_manifest = service.scan_manifest(artifact.path)
            compare_files = _manifest_file_map(compare_manifest)
            compare_root = Path(artifact.path)
        else:
            compare_version = _resolve_version_ref(versions, compare_ref)
            compare_files = _version_file_map(session, compare_version.id)

        if _is_current_ref(base):
            base_manifest = service.scan_manifest(artifact.path)
            base_files = _manifest_file_map(base_manifest)
            base_root = Path(artifact.path)
        elif base:
            base_version = _resolve_version_ref(versions, base)
            base_files = _version_file_map(session, base_version.id)
        else:
            base_version = versions[0] if _is_current_ref(compare_ref) else (versions[1] if len(versions) > 1 else versions[0])
            base_files = _version_file_map(session, base_version.id)
    else:
        if base_id is None or compare_id is None:
            raise ValueError("Both versions are required for diff")

        base_version = _get_version(session, base_id)
        compare_version = _get_version(session, compare_id)
        artifact = session.get(Artifact, compare_version.artifact_id) or session.get(Artifact, base_version.artifact_id)
        base_files = _version_file_map(session, base_version.id)
        compare_files = _version_file_map(session, compare_version.id)

    changes = []
    text_diffs = []
    for rel_path in sorted(set(base_files) | set(compare_files)):
        before = base_files.get(rel_path)
        after = compare_files.get(rel_path)
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        elif before.content_hash != after.content_hash:
            status = "modified"
        else:
            continue
        change = {
            "path": rel_path,
            "status": status,
            "kind": _kind_for_path(rel_path),
            "label": _change_label(status, rel_path),
            "humanLabel": _change_label(status, rel_path),
            "before": _file_payload(before) if before else None,
            "after": _file_payload(after) if after else None,
            "sizeDelta": (after.size if after else 0) - (before.size if before else 0),
        }
        if kind == "text" and _is_text_path(rel_path, before, after):
            text = _text_diff(
                session,
                base_version,
                compare_version,
                rel_path,
                before,
                after,
                base_root=base_root,
                compare_root=compare_root,
            )
            if text is not None:
                change["textDiff"] = text
                text_diffs.append(text)
        changes.append(change)

    dataset_diffs = [
        diff for diff in (
            _dataset_diff(
                session,
                base_version,
                compare_version,
                rel_path,
                base_files.get(rel_path),
                compare_files.get(rel_path),
                base_root=base_root,
                compare_root=compare_root,
            )
            for rel_path in sorted(set(base_files) | set(compare_files))
            if _is_dataset_diff_path(rel_path)
        )
        if diff is not None
    ]
    visual_diff = _visual_diff(
        service,
        artifact=artifact,
        base_version=base_version,
        compare_version=compare_version,
        base_files=base_files,
        compare_files=compare_files,
        base_root=base_root,
        compare_root=compare_root,
        request_base_url=request_base_url,
    )
    modified = sum(1 for change in changes if change["status"] == "modified")
    summary = {
        "added": sum(1 for change in changes if change["status"] == "added"),
        "modified": modified,
        "removed": sum(1 for change in changes if change["status"] == "removed"),
        "unchanged": len(set(base_files) & set(compare_files)) - modified,
        "totalChanged": len(changes),
    }
    artifact_external_id = _external_artifact_id(artifact) if artifact else None
    fallback_artifact_id = (
        artifact_external_id
        or (str(artifact.id) if artifact is not None else None)
        or (str(compare_version.artifact_id) if compare_version is not None else None)
        or (str(base_version.artifact_id) if base_version is not None else "")
    )
    return {
        "artifactId": fallback_artifact_id,
        "artifactPath": artifact.path if artifact else None,
        "base": _ref_to_dict(
            base_version,
            manifest=base_manifest,
            session=session,
            artifact=artifact,
            artifact_external_id=artifact_external_id,
            include_files=False,
        ),
        "compare": _ref_to_dict(
            compare_version,
            manifest=compare_manifest,
            session=session,
            artifact=artifact,
            artifact_external_id=artifact_external_id,
            include_files=False,
        ),
        "summary": summary,
        "changes": changes,
        "changedFiles": changes,
        "manifestDiff": changes,
        "textDiff": "\n".join(part for part in text_diffs if part),
        "datasetDiffs": dataset_diffs,
        "datasetDiff": dataset_diffs[0] if len(dataset_diffs) == 1 else None,
        "visualDiff": visual_diff,
        "available": True,
    }


def diff_artifact_versions(*args, **kwargs) -> dict:
    return diff_versions(*args, **kwargs)


def _ref_to_dict(
    version: ArtifactVersion | None,
    *,
    manifest: SnapshotManifest | None,
    session: Session,
    artifact: Artifact | None,
    artifact_external_id: str | None,
    include_files: bool,
) -> dict:
    if version is not None:
        return version_to_dict(
            version,
            session=session,
            artifact_external_id=artifact_external_id,
            include_files=include_files,
        )
    if manifest is None:
        raise ValueError("Current draft manifest is required")
    files = [_file_payload(file) for file in manifest.files] if include_files else []
    parent_version_id = artifact.current_version_id if artifact is not None else None
    return {
        "id": "current",
        "versionId": "current",
        "artifactId": artifact_external_id or (str(artifact.id) if artifact is not None else "current"),
        "versionNumber": None,
        "label": "Current draft",
        "humanLabel": "Current draft",
        "summary": "Live artifact files that have not been saved as a checkpoint.",
        "createdAt": None,
        "fileCount": manifest.file_count,
        "totalBytes": manifest.total_bytes,
        "filesHash": manifest.files_hash,
        "manifestHash": manifest.manifest_hash,
        "operationType": "draft",
        "previewStatus": "pending",
        "publishStatus": "unpublished",
        "parentVersionId": str(parent_version_id) if parent_version_id else None,
        "restoredFromVersionId": None,
        "branchName": None,
        "forkedFromVersionId": None,
        "files": files,
        "manifest": {"fileCount": manifest.file_count, "files": files},
        "workingTree": True,
    }


def version_to_dict(
    version: ArtifactVersion,
    *,
    session: Session | None = None,
    artifact_external_id: str | None = None,
    include_files: bool = True,
) -> dict:
    files: list[dict] = []
    if include_files:
        if session is not None:
            files = [_file_payload(file) for file in _version_files(session, version.id)]
        else:
            try:
                files = [_file_payload(file) for file in sorted(version.files, key=lambda item: item.path)]
            except Exception:
                files = []
    label = version.label or _default_version_label(version)
    return {
        "id": str(version.id),
        "versionId": str(version.id),
        "artifactId": artifact_external_id or str(version.artifact_id),
        "versionNumber": version.version_number,
        "label": label,
        "humanLabel": label,
        "summary": version.prompt or "",
        "prompt": version.prompt,
        "createdAt": version.created_at.isoformat() if version.created_at else None,
        "fileCount": version.file_count,
        "totalBytes": version.total_bytes,
        "filesHash": version.files_hash,
        "manifestHash": version.manifest_hash,
        "operationType": version.operation_type,
        "previewStatus": version.preview_status,
        "publishStatus": version.publish_status,
        "sourceConversationId": str(version.source_conversation_id) if version.source_conversation_id else None,
        "sourceMessageId": str(version.source_message_id) if version.source_message_id else None,
        "parentVersionId": str(version.parent_version_id) if version.parent_version_id else None,
        "restoredFromVersionId": str(version.restored_from_version_id) if version.restored_from_version_id else None,
        "branchName": version.branch_name,
        "forkedFromVersionId": str(version.forked_from_version_id) if version.forked_from_version_id else None,
        "files": files,
        "manifest": {"fileCount": version.file_count, "files": files},
    }


def record_deployment(
    session: Session,
    version: ArtifactVersion,
    *,
    target: str,
    status: str,
    url: str | None,
    details: dict | None = None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> ArtifactDeployment:
    actor_details = _actor_details(actor_name=actor_name, actor_email=actor_email, actor_subject=actor_subject)
    deployment = ArtifactDeployment(
        artifact_id=version.artifact_id,
        version_id=version.id,
        target=target,
        status=status,
        url=url,
        details={**(details or {}), **actor_details},
    )
    if target == "preview":
        version.preview_status = status
    else:
        version.publish_status = status
    artifact = session.get(Artifact, version.artifact_id)
    if artifact is not None and _is_good_status(version.preview_status, version.publish_status):
        artifact.last_known_good_version_id = version.id
        session.add(artifact)
    session.add(version)
    session.add(deployment)
    activity_details = {"target": target, "url": url or "", **(details or {}), **actor_details}
    if artifact is not None:
        notification_event_type = _deployment_notification_event(target, status)
        if notification_event_type:
            deliveries = dispatch_project_notification(
                session,
                artifact,
                notification_event_type,
                details={
                    "versionId": str(version.id),
                    "versionNumber": version.version_number,
                    "target": target,
                    "status": status,
                    "url": url or "",
                    **(details or {}),
                    **actor_details,
                },
            )
            activity_details["notificationDeliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
    session.add(
        ArtifactActivityEvent(
            artifact_id=version.artifact_id,
            version_id=version.id,
            event_type=status,
            actor_name=actor_name,
            details=activity_details,
        )
    )
    session.commit()
    session.refresh(deployment)
    return deployment


def list_comments(
    session: Session,
    *,
    artifact_id: str | UUID | None = None,
    path: str | None = None,
    viewer_email: str | None = None,
) -> dict:
    artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
    comments = session.exec(
        select(ArtifactComment)
        .where(ArtifactComment.artifact_id == artifact.id)
        .order_by(ArtifactComment.created_at.desc())
    ).all()
    events = session.exec(
        select(ArtifactActivityEvent)
        .where(ArtifactActivityEvent.artifact_id == artifact.id)
        .order_by(ArtifactActivityEvent.created_at.desc())
        .limit(50)
    ).all()
    viewer_state = _artifact_viewer_state(session, artifact, comments, events, viewer_email=viewer_email)
    return {
        "artifactId": _external_artifact_id(artifact),
        "artifactPath": artifact.path,
        "comments": [
            comment_to_dict(
                comment,
                viewer_state=_comment_viewer_state(comment, viewer_state),
            )
            for comment in comments
        ],
        "activity": [activity_to_dict(event) for event in events],
        "viewerState": _public_viewer_state(viewer_state),
    }


def create_comment(
    session: Session,
    *,
    path: str | None = None,
    artifact_id: str | UUID | None = None,
    body: str,
    kind: str = "comment",
    anchor: dict | None = None,
    proposed_patch: dict | None = None,
    parent_comment_id: str | UUID | None = None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
    parent_uuid = _uuid_or_none(parent_comment_id)
    if parent_uuid is not None:
        parent = session.get(ArtifactComment, parent_uuid)
        if parent is None or parent.artifact_id != artifact.id:
            raise ValueError("Parent comment must belong to the same artifact")
    comment = ArtifactComment(
        artifact_id=artifact.id,
        version_id=artifact.current_version_id,
        parent_comment_id=parent_uuid,
        kind=kind if kind in {"comment", "suggestion", "review"} else "comment",
        body=body,
        anchor=anchor or {},
        proposed_patch=_normalize_proposed_patch(proposed_patch) if proposed_patch else {},
        status="open",
        actor_name=actor_name or "You",
        notification_state={},
    )
    event_type = (
        "review_requested"
        if comment.kind == "review"
        else "suggested" if comment.kind == "suggestion" else "commented"
    )
    session.add(comment)
    session.flush()
    actor_details = _actor_details(actor_name=comment.actor_name, actor_email=actor_email, actor_subject=actor_subject)
    session.add(
        ArtifactActivityEvent(
            artifact_id=artifact.id,
            version_id=artifact.current_version_id,
            event_type=event_type,
            actor_name=comment.actor_name,
            details={"commentId": str(comment.id), "kind": comment.kind, **actor_details},
        )
    )
    deliveries = dispatch_project_notification(
        session,
        artifact,
        event_type,
        details={
            "commentId": str(comment.id),
            "kind": comment.kind,
            "actorName": comment.actor_name,
            **actor_details,
            "anchor": comment.anchor or {},
        },
    )
    comment.notification_state = {
        "eventType": event_type,
        **actor_details,
        "deliveries": [delivery_to_dict(delivery) for delivery in deliveries],
    }
    session.add(comment)
    session.commit()
    session.refresh(comment)
    return {
        "comment": comment_to_dict(comment),
        **list_comments(session, artifact_id=artifact.id, viewer_email=actor_email),
    }


def set_comment_status(
    session: Session,
    comment_id: UUID,
    *,
    status: str,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    comment = session.get(ArtifactComment, comment_id)
    if comment is None:
        raise ValueError("Comment not found")
    allowed = {"open", "resolved", "accepted", "rejected"}
    clean_status = status if status in allowed else "open"
    if clean_status == "accepted" and comment.proposed_patch and not _comment_patch_applied(comment):
        return apply_comment_patch(
            session,
            comment_id,
            actor_name=actor_name,
            actor_email=actor_email,
            actor_subject=actor_subject,
        )
    comment.status = clean_status
    session.add(comment)
    event_type = {
        "open": "reopened",
        "resolved": "resolved",
        "accepted": "accepted",
        "rejected": "rejected",
    }[clean_status]
    acting_name = actor_name or comment.actor_name
    actor_details = _actor_details(actor_name=acting_name, actor_email=actor_email, actor_subject=actor_subject)
    session.add(
        ArtifactActivityEvent(
            artifact_id=comment.artifact_id,
            version_id=comment.version_id,
            event_type=event_type,
            actor_name=acting_name,
            details={"commentId": str(comment.id), **actor_details},
        )
    )
    artifact = session.get(Artifact, comment.artifact_id)
    if artifact is not None:
        deliveries = dispatch_project_notification(
            session,
            artifact,
            event_type,
            details={
                "commentId": str(comment.id),
                "status": clean_status,
                "actorName": acting_name,
                **actor_details,
            },
        )
        state = dict(comment.notification_state or {})
        state["lastStatusEvent"] = event_type
        state["lastStatusDeliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
        comment.notification_state = state
        session.add(comment)
    session.commit()
    session.refresh(comment)
    return {"comment": comment_to_dict(comment)}


def preview_comment_patch(session: Session, comment_id: UUID) -> dict:
    comment = session.get(ArtifactComment, comment_id)
    if comment is None:
        raise ValueError("Comment not found")
    artifact = session.get(Artifact, comment.artifact_id)
    if artifact is None:
        raise ValueError("Artifact not found")
    patch = _normalize_proposed_patch(comment.proposed_patch)
    if not patch["operations"]:
        raise ValueError("Suggestion has no proposed patch")
    if comment.version_id is not None and artifact.current_version_id is not None:
        if comment.version_id != artifact.current_version_id:
            raise ValueError("Suggestion was created for an older artifact version. Review the latest version before previewing it.")
    live_root = Path(artifact.path).expanduser().resolve(strict=False)
    with TemporaryDirectory(prefix="cowork-review-preview-") as tmp:
        preview_root = Path(tmp) / live_root.name
        shutil.copytree(live_root, preview_root)
        changed_paths = _apply_patch_operations(preview_root, patch)
        diff = _diff_roots(session, artifact, live_root, preview_root)
    return {
        "available": True,
        "comment": comment_to_dict(comment),
        "proposedPatch": patch,
        "changedPaths": changed_paths,
        "diff": diff,
    }


def apply_comment_patch(
    session: Session,
    comment_id: UUID,
    *,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    comment = session.get(ArtifactComment, comment_id)
    if comment is None:
        raise ValueError("Comment not found")
    artifact = session.get(Artifact, comment.artifact_id)
    if artifact is None:
        raise ValueError("Artifact not found")
    patch = _normalize_proposed_patch(comment.proposed_patch)
    if not patch["operations"]:
        raise ValueError("Suggestion has no proposed patch")
    if _comment_patch_applied(comment):
        return {"comment": comment_to_dict(comment), "alreadyApplied": True}
    if comment.version_id is not None and artifact.current_version_id is not None:
        if comment.version_id != artifact.current_version_id:
            raise ValueError("Suggestion was created for an older artifact version. Review the latest version before accepting it.")

    service = ArtifactVersionService(session)
    root = Path(artifact.path).expanduser().resolve(strict=False)
    safety = service.snapshot_artifact(
        root,
        artifact_id=artifact.id,
        label="Before accepting suggestion",
        operation_type="review_safety",
    )
    backup: Path | None = None
    with TemporaryDirectory(prefix="cowork-review-apply-", dir=root.parent) as tmp:
        staged_root = Path(tmp) / root.name
        shutil.copytree(root, staged_root)
        changed_paths = _apply_patch_operations(staged_root, patch)
        backup = service._replace_artifact_folder(root, staged_root, keep_backup=True)

    try:
        applied = service.snapshot_artifact(
            root,
            artifact_id=artifact.id,
            label="Accepted suggestion",
            operation_type="review_accept",
            prompt=comment.body,
            preview_status="ready",
        )

        comment = session.get(ArtifactComment, comment_id)
        if comment is None:
            raise ValueError("Comment not found")
        comment.status = "accepted"
        comment.version_id = applied.id
        state = dict(comment.notification_state or {})
        state["appliedVersionId"] = str(applied.id)
        state["preApplyVersionId"] = str(safety.id)
        state["changedPaths"] = changed_paths
        comment.notification_state = state
        session.add(comment)
        acting_name = actor_name or comment.actor_name
        actor_details = _actor_details(actor_name=acting_name, actor_email=actor_email, actor_subject=actor_subject)
        session.add(
            ArtifactActivityEvent(
                artifact_id=artifact.id,
                version_id=applied.id,
                event_type="accepted_patch",
                actor_name=acting_name,
                details={
                    "commentId": str(comment.id),
                    "changedPaths": changed_paths,
                    "preApplyVersionId": str(safety.id),
                    **actor_details,
                },
            )
        )
        deliveries = dispatch_project_notification(
            session,
            artifact,
            "accepted",
            details={
                "commentId": str(comment.id),
                "status": "accepted",
                "actorName": acting_name,
                "versionId": str(applied.id),
                **actor_details,
            },
        )
        state = dict(comment.notification_state or {})
        state["lastStatusEvent"] = "accepted_patch"
        state["lastStatusDeliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
        comment.notification_state = state
        session.add(comment)
        session.commit()
        session.refresh(comment)
    except Exception:
        session.rollback()
        if backup is not None and backup.exists():
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            os.replace(backup, root)
            artifact = session.get(Artifact, artifact.id)
            if artifact is not None:
                artifact.current_version_id = safety.id
                session.add(artifact)
                session.commit()
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return {
        "comment": comment_to_dict(comment),
        "createdCheckpoint": version_to_dict(safety, session=session, artifact_external_id=_external_artifact_id(artifact)),
        "version": version_to_dict(applied, session=session, artifact_external_id=_external_artifact_id(artifact)),
        "changedPaths": changed_paths,
    }


def mark_comments_read(
    session: Session,
    *,
    path: str | None = None,
    artifact_id: str | UUID | None = None,
    viewer_email: str | None,
    comment_id: str | UUID | None = None,
    activity_id: str | UUID | None = None,
) -> dict:
    artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
    collaborator = _viewer_collaborator(session, artifact, viewer_email)
    if collaborator is None:
        raise ValueError("A project collaborator is required to track review read state")

    read_at = datetime.now(timezone.utc)
    if comment_id is not None:
        comment = session.get(ArtifactComment, _uuid_or_none(comment_id))
        if comment is None or comment.artifact_id != artifact.id:
            raise ValueError("Comment not found")
        read_at = _normalize_dt(comment.created_at) or read_at
    elif activity_id is not None:
        event = session.get(ArtifactActivityEvent, _uuid_or_none(activity_id))
        if event is None or event.artifact_id != artifact.id:
            raise ValueError("Activity event not found")
        read_at = _normalize_dt(event.created_at) or read_at

    state = dict(collaborator.notification_state or {})
    artifact_states = dict(state.get("artifacts") or {})
    key = str(artifact.id)
    artifact_state = dict(artifact_states.get(key) or {})
    current_read_at = _parse_dt(artifact_state.get("lastReadAt"))
    if current_read_at is not None and current_read_at > read_at:
        read_at = current_read_at
    artifact_state["lastReadAt"] = read_at.isoformat()
    if comment_id is not None:
        artifact_state["lastSeenCommentId"] = str(comment_id)
    if activity_id is not None:
        artifact_state["lastSeenActivityId"] = str(activity_id)
    artifact_states[key] = artifact_state
    state["artifacts"] = artifact_states
    collaborator.notification_state = state
    session.add(collaborator)
    session.commit()
    session.refresh(collaborator)
    return list_comments(session, artifact_id=artifact.id, viewer_email=collaborator.email)


def _viewer_collaborator(session: Session, artifact: Artifact, viewer_email: str | None) -> ProjectCollaborator | None:
    if artifact.project_id is None or not viewer_email:
        return None
    try:
        email = normalize_email(viewer_email)
    except ValueError:
        return None
    return session.exec(
        select(ProjectCollaborator)
        .where(ProjectCollaborator.project_id == artifact.project_id)
        .where(ProjectCollaborator.email == email)
    ).first()


def _artifact_viewer_state(
    session: Session,
    artifact: Artifact,
    comments: list[ArtifactComment],
    events: list[ArtifactActivityEvent],
    *,
    viewer_email: str | None,
) -> dict:
    collaborator = _viewer_collaborator(session, artifact, viewer_email)
    if collaborator is None:
        return {"available": False, "byCommentId": {}}
    artifact_state = dict((collaborator.notification_state or {}).get("artifacts", {}).get(str(artifact.id), {}) or {})
    last_read_at = _parse_dt(artifact_state.get("lastReadAt"))
    viewer_email = collaborator.email
    role = collaborator.role

    by_comment_id: dict[str, dict] = {}
    unread_comments = 0
    open_review_requests = 0
    needs_action = 0
    unread_review_requests = 0
    for comment in comments:
        own_comment = _same_email(_comment_actor_email(comment), viewer_email)
        unread = bool(
            not own_comment
            and last_read_at is not None
            and _created_after(comment.created_at, last_read_at)
        )
        if last_read_at is None and not own_comment:
            unread = True
        closed = comment.status in {"resolved", "accepted", "rejected"}
        review_request = comment.kind == "review" and not closed
        needs = bool(review_request and role_allows(role, "review") and not own_comment)
        if unread:
            unread_comments += 1
        if review_request:
            open_review_requests += 1
            if unread:
                unread_review_requests += 1
        if needs:
            needs_action += 1
        by_comment_id[str(comment.id)] = {
            "unread": unread,
            "seen": own_comment or (last_read_at is not None and not unread),
            "needsAction": needs,
            "reviewRequest": review_request,
        }

    unread_activity = 0
    for event in events:
        own_event = _same_email(_event_actor_email(event), viewer_email)
        if own_event:
            continue
        if last_read_at is None or _created_after(event.created_at, last_read_at):
            unread_activity += 1

    return {
        "available": True,
        "collaboratorId": str(collaborator.id),
        "role": role,
        "lastReadAt": last_read_at.isoformat() if last_read_at else None,
        "unreadComments": unread_comments,
        "unreadActivity": unread_activity,
        "needsAction": needs_action,
        "reviewRequests": {
            "open": open_review_requests,
            "needsAction": needs_action,
            "unread": unread_review_requests,
        },
        "byCommentId": by_comment_id,
    }


def _comment_viewer_state(comment: ArtifactComment, viewer_state: dict) -> dict:
    by_comment_id = viewer_state.get("byCommentId") if isinstance(viewer_state, dict) else None
    if not isinstance(by_comment_id, dict):
        return {}
    state = by_comment_id.get(str(comment.id))
    return dict(state) if isinstance(state, dict) else {}


def _public_viewer_state(viewer_state: dict) -> dict:
    if not viewer_state.get("available"):
        return {"available": False}
    return {
        key: value
        for key, value in viewer_state.items()
        if key != "byCommentId"
    }


def _comment_actor_email(comment: ArtifactComment) -> str | None:
    state = comment.notification_state or {}
    value = state.get("actorEmail") if isinstance(state, dict) else None
    return str(value).strip().lower() if isinstance(value, str) and value.strip() else None


def _event_actor_email(event: ArtifactActivityEvent) -> str | None:
    details = event.details or {}
    value = details.get("actorEmail") if isinstance(details, dict) else None
    return str(value).strip().lower() if isinstance(value, str) and value.strip() else None


def _same_email(left: str | None, right: str | None) -> bool:
    return bool(left and right and left.strip().lower() == right.strip().lower())


def _created_after(created_at, last_read_at: datetime) -> bool:
    created = _normalize_dt(created_at)
    return bool(created is not None and created > last_read_at)


def _parse_dt(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _normalize_dt(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _normalize_dt(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _normalize_dt(value) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def comment_to_dict(comment: ArtifactComment, *, viewer_state: dict | None = None) -> dict:
    return {
        "id": str(comment.id),
        "artifactId": str(comment.artifact_id),
        "versionId": str(comment.version_id) if comment.version_id else None,
        "parentCommentId": str(comment.parent_comment_id) if comment.parent_comment_id else None,
        "kind": comment.kind,
        "body": comment.body,
        "text": comment.body,
        "anchor": comment.anchor or {},
        "proposedPatch": comment.proposed_patch or {},
        "status": comment.status,
        "resolved": comment.status == "resolved",
        "actorName": comment.actor_name,
        "createdAt": comment.created_at.isoformat() if comment.created_at else None,
        "notificationState": comment.notification_state or {},
        "viewerState": viewer_state or {},
    }


def activity_to_dict(event: ArtifactActivityEvent) -> dict:
    return {
        "id": str(event.id),
        "artifactId": str(event.artifact_id),
        "versionId": str(event.version_id) if event.version_id else None,
        "eventType": event.event_type,
        "actorName": event.actor_name,
        "details": event.details or {},
        "createdAt": event.created_at.isoformat() if event.created_at else None,
    }


def _actor_details(
    *,
    actor_name: str | None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    details = {"actorName": actor_name or ""}
    if actor_email:
        details["actorEmail"] = actor_email
    if actor_subject:
        details["actorSubject"] = actor_subject
    return details


def _is_good_status(preview_status: str, publish_status: str) -> bool:
    return preview_status in {"ready", "ok", "success"} or publish_status in {"published", "ready", "success"}


def _snapshot_notification_event(operation_type: str) -> str | None:
    clean = (operation_type or "").strip().lower()
    return {
        "restore": "restored",
        "restore_deleted": "restored",
        "fork": "forked",
        "generated_update": "generated_updated",
    }.get(clean)


def _deployment_notification_event(target: str, status: str) -> str | None:
    clean_target = (target or "").strip().lower()
    clean_status = (status or "").strip().lower()
    if clean_target == "publish":
        if clean_status == "published":
            return "published"
        if clean_status == "failed":
            return "publish_failed"
    if clean_target == "preview" and clean_status == "failed":
        return "preview_failed"
    return None


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


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned[:64].strip("-._") or "artifact-copy"


def _unique_artifact_folder(parent: Path, slug: str) -> Path:
    base = _safe_slug(slug)
    candidate = parent / base
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    index = 2
    while True:
        candidate = parent / f"{base}-{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def _artifact_folder_from_request(
    session: Session,
    *,
    artifact_id: str | UUID | None,
    path: str | None,
    create: bool = False,
) -> Path:
    if path:
        folder = _artifact_folder_from_path(session, path, create=create)
        if create and artifact_id is not None:
            _ensure_metadata(folder, artifact_id)
        return folder
    artifact = _artifact_by_identifier(session, artifact_id)
    if artifact is not None:
        return Path(artifact.path)
    folder = _folder_by_external_id(str(artifact_id or ""))
    if folder is None:
        raise FileNotFoundError("Artifact not found")
    return folder


def _artifact_from_identifier_or_path(
    session: Session,
    *,
    artifact_id: str | UUID | None,
    path: str | None,
) -> Artifact:
    if path:
        return get_or_create_artifact_for_path(session, path, artifact_id=artifact_id)
    artifact = _artifact_by_identifier(session, artifact_id)
    if artifact is not None:
        return artifact
    folder = _folder_by_external_id(str(artifact_id or ""))
    if folder is None:
        raise FileNotFoundError("Artifact not found")
    return get_or_create_artifact_for_path(session, str(folder), artifact_id=artifact_id)


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
    slug_matches = session.exec(select(Artifact).where(Artifact.slug == clean)).all()
    if len(slug_matches) == 1:
        return slug_matches[0]
    if len(slug_matches) > 1:
        raise ValueError("Artifact identifier is ambiguous; include an artifact path or project")
    return _artifact_by_deleted_external_id(session, clean)


def _artifact_by_deleted_external_id(session: Session, artifact_id: str) -> Artifact | None:
    if not artifact_id:
        return None
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
        if artifact_id not in identifiers:
            continue
        artifact = session.get(Artifact, event.artifact_id)
        if artifact is not None:
            return artifact
    return None


def _deleted_external_artifact_id(session: Session | None, artifact: Artifact) -> str | None:
    if session is None:
        return None
    event = session.exec(
        select(ArtifactActivityEvent)
        .where(ArtifactActivityEvent.artifact_id == artifact.id)
        .where(ArtifactActivityEvent.event_type == "deleted")
        .order_by(ArtifactActivityEvent.created_at.desc())
    ).first()
    if event is None:
        return None
    details = event.details or {}
    value = details.get("externalArtifactId") or details.get("artifactId")
    return value if isinstance(value, str) and value else None


def _deleted_original_path(session: Session | None, artifact: Artifact) -> str | None:
    """The folder path an artifact had when it was deleted.

    On delete we tombstone ``artifact.path`` (so a new artifact can reuse the
    original path) but stash the original in the "deleted" event details. Recovery
    uses this to restore the artifact back to where it lived.
    """
    if session is None:
        return None
    event = session.exec(
        select(ArtifactActivityEvent)
        .where(ArtifactActivityEvent.artifact_id == artifact.id)
        .where(ArtifactActivityEvent.event_type == "deleted")
        .order_by(ArtifactActivityEvent.created_at.desc())
    ).first()
    if event is None:
        return None
    value = (event.details or {}).get("path")
    return value if isinstance(value, str) and value else None


def _folder_by_external_id(artifact_id: str) -> Path | None:
    if not artifact_id:
        return None
    matches: list[Path] = []
    for root in _scan_artifact_dirs():
        try:
            folders = sorted(root.iterdir())
        except OSError:
            continue
        for folder in folders:
            if not folder.is_dir():
                continue
            metadata = _load_metadata(folder) or {}
            identifiers = {folder.name, str(metadata.get("id") or ""), str(metadata.get("slug") or "")}
            if artifact_id in identifiers:
                matches.append(folder.resolve())
    if not matches:
        return None
    unique = {str(path): path for path in matches}
    if len(unique) > 1:
        raise ValueError("Artifact identifier is ambiguous; include an artifact path or project")
    return next(iter(unique.values()))


def _artifact_folder_from_path(session: Session, raw_path: str, *, create: bool = False) -> Path:
    try:
        resolved = resolve_artifact_path(raw_path, allow_dir=True)
    except (FileNotFoundError, ValueError):
        target = Path(raw_path).expanduser().resolve(strict=False)
        if create:
            if not _is_known_project_artifact_folder(session, target):
                raise ValueError("Artifact path must be inside a project artifacts folder")
            target.mkdir(parents=True, exist_ok=True)
            return target
        if not target.exists():
            raise
        folder = target if target.is_dir() else _artifact_root_for_file(target)
        if not _is_known_project_artifact_folder(session, folder):
            raise ValueError("Artifact path must be inside a project artifacts folder")
        resolved = target
    if resolved is None:
        raise FileNotFoundError("Artifact not found")
    if resolved.is_dir():
        return resolved.resolve()
    return _artifact_root_for_file(resolved)


def _is_known_project_artifact_folder(session: Session, folder: Path) -> bool:
    resolved = folder.expanduser().resolve(strict=False)
    for project in session.exec(select(Project)).all():
        try:
            artifacts_root = Path(project.path).expanduser().resolve(strict=False) / ".anton" / "artifacts"
            rel = resolved.relative_to(artifacts_root)
        except (OSError, ValueError, RuntimeError):
            continue
        return len(rel.parts) == 1 and rel.parts[0] not in {"", ".", ".."}
    return False


def _artifact_root_for_file(path: Path) -> Path:
    current = path.parent.resolve()
    while current.parent != current:
        if (current / "metadata.json").is_file():
            return current
        if current.name == "artifacts" and current.parent.name == ".anton":
            break
        current = current.parent
    return path.parent.resolve()


def _project_id_for_folder(session: Session, folder: Path) -> UUID | None:
    resolved = folder.resolve(strict=False)
    for project in session.exec(select(Project)).all():
        try:
            resolved.relative_to(Path(project.path).resolve(strict=False))
            return project.id
        except ValueError:
            continue
    return None


def _external_artifact_id(artifact: Artifact | None, session: Session | None = None) -> str:
    if artifact is None:
        return ""
    metadata = _load_metadata(Path(artifact.path)) or {}
    return str(metadata.get("id") or metadata.get("slug") or _deleted_external_artifact_id(session, artifact) or artifact.slug or artifact.id)


def _artifact_integration(folder: Path) -> dict:
    metadata = _load_metadata(folder) or {}
    files = _user_files(folder)
    primary = _pick_primary(folder, files, primary_hint=metadata.get("primary"))
    primary_path = str(primary) if primary is not None else str(folder)
    artifact_type = str(metadata.get("type") or "mixed")
    primary_ext = primary.suffix.lower() if primary is not None else ""
    access = _published_access_for(folder, primary)
    return {
        "artifact": {
            "id": str(metadata.get("id") or metadata.get("slug") or folder.name),
            "slug": str(metadata.get("slug") or folder.name),
            "title": str(metadata.get("name") or folder.name),
            "description": str(metadata.get("description") or ""),
            "type": artifact_type,
            "kind": KIND_BY_TYPE.get(artifact_type) or KIND_BY_EXT.get(primary_ext, "File"),
            "path": primary_path,
            "folder": str(folder),
            "primary": metadata.get("primary") or None,
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


def _ensure_metadata(folder: Path, artifact_id: str | UUID, *, body: dict | None = None) -> None:
    path = folder / "metadata.json"
    body = body or {}
    metadata = _load_metadata(folder) or {}
    if metadata and not body:
        return
    title = _body_str(body, "title") or _body_str(body, "name")
    description = _body_str(body, "description")
    artifact_type = _body_str(body, "artifact_type") or _body_str(body, "artifactType") or _body_str(body, "type")
    primary = _body_str(body, "primary")
    metadata = {
        **metadata,
        "id": str(artifact_id),
        "slug": metadata.get("slug") or folder.name,
        "name": title or metadata.get("name") or folder.name.replace("-", " ").strip().title() or folder.name,
        "description": description if description is not None else metadata.get("description", ""),
        "type": artifact_type or metadata.get("type") or "mixed",
    }
    if primary or metadata.get("primary"):
        metadata["primary"] = primary or metadata.get("primary")
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _metadata_str(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _body_str(body: dict, key: str) -> str | None:
    value = body.get(key)
    return value if isinstance(value, str) and value else None


def _target_project_from_body(session: Session, body: dict) -> Project | None:
    project_id = (
        body.get("target_project_id")
        or body.get("targetProjectId")
        or body.get("project_id")
        or body.get("projectId")
    )
    if project_id:
        project = session.get(Project, _uuid_or_none(project_id))
        if project is None:
            raise ValueError("Target project not found")
        return project

    project_ref = _body_str(body, "target_project") or _body_str(body, "targetProject") or _body_str(body, "project")
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
    raise ValueError("Target project not found")


def _get_version(session: Session, version_id: UUID | str) -> ArtifactVersion:
    clean_id = version_id if isinstance(version_id, UUID) else UUID(str(version_id))
    version = session.get(ArtifactVersion, clean_id)
    if version is None:
        raise ValueError("Artifact version not found")
    return version


def _uuid_or_none(value: str | UUID | None) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    return UUID(str(value))


def _resolve_version_ref(versions: list[ArtifactVersion], ref: str | None) -> ArtifactVersion:
    clean = (ref or "latest").strip()
    if clean in {"latest", "head"}:
        return versions[0]
    for version in versions:
        if str(version.id) == clean or str(version.version_number) == clean:
            return version
    raise FileNotFoundError("Checkpoint not found")


def _is_current_ref(ref: str | None) -> bool:
    return (ref or "").strip().lower() in {"current", "working", "workspace", "draft"}


def _manifest_file_map(manifest: SnapshotManifest) -> dict[str, SnapshotFile]:
    return {file.path: file for file in manifest.files}


def _version_files(session: Session, version_id: UUID) -> list[ArtifactVersionFile]:
    return list(
        session.exec(
            select(ArtifactVersionFile)
            .where(ArtifactVersionFile.version_id == version_id)
            .order_by(ArtifactVersionFile.path)
        ).all()
    )


def _version_file_map(session: Session, version_id: UUID) -> dict[str, ArtifactVersionFile]:
    return {file.path: file for file in _version_files(session, version_id)}


def _file_payload(file: ArtifactVersionFile) -> dict:
    return {
        "path": file.path,
        "name": Path(file.path).name,
        "sha256": file.content_hash,
        "contentHash": file.content_hash,
        "size": file.size,
        "kind": _kind_for_path(file.path),
        "text": _is_text_path(file.path, file, file),
    }


def _kind_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".js", ".ts", ".tsx", ".css", ".sql", ".sh", ".html"}:
        return "Code"
    return KIND_BY_EXT.get(suffix, "File")


def _change_label(status: str, path: str) -> str:
    verbs = {"added": "Added", "modified": "Updated", "removed": "Removed"}
    return f"{verbs.get(status, 'Changed')} {path}"


def _is_text_path(path: str, before: ArtifactVersionFile | None, after: ArtifactVersionFile | None) -> bool:
    suffix = Path(path).suffix.lower()
    size = max(before.size if before else 0, after.size if after else 0)
    return size <= 200_000 and (suffix in TEXT_EXTENSIONS or suffix in {".html", ".css", ".js", ".json", ".csv", ".txt"})


def _text_diff(
    session: Session,
    base_version: ArtifactVersion | None,
    compare_version: ArtifactVersion | None,
    rel_path: str,
    before: ArtifactVersionFile | SnapshotFile | None,
    after: ArtifactVersionFile | SnapshotFile | None,
    *,
    base_root: Path | None = None,
    compare_root: Path | None = None,
) -> str | None:
    old_text = _entry_text(session, before, live_root=base_root) if before is not None else ""
    new_text = _entry_text(session, after, live_root=compare_root) if after is not None else ""
    if old_text is None or new_text is None:
        return None
    base_label = f"v{base_version.version_number}" if base_version is not None else "current"
    compare_label = f"v{compare_version.version_number}" if compare_version is not None else "current"
    return "\n".join(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"{base_label}/{rel_path}",
            tofile=f"{compare_label}/{rel_path}",
            lineterm="",
        )
    )


def _entry_text(
    session: Session,
    file: ArtifactVersionFile | SnapshotFile | None,
    *,
    live_root: Path | None = None,
) -> str | None:
    if file is None:
        return ""
    if live_root is not None:
        path = live_root / Path(file.path)
    else:
        blob_path = getattr(file, "blob_path", None)
        if not blob_path:
            return None
        path = ArtifactVersionService(session).store_root / blob_path
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _blob_text(session: Session, file: ArtifactVersionFile | None) -> str | None:
    return _entry_text(session, file)


def _dataset_diff(
    session: Session,
    base_version: ArtifactVersion | None,
    compare_version: ArtifactVersion | None,
    rel_path: str,
    before: ArtifactVersionFile | SnapshotFile | None,
    after: ArtifactVersionFile | SnapshotFile | None,
    *,
    base_root: Path | None = None,
    compare_root: Path | None = None,
) -> dict | None:
    before_table = _dataset_table(rel_path, _entry_text(session, before, live_root=base_root)) if before is not None else ([], [])
    after_table = _dataset_table(rel_path, _entry_text(session, after, live_root=compare_root)) if after is not None else ([], [])
    if before_table is None or after_table is None:
        return None
    before_columns, before_rows = before_table
    after_columns, after_rows = after_table

    before_column_set = set(before_columns)
    after_column_set = set(after_columns)
    key = _dataset_key(before_columns, after_columns)
    before_index = _index_rows(before_rows, key)
    after_index = _index_rows(after_rows, key)
    changed_rows = []
    for row_key in sorted(set(before_index) | set(after_index)):
        old = before_index.get(row_key)
        new = after_index.get(row_key)
        if old is None:
            changed_rows.append({"key": row_key, "status": "added", "before": None, "after": new})
        elif new is None:
            changed_rows.append({"key": row_key, "status": "removed", "before": old, "after": None})
        elif old != new:
            changed_rows.append({"key": row_key, "status": "modified", "before": old, "after": new})

    return {
        "path": rel_path,
        "baseVersionId": str(base_version.id) if base_version is not None else "current",
        "compareVersionId": str(compare_version.id) if compare_version is not None else "current",
        "rowKey": key or "__row_number__",
        "schema": {
            "before": before_columns,
            "after": after_columns,
            "added": sorted(after_column_set - before_column_set),
            "removed": sorted(before_column_set - after_column_set),
            "unchanged": [column for column in before_columns if column in after_column_set],
        },
        "rows": {
            "before": len(before_rows),
            "after": len(after_rows),
            "added": sum(1 for row in changed_rows if row["status"] == "added"),
            "removed": sum(1 for row in changed_rows if row["status"] == "removed"),
            "modified": sum(1 for row in changed_rows if row["status"] == "modified"),
        },
        "changedRows": changed_rows[:50],
        "changedRowsTruncated": len(changed_rows) > 50,
    }


def _visual_diff(
    service: ArtifactVersionService,
    *,
    artifact: Artifact | None,
    base_version: ArtifactVersion | None,
    compare_version: ArtifactVersion | None,
    base_files: dict[str, ArtifactVersionFile | SnapshotFile],
    compare_files: dict[str, ArtifactVersionFile | SnapshotFile],
    base_root: Path | None,
    compare_root: Path | None,
    request_base_url: str | None,
) -> dict:
    is_runtime_visual = _is_runtime_visual_artifact(artifact)
    base_path, compare_path = _visual_html_paths(artifact, base_files, compare_files)
    if base_path is None or compare_path is None:
        return _visual_unavailable(
            "no-html-entry",
            "Visual comparison is available for HTML artifacts once both versions include an HTML file.",
        )
    try:
        preview_ref = _runtime_visual_preview_ref if is_runtime_visual else _visual_preview_ref
        base_ref = preview_ref(
            service,
            version=base_version,
            live_root=base_root,
            rel_path=base_path,
            file=base_files.get(base_path),
            side="base",
        )
        compare_ref = preview_ref(
            service,
            version=compare_version,
            live_root=compare_root,
            rel_path=compare_path,
            file=compare_files.get(compare_path),
            side="compare",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _visual_unavailable("preview-unavailable", str(exc))
    fallback = {
        "available": True,
        "kind": "runtime-preview" if is_runtime_visual else "html-preview",
        "mode": "side-by-side",
        "base": base_ref,
        "compare": compare_ref,
        "limitations": ["runtime-proxy-preview"] if is_runtime_visual else ["static-html-preview"],
    }
    if is_runtime_visual:
        if not request_base_url:
            return {
                **fallback,
                "screenshotUnavailable": "request-origin-unavailable",
            }
        try:
            output_dir = _visual_screenshot_dir(service, base_ref, compare_ref)
            screenshot = render_url_screenshot_diff(
                _absolute_api_url(request_base_url, base_ref["relUrl"]),
                _absolute_api_url(request_base_url, compare_ref["relUrl"]),
                output_dir,
            )
            token = register_preview_mount(
                output_dir,
                salt=f"runtime-visual-diff:{base_ref['id']}:{compare_ref['id']}:{base_ref['contentHash']}:{compare_ref['contentHash']}",
            )
            base_screenshot = _preview_asset_url(token, "base.png")
            compare_screenshot = _preview_asset_url(token, "compare.png")
            diff_screenshot = _preview_asset_url(token, "diff.png")
            return {
                **fallback,
                "kind": "screenshot-pixel-diff",
                "mode": "visual-diff",
                "base": {**base_ref, "screenshotRelUrl": base_screenshot, "screenshotUrl": base_screenshot},
                "compare": {**compare_ref, "screenshotRelUrl": compare_screenshot, "screenshotUrl": compare_screenshot},
                "diff": {"imageRelUrl": diff_screenshot, "imageUrl": diff_screenshot},
                "changedPixels": screenshot["changedPixels"],
                "totalPixels": screenshot["totalPixels"],
                "ratio": screenshot["ratio"],
                "threshold": screenshot["threshold"],
                "viewport": screenshot["viewport"],
                "width": screenshot["width"],
                "height": screenshot["height"],
                "limitations": ["runtime-proxy-screenshot"],
            }
        except ScreenshotDiffUnavailable as exc:
            return {
                **fallback,
                "screenshotUnavailable": exc.reason,
            }
        except Exception:
            return {
                **fallback,
                "screenshotUnavailable": "runtime-screenshot-render-failed",
            }
    try:
        base_entry = _visual_entry_path(
            service,
            version=base_version,
            live_root=base_root,
            rel_path=base_path,
            file=base_files.get(base_path),
            side="base",
        )
        compare_entry = _visual_entry_path(
            service,
            version=compare_version,
            live_root=compare_root,
            rel_path=compare_path,
            file=compare_files.get(compare_path),
            side="compare",
        )
        output_dir = _visual_screenshot_dir(service, base_ref, compare_ref)
        screenshot = render_static_html_screenshot_diff(base_entry, compare_entry, output_dir)
        token = register_preview_mount(
            output_dir,
            salt=f"visual-diff:{base_ref['id']}:{compare_ref['id']}:{base_ref['contentHash']}:{compare_ref['contentHash']}",
        )
        base_screenshot = _preview_asset_url(token, "base.png")
        compare_screenshot = _preview_asset_url(token, "compare.png")
        diff_screenshot = _preview_asset_url(token, "diff.png")
        return {
            **fallback,
            "kind": "screenshot-pixel-diff",
            "mode": "visual-diff",
            "base": {**base_ref, "screenshotRelUrl": base_screenshot, "screenshotUrl": base_screenshot},
            "compare": {**compare_ref, "screenshotRelUrl": compare_screenshot, "screenshotUrl": compare_screenshot},
            "diff": {"imageRelUrl": diff_screenshot, "imageUrl": diff_screenshot},
            "changedPixels": screenshot["changedPixels"],
            "totalPixels": screenshot["totalPixels"],
            "ratio": screenshot["ratio"],
            "threshold": screenshot["threshold"],
            "viewport": screenshot["viewport"],
            "width": screenshot["width"],
            "height": screenshot["height"],
            "limitations": ["static-file-screenshot"],
        }
    except ScreenshotDiffUnavailable as exc:
        return {
            **fallback,
            "screenshotUnavailable": exc.reason,
        }
    except Exception:
        return {
            **fallback,
            "screenshotUnavailable": "screenshot-render-failed",
        }


def _visual_unavailable(reason: str, message: str) -> dict:
    return {
        "available": False,
        "kind": "html-preview",
        "mode": "side-by-side",
        "reason": reason,
        "message": message,
    }


def _is_runtime_visual_artifact(artifact: Artifact | None) -> bool:
    artifact_type = str(getattr(artifact, "artifact_type", "") or "").lower()
    if artifact_type.startswith("fullstack") or "stateful" in artifact_type:
        return True
    if artifact is None:
        return False
    try:
        metadata = _load_metadata(Path(artifact.path)) or {}
    except Exception:
        metadata = {}
    metadata_type = str(metadata.get("type") or "").lower()
    return metadata_type.startswith("fullstack") or "stateful" in metadata_type


def _visual_html_paths(
    artifact: Artifact | None,
    base_files: dict[str, ArtifactVersionFile | SnapshotFile],
    compare_files: dict[str, ArtifactVersionFile | SnapshotFile],
) -> tuple[str | None, str | None]:
    candidates = []
    if artifact is not None:
        try:
            metadata = _load_metadata(Path(artifact.path)) or {}
        except Exception:
            metadata = {}
        for key in ("primary", "entry", "file"):
            value = metadata.get(key)
            if isinstance(value, str) and Path(value).suffix.lower() == ".html":
                candidates.append(value)
    candidates.extend(["index.html", "slides.html", "deck.html"])
    html_paths = sorted(
        path
        for path in set(base_files) | set(compare_files)
        if Path(path).suffix.lower() == ".html"
    )
    candidates.extend(path for path in html_paths if path not in candidates)

    for path in candidates:
        if path in base_files and path in compare_files:
            return path, path
    base_path = next((path for path in candidates if path in base_files), None)
    compare_path = next((path for path in candidates if path in compare_files), None)
    return base_path, compare_path


def _visual_preview_ref(
    service: ArtifactVersionService,
    *,
    version: ArtifactVersion | None,
    live_root: Path | None,
    rel_path: str,
    file: ArtifactVersionFile | SnapshotFile | None,
    side: str,
) -> dict:
    entry = _visual_entry_path(
        service,
        version=version,
        live_root=live_root,
        rel_path=rel_path,
        file=file,
        side=side,
    )
    root = entry.parent
    for _ in Path(rel_path).parts[:-1]:
        root = root.parent
    version_id = "current" if version is None else str(version.id)
    label = "Current draft" if version is None else version.label or _default_version_label(version)
    token = register_preview_mount(
        root,
        salt=f"{side}:{version_id}:{rel_path}:{file.content_hash}",
    )
    return {
        "id": version_id,
        "versionId": version_id,
        "label": label,
        "path": rel_path,
        "contentHash": file.content_hash,
        "relUrl": _preview_asset_url(token, rel_path),
    }


def _runtime_visual_preview_ref(
    service: ArtifactVersionService,
    *,
    version: ArtifactVersion | None,
    live_root: Path | None,
    rel_path: str,
    file: ArtifactVersionFile | SnapshotFile | None,
    side: str,
) -> dict:
    entry = _visual_entry_path(
        service,
        version=version,
        live_root=live_root,
        rel_path=rel_path,
        file=file,
        side=side,
    )
    root = entry.parent
    for _ in Path(rel_path).parts[:-1]:
        root = root.parent
    version_id = "current" if version is None else str(version.id)
    label = "Current draft" if version is None else version.label or _default_version_label(version)
    token = register_preview_mount(
        root,
        salt=f"runtime:{side}:{version_id}:{rel_path}:{file.content_hash}",
    )
    rel_url = _proxy_preview_url(token)
    return {
        "id": version_id,
        "versionId": version_id,
        "label": label,
        "path": rel_path,
        "entryPath": rel_path,
        "contentHash": file.content_hash,
        "relUrl": rel_url,
        "proxyRelUrl": rel_url,
    }


def _visual_entry_path(
    service: ArtifactVersionService,
    *,
    version: ArtifactVersion | None,
    live_root: Path | None,
    rel_path: str,
    file: ArtifactVersionFile | SnapshotFile | None,
    side: str,
) -> Path:
    if file is None:
        raise FileNotFoundError(f"Missing HTML file for {side} preview: {rel_path}")
    if version is None:
        if live_root is None:
            raise ValueError("Current draft preview root is unavailable")
        root = Path(live_root).expanduser().resolve(strict=False)
    else:
        root = _materialized_preview_root(service, version)
    entry = root / Path(rel_path)
    if not entry.is_file():
        raise FileNotFoundError(f"Preview file does not exist: {rel_path}")
    return entry


def _visual_screenshot_dir(service: ArtifactVersionService, base_ref: dict, compare_ref: dict) -> Path:
    key = _sha256_bytes(
        _canonical_json(
            {
                "base": {
                    "id": base_ref.get("id"),
                    "path": base_ref.get("path"),
                    "hash": base_ref.get("contentHash"),
                },
                "compare": {
                    "id": compare_ref.get("id"),
                    "path": compare_ref.get("path"),
                    "hash": compare_ref.get("contentHash"),
                },
            }
        )
    )[:24]
    return service.store_root / "previews" / "visual-diffs" / key


def _materialized_preview_root(service: ArtifactVersionService, version: ArtifactVersion) -> Path:
    target = service.store_root / "previews" / "versions" / str(version.artifact_id) / str(version.id)
    service.materialize_version(version.id, target, clean=True)
    service.write_version_housekeeping(version.id, target)
    return target


def _preview_asset_url(token: str, rel_path: str) -> str:
    rel = _safe_relative_path(rel_path)
    encoded = "/".join(quote(part) for part in rel.parts)
    return f"/artifacts/preview-asset/{quote(token)}/{encoded}"


def _proxy_preview_url(token: str) -> str:
    return f"/artifacts/proxy/{quote(token)}/"


def _absolute_api_url(request_base_url: str, rel_url: str) -> str:
    base = str(request_base_url or "").rstrip("/")
    rel = str(rel_url or "")
    if not rel.startswith("/"):
        rel = f"/{rel}"
    if base.endswith("/api/v1"):
        return f"{base}{rel}"
    return f"{base}/api/v1{rel}"


def _diff_roots(session: Session, artifact: Artifact, base_root: Path, compare_root: Path) -> dict:
    service = ArtifactVersionService(session)
    base_manifest = service.scan_manifest(base_root)
    compare_manifest = service.scan_manifest(compare_root)
    base_files = _manifest_file_map(base_manifest)
    compare_files = _manifest_file_map(compare_manifest)
    changes = []
    text_diffs = []
    for rel_path in sorted(set(base_files) | set(compare_files)):
        before = base_files.get(rel_path)
        after = compare_files.get(rel_path)
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        elif before.content_hash != after.content_hash:
            status = "modified"
        else:
            continue
        change = {
            "path": rel_path,
            "status": status,
            "kind": _kind_for_path(rel_path),
            "label": _change_label(status, rel_path),
            "humanLabel": _change_label(status, rel_path),
            "before": _file_payload(before) if before else None,
            "after": _file_payload(after) if after else None,
            "sizeDelta": (after.size if after else 0) - (before.size if before else 0),
        }
        if _is_text_path(rel_path, before, after):
            text = _text_diff(
                session,
                None,
                None,
                rel_path,
                before,
                after,
                base_root=base_root,
                compare_root=compare_root,
            )
            if text is not None:
                change["textDiff"] = text
                text_diffs.append(text)
        changes.append(change)

    dataset_diffs = [
        diff for diff in (
            _dataset_diff(
                session,
                None,
                None,
                rel_path,
                base_files.get(rel_path),
                compare_files.get(rel_path),
                base_root=base_root,
                compare_root=compare_root,
            )
            for rel_path in sorted(set(base_files) | set(compare_files))
            if _is_dataset_diff_path(rel_path)
        )
        if diff is not None
    ]
    modified = sum(1 for change in changes if change["status"] == "modified")
    return {
        "artifactId": _external_artifact_id(artifact),
        "artifactPath": artifact.path,
        "base": {
            "id": "current",
            "label": "Current draft",
            "fileCount": base_manifest.file_count,
            "filesHash": base_manifest.files_hash,
            "manifestHash": base_manifest.manifest_hash,
        },
        "compare": {
            "id": "proposed",
            "label": "Proposed change",
            "fileCount": compare_manifest.file_count,
            "filesHash": compare_manifest.files_hash,
            "manifestHash": compare_manifest.manifest_hash,
        },
        "summary": {
            "added": sum(1 for change in changes if change["status"] == "added"),
            "modified": modified,
            "removed": sum(1 for change in changes if change["status"] == "removed"),
            "unchanged": len(set(base_files) & set(compare_files)) - modified,
            "totalChanged": len(changes),
        },
        "changes": changes,
        "changedFiles": changes,
        "manifestDiff": changes,
        "textDiff": "\n".join(part for part in text_diffs if part),
        "datasetDiffs": dataset_diffs,
        "datasetDiff": dataset_diffs[0] if len(dataset_diffs) == 1 else None,
    }


def _comment_patch_applied(comment: ArtifactComment) -> bool:
    state = comment.notification_state or {}
    return bool(state.get("appliedVersionId"))


def _normalize_proposed_patch(patch: dict | None) -> dict:
    if not patch:
        return {"operations": []}
    if not isinstance(patch, dict):
        raise ValueError("Proposed patch must be an object")
    raw_operations = patch.get("operations")
    if raw_operations is None:
        raw_operations = [patch] if (patch.get("op") or patch.get("type")) else []
    if not isinstance(raw_operations, list):
        raise ValueError("Proposed patch operations must be a list")
    operations = []
    for raw in raw_operations:
        if not isinstance(raw, dict):
            raise ValueError("Each proposed patch operation must be an object")
        op_type = str(raw.get("type") or raw.get("op") or "").strip().lower().replace("-", "_")
        if op_type == "replace":
            op_type = "replace_text" if (raw.get("find") or raw.get("findText")) is not None else "replace_file"
        if op_type in {"write", "write_file"}:
            op_type = "replace_file"
        if op_type in {"delete", "remove"}:
            op_type = "remove_file"
        if op_type not in {"replace_text", "replace_file", "append_text", "remove_file"}:
            raise ValueError(f"Unsupported proposed patch operation: {op_type or 'missing'}")
        rel_path = str(raw.get("path") or raw.get("file") or "").strip()
        if not rel_path:
            raise ValueError("Proposed patch operation path is required")
        _safe_relative_path(rel_path)
        operation = {"type": op_type, "path": rel_path}
        if op_type == "replace_text":
            find = raw.get("find") if raw.get("find") is not None else raw.get("findText")
            replace = raw.get("replace") if raw.get("replace") is not None else raw.get("replaceText")
            if not isinstance(find, str) or not find:
                raise ValueError("replace_text requires a non-empty find string")
            if not isinstance(replace, str):
                raise ValueError("replace_text requires a replace string")
            operation["find"] = find
            operation["replace"] = replace
            if raw.get("all") is not None:
                operation["all"] = bool(raw.get("all"))
            if raw.get("count") is not None:
                operation["count"] = int(raw.get("count") or 0)
            if raw.get("expectedCount") is not None:
                operation["expectedCount"] = int(raw.get("expectedCount") or 0)
        elif op_type in {"replace_file", "append_text"}:
            content = raw.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{op_type} requires content")
            operation["content"] = content
        operations.append(operation)
    return {"operations": operations}


def _apply_patch_operations(root: Path, patch: dict) -> list[str]:
    root = root.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact folder does not exist: {root}")
    changed: list[str] = []
    for operation in _normalize_proposed_patch(patch)["operations"]:
        rel = _safe_relative_path(operation["path"])
        target = root.joinpath(*rel.parts).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Proposed patch path escapes artifact folder: {operation['path']}") from exc
        op_type = operation["type"]
        if op_type == "replace_text":
            if not target.is_file():
                raise FileNotFoundError(f"Patch target does not exist: {operation['path']}")
            original = target.read_text(encoding="utf-8", errors="replace")
            find = operation["find"]
            actual_count = original.count(find)
            if actual_count == 0:
                raise ValueError(f"Patch text was not found in {operation['path']}")
            expected_count = operation.get("expectedCount")
            if expected_count is not None and actual_count != expected_count:
                raise ValueError(f"Patch text matched {actual_count} times in {operation['path']}, expected {expected_count}")
            count = -1 if operation.get("all") else int(operation.get("count") or 1)
            updated = original.replace(find, operation["replace"], count if count > 0 else -1)
            target.write_text(updated, encoding="utf-8")
        elif op_type == "replace_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(operation["content"], encoding="utf-8")
        elif op_type == "append_text":
            current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(current + operation["content"], encoding="utf-8")
        elif op_type == "remove_file":
            if not target.is_file():
                raise FileNotFoundError(f"Patch target does not exist: {operation['path']}")
            target.unlink()
        changed.append(operation["path"])
    return sorted(set(changed))


def _csv_table(text: str | None) -> tuple[list[str], list[dict[str, str]]] | None:
    if text is None:
        return None
    try:
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            return ([], [])
        columns = [str(column or "") for column in reader.fieldnames if column is not None]
        rows = [
            {str(key or ""): str(value or "") for key, value in row.items() if key is not None}
            for row in reader
        ]
        return (columns, rows)
    except csv.Error:
        return None


def _is_dataset_diff_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".csv", ".json"}


def _dataset_table(path: str, text: str | None) -> tuple[list[str], list[dict[str, str]]] | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return _csv_table(text)
    if suffix == ".json":
        return _json_table(text)
    return None


def _json_table(text: str | None) -> tuple[list[str], list[dict[str, str]]] | None:
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        raw_rows = data
    elif isinstance(data, dict):
        list_value = next((value for value in data.values() if isinstance(value, list)), None)
        if list_value is not None:
            raw_rows = list_value
        else:
            raw_rows = [{"key": key, "value": value} for key, value in data.items()]
    else:
        raw_rows = [{"value": data}]

    rows: list[dict[str, str]] = []
    columns: list[str] = []
    for item in raw_rows:
        if isinstance(item, dict):
            row = {str(key): _json_cell(value) for key, value in item.items()}
        else:
            row = {"value": _json_cell(item)}
        for column in row:
            if column not in columns:
                columns.append(column)
        rows.append(row)
    return columns, rows


def _json_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _csv_rows(text: str | None) -> list[dict[str, str]] | None:
    table = _csv_table(text)
    return table[1] if table is not None else None


def _dataset_key(before_columns: list[str], after_columns: list[str]) -> str | None:
    common = [column for column in before_columns if column in set(after_columns)]
    for preferred in ("id", "ID", "uuid", "key", "name"):
        if preferred in common:
            return preferred
    return common[0] if common else None


def _index_rows(rows: list[dict[str, str]], key: str | None) -> dict[str, dict[str, str]]:
    indexed = {}
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        base_key = row.get(key, "") if key else ""
        base_key = base_key or str(index)
        seen[base_key] = seen.get(base_key, 0) + 1
        row_key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"
        indexed[row_key] = row
    return indexed


def _default_version_label(version: ArtifactVersion) -> str:
    if version.operation_type == "restore":
        return f"Restored version {version.version_number}"
    if version.publish_status == "published":
        return f"Published version {version.version_number}"
    return f"Version {version.version_number}"
