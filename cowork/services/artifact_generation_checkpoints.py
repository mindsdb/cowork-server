from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.artifact import Artifact, ArtifactVersion
from cowork.models.project import Project
from cowork.services.artifact_versions import ArtifactVersionService


@dataclass
class GeneratedArtifactCheckpointTracker:
    session: Session
    artifacts_root: Path
    source_conversation_id: UUID | None = None
    prompt: str | None = None
    store_root: Path | None = None
    _before_hashes: dict[Path, str] = field(default_factory=dict)
    _before_versions: dict[Path, UUID] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifacts_root = Path(self.artifacts_root).expanduser().resolve(strict=False)

    def snapshot_before(self, *, label: str | None = None) -> list:
        versions = []
        service = ArtifactVersionService(self.session, self.store_root)
        for folder in self._artifact_folders(existing_only=True):
            manifest = self._scan(service, folder)
            if manifest is None or manifest.file_count == 0:
                continue
            self._before_hashes[folder] = manifest.files_hash
            current_version = self._current_version(folder)
            if current_version is not None and current_version.files_hash == manifest.files_hash:
                self._before_versions[folder] = current_version.id
                continue
            version = service.snapshot_artifact(
                folder,
                project_id=self._project_id_for_folder(folder),
                source_conversation_id=self.source_conversation_id,
                prompt=self.prompt,
                label=label or "Before generated update",
                operation_type="pre_generated_update",
                snapshot_role="pre",
            )
            versions.append(version)
            self._before_versions[folder] = version.id
        return versions

    def snapshot_after(self, *, label: str | None = None) -> list:
        versions = []
        service = ArtifactVersionService(self.session, self.store_root)
        for folder in self._artifact_folders(existing_only=False):
            manifest = self._scan(service, folder)
            if manifest is None:
                continue
            before_hash = self._before_hashes.get(folder)
            current_version = self._current_version(folder)
            current_hash = current_version.files_hash if current_version is not None else None
            pre_snapshot_version_id = self._before_versions.get(folder)
            if pre_snapshot_version_id is None and before_hash is not None and current_version is not None:
                pre_snapshot_version_id = current_version.id
            if manifest.file_count == 0 and before_hash is None and current_version is None:
                continue
            if current_hash == manifest.files_hash:
                continue
            if before_hash is not None and before_hash == manifest.files_hash:
                continue
            version = service.snapshot_artifact(
                folder,
                project_id=self._project_id_for_folder(folder),
                source_conversation_id=self.source_conversation_id,
                prompt=self.prompt,
                label=label or "Generated update",
                operation_type="generated_update",
                snapshot_role="post" if pre_snapshot_version_id is not None else "single",
                pre_snapshot_version_id=pre_snapshot_version_id,
            )
            versions.append(version)
            self._before_hashes[folder] = manifest.files_hash
            self._before_versions[folder] = version.id
        return versions

    def _artifact_folders(self, *, existing_only: bool) -> list[Path]:
        if not self.artifacts_root.is_dir():
            return []
        folders = []
        for child in sorted(self.artifacts_root.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if existing_only and not (child / "metadata.json").is_file():
                continue
            folders.append(child.resolve(strict=False))
        return folders

    def _scan(self, service: ArtifactVersionService, folder: Path):
        try:
            return service.scan_manifest(folder)
        except (FileNotFoundError, OSError):
            return None

    def _current_version(self, folder: Path) -> ArtifactVersion | None:
        artifact = self.session.exec(select(Artifact).where(Artifact.path == str(folder))).first()
        if artifact is None or artifact.current_version_id is None:
            return None
        return self.session.get(ArtifactVersion, artifact.current_version_id)

    def _project_id_for_folder(self, folder: Path) -> UUID | None:
        resolved = folder.resolve(strict=False)
        for project in self.session.exec(select(Project)).all():
            try:
                resolved.relative_to(Path(project.path).resolve(strict=False))
                return project.id
            except ValueError:
                continue
        return None
