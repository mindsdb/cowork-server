import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import run_schema_migrations
from cowork.models.artifact import Artifact, ArtifactActivityEvent, ArtifactVersion, ArtifactVersionFile
from cowork.services.artifact_versions import ArtifactVersionService, version_to_dict


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _artifact_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "project" / ".anton" / "artifacts" / "demo"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "name": "Demo Artifact",
                "description": "A small artifact",
                "type": "html-app",
            }
        ),
        encoding="utf-8",
    )
    (folder / "README.md").write_text("housekeeping readme", encoding="utf-8")
    (folder / ".published.json").write_text("{}", encoding="utf-8")
    (folder / "index.html").write_text("<h1>Hello</h1>\n", encoding="utf-8")
    (folder / "assets").mkdir()
    (folder / "assets" / "app.js").write_text("console.log('hello')\n", encoding="utf-8")
    return folder


def test_deterministic_hashing_ignores_housekeeping(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")

    first = service.scan_manifest(folder)
    (folder / "metadata.json").write_text('{"name":"changed"}', encoding="utf-8")
    (folder / "README.md").write_text("changed", encoding="utf-8")
    (folder / ".published.json").write_text('{"index.html":{"url":"https://example.com"}}', encoding="utf-8")
    second = service.scan_manifest(folder)

    assert first.manifest_hash == second.manifest_hash
    assert first.files_hash == second.files_hash
    assert [entry.path for entry in first.files] == ["assets/app.js", "index.html"]


def test_snapshot_stores_files(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    store = tmp_path / "store"
    service = ArtifactVersionService(session, store)
    conversation_id = uuid4()
    message_id = uuid4()

    version = service.snapshot_artifact(
        folder,
        source_conversation_id=conversation_id,
        source_message_id=message_id,
        prompt="make a demo",
        operation_type="snapshot",
        preview_status="ready",
        publish_status="unpublished",
    )

    artifact = session.get(Artifact, version.artifact_id)
    files = session.exec(
        select(ArtifactVersionFile)
        .where(ArtifactVersionFile.version_id == version.id)
        .order_by(ArtifactVersionFile.path)
    ).all()

    assert artifact is not None
    assert artifact.slug == "demo"
    assert artifact.current_version_id == version.id
    assert version.version_number == 1
    assert version.created_at is not None
    assert version.prompt == "make a demo"
    assert version.source_conversation_id == conversation_id
    assert version.source_message_id == message_id
    assert [entry.path for entry in files] == ["assets/app.js", "index.html"]
    assert (store / version.store_path).is_file()
    for entry in files:
        assert (store / entry.blob_path).is_file()
    payload = version_to_dict(version, session=session)
    assert payload["prompt"] == "make a demo"
    assert payload["sourceConversationId"] == str(conversation_id)
    assert payload["sourceMessageId"] == str(message_id)
    assert payload["createdAt"]
    assert payload["filesHash"] == version.files_hash


def test_snapshot_rejects_file_changed_after_manifest_scan(
    tmp_path: Path,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")
    original_scan_manifest = service.scan_manifest

    def scan_then_mutate(path):
        manifest = original_scan_manifest(path)
        (folder / "index.html").write_text("<h1>Changed while copying</h1>\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(service, "scan_manifest", scan_then_mutate)

    with pytest.raises(IOError, match="Artifact file changed while snapshotting"):
        service.snapshot_artifact(folder)
    session.rollback()

    assert session.exec(select(ArtifactVersion)).all() == []


def test_snapshot_can_shadow_manifest_backend(tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COWORK_ARTIFACT_VERSION_SHADOW_BACKENDS", "manifest")
    folder = _artifact_dir(tmp_path)
    store = tmp_path / "store"
    service = ArtifactVersionService(session, store)

    version = service.snapshot_artifact(folder, operation_type="snapshot")
    event = session.exec(
        select(ArtifactActivityEvent).where(ArtifactActivityEvent.version_id == version.id)
    ).one()
    shadow = event.details["shadowBackends"][0]
    shadow_path = Path(shadow["path"])
    shadow_payload = json.loads(shadow_path.read_text(encoding="utf-8"))

    assert shadow["backend"] == "manifest"
    assert shadow["status"] == "ok"
    assert shadow_payload["versionId"] == str(version.id)
    assert shadow_payload["filesHash"] == version.files_hash


def test_unknown_shadow_backend_does_not_block_snapshot(
    tmp_path: Path,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COWORK_ARTIFACT_VERSION_SHADOW_BACKENDS", "not-a-backend")
    folder = _artifact_dir(tmp_path)

    version = ArtifactVersionService(session, tmp_path / "store").snapshot_artifact(folder)

    assert session.get(ArtifactVersion, version.id) is not None


def test_lix_shadow_backend_is_optional_and_fail_open(
    tmp_path: Path,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COWORK_ARTIFACT_VERSION_SHADOW_BACKENDS", "lix")
    folder = _artifact_dir(tmp_path)

    version = ArtifactVersionService(session, tmp_path / "store").snapshot_artifact(folder)
    event = session.exec(
        select(ArtifactActivityEvent).where(ArtifactActivityEvent.version_id == version.id)
    ).one()
    shadow = event.details["shadowBackends"][0]

    assert session.get(ArtifactVersion, version.id) is not None
    assert shadow["backend"] == "lix"
    assert shadow["status"] in {"unavailable", "failed", "ok"}


def test_materialize_version_restores_content(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")
    version = service.snapshot_artifact(folder)

    target = tmp_path / "materialized"
    target.mkdir()
    (target / "stale.txt").write_text("remove me", encoding="utf-8")
    service.materialize_version(version.id, target)

    assert not (target / "stale.txt").exists()
    assert (target / "index.html").read_text(encoding="utf-8") == "<h1>Hello</h1>\n"
    assert (target / "assets" / "app.js").read_text(encoding="utf-8") == "console.log('hello')\n"
    assert not (target / "metadata.json").exists()

    service.write_version_housekeeping(version.id, target)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "Demo Artifact"
    assert metadata["type"] == "html-app"
    assert (target / "README.md").read_text(encoding="utf-8") == "housekeeping readme"


def test_replace_with_version_can_preserve_published_sidecar(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")
    first = service.snapshot_artifact(folder)
    published = '{"index.html":{"url":"https://example.com/live","version_id":"old-good"}}'
    (folder / ".published.json").write_text(published, encoding="utf-8")
    (folder / "index.html").write_text("<h1>Broken draft</h1>\n", encoding="utf-8")
    service.snapshot_artifact(folder)

    service.replace_with_version(first.id, folder, preserve_published=True)

    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello</h1>\n"
    assert (folder / ".published.json").read_text(encoding="utf-8") == published


def test_publish_housekeeping_uses_version_metadata_not_live_metadata(
    tmp_path: Path,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints.publish import _copy_publish_housekeeping
    from cowork.api.v1.endpoints import publish as publish_endpoint

    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")
    version = service.snapshot_artifact(folder)
    monkeypatch.setattr(publish_endpoint, "ArtifactVersionService", lambda active_session: service)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "name": "Live Mutated",
                "description": "Mutated after checkpoint",
                "type": "document",
                "primary": "later.md",
            }
        ),
        encoding="utf-8",
    )
    (folder / "README.md").write_text("live readme", encoding="utf-8")

    target = tmp_path / "publish-source"
    service.materialize_version(version.id, target)
    _copy_publish_housekeeping(session, version, target)

    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "Demo Artifact"
    assert metadata["type"] == "html-app"
    assert (target / "README.md").read_text(encoding="utf-8") == "housekeeping readme"


def test_restore_creates_new_version_without_mutating_history(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    service = ArtifactVersionService(session, tmp_path / "store")
    first = service.snapshot_artifact(folder, prompt="first")
    (folder / ".published.json").write_text('{"index.html":{"url":"https://example.com/live"}}', encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "name": "Changed Metadata",
                "description": "Changed metadata",
                "type": "document",
                "primary": "index.html",
            }
        ),
        encoding="utf-8",
    )
    (folder / "index.html").write_text("<h1>Changed</h1>\n", encoding="utf-8")
    second = service.snapshot_artifact(folder, prompt="second")

    restored = service.restore_version(first.id, folder, prompt="restore first")
    versions = session.exec(
        select(ArtifactVersion)
        .where(ArtifactVersion.artifact_id == first.artifact_id)
        .order_by(ArtifactVersion.version_number)
    ).all()

    assert [version.version_number for version in versions] == [1, 2, 3]
    assert restored.id != first.id
    assert restored.id != second.id
    assert restored.restored_from_version_id == first.id
    assert restored.operation_type == "restore"
    assert restored.manifest_hash == first.manifest_hash
    assert session.get(ArtifactVersion, first.id).manifest_hash == first.manifest_hash
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello</h1>\n"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "Demo Artifact"
    assert metadata["type"] == "html-app"
    assert not (folder / ".published.json").exists()


def test_restore_missing_blob_leaves_live_folder_unchanged(tmp_path: Path, session: Session):
    folder = _artifact_dir(tmp_path)
    store = tmp_path / "store"
    service = ArtifactVersionService(session, store)
    first = service.snapshot_artifact(folder, prompt="first")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "name": "Live Metadata",
                "description": "Live version",
                "type": "document",
            }
        ),
        encoding="utf-8",
    )
    (folder / "index.html").write_text("<h1>Live</h1>\n", encoding="utf-8")
    service.snapshot_artifact(folder, prompt="second")

    entry = session.exec(
        select(ArtifactVersionFile)
        .where(ArtifactVersionFile.version_id == first.id)
        .where(ArtifactVersionFile.path == "index.html")
    ).one()
    (store / entry.blob_path).unlink()

    with pytest.raises(FileNotFoundError):
        service.restore_version(first.id, folder, prompt="restore first")
    session.rollback()

    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Live</h1>\n"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "Live Metadata"


def test_snapshot_works_against_migrated_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    try:
        db_path = tmp_path / "migrated.db"
        uri = f"sqlite:///{db_path}"
        engine = create_engine(uri)
        run_schema_migrations(engine, uri)

        folder = _artifact_dir(tmp_path)
        with Session(engine) as migrated_session:
            version = ArtifactVersionService(migrated_session, tmp_path / "store").snapshot_artifact(folder)

            assert version.version_number == 1
            assert version.file_count == 2
            assert session_file_count(migrated_session, version.id) == 2
    finally:
        get_app_settings.cache_clear()


def session_file_count(session: Session, version_id) -> int:
    return len(
        session.exec(
            select(ArtifactVersionFile).where(ArtifactVersionFile.version_id == version_id)
        ).all()
    )
