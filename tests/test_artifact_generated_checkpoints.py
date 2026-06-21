from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from uuid import UUID
from uuid import uuid4

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact import Artifact, ArtifactVersion
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.harnesses.hermes_harness.harness import HermesHarness
from cowork.services.artifact_generation_checkpoints import GeneratedArtifactCheckpointTracker
from cowork.services.artifact_versions import ArtifactVersionService


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _make_artifact(root: Path, slug: str = "demo", content: str = "first") -> Path:
    folder = root / slug
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"slug": slug, "name": slug.title(), "type": "html-app"}),
        encoding="utf-8",
    )
    (folder / "index.html").write_text(content, encoding="utf-8")
    return folder


def _versions(session: Session):
    return session.exec(select(ArtifactVersion).order_by(ArtifactVersion.version_number)).all()


def test_generated_tracker_snapshots_before_and_after_changed_artifact(tmp_path: Path):
    with _session() as session:
        project = Project(name="Demo", path=str(tmp_path / "project"))
        session.add(project)
        session.commit()
        artifacts_root = Path(project.path) / ".anton" / "artifacts"
        folder = _make_artifact(artifacts_root)
        ArtifactVersionService(session, tmp_path / "store").snapshot_artifact(folder, project_id=project.id)

        tracker = GeneratedArtifactCheckpointTracker(session, artifacts_root, store_root=tmp_path / "store")
        before = tracker.snapshot_before()
        (folder / "index.html").write_text("second", encoding="utf-8")
        after = tracker.snapshot_after()

        versions = _versions(session)
        assert len(before) == 0
        assert len(after) == 1
        assert [version.operation_type for version in versions] == ["snapshot", "generated_update"]
        assert versions[-1].snapshot_role == "post"
        assert versions[-1].pre_snapshot_version_id == versions[0].id
        assert versions[-1].files_hash != versions[0].files_hash


def test_generated_tracker_records_unsnapshotted_existing_state_before_mutation(tmp_path: Path):
    with _session() as session:
        project = Project(name="Demo", path=str(tmp_path / "project"))
        session.add(project)
        session.commit()
        artifacts_root = Path(project.path) / ".anton" / "artifacts"
        folder = _make_artifact(artifacts_root)

        tracker = GeneratedArtifactCheckpointTracker(session, artifacts_root, store_root=tmp_path / "store")
        before = tracker.snapshot_before()
        (folder / "index.html").write_text("changed", encoding="utf-8")
        after = tracker.snapshot_after()

        versions = _versions(session)
        assert len(before) == 1
        assert len(after) == 1
        assert [version.operation_type for version in versions] == ["pre_generated_update", "generated_update"]
        assert [version.snapshot_role for version in versions] == ["pre", "post"]
        assert versions[1].pre_snapshot_version_id == versions[0].id


def test_generated_tracker_records_source_conversation_and_prompt(tmp_path: Path):
    with _session() as session:
        project = Project(name="Demo", path=str(tmp_path / "project"))
        conversation = Conversation(topic="Build slides", project=project)
        session.add(project)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        artifacts_root = Path(project.path) / ".anton" / "artifacts"
        folder = _make_artifact(artifacts_root)

        tracker = GeneratedArtifactCheckpointTracker(
            session,
            artifacts_root,
            source_conversation_id=UUID(str(conversation.id)),
            prompt="Make this artifact clearer.",
            store_root=tmp_path / "store",
        )
        tracker.snapshot_before()
        (folder / "index.html").write_text("changed", encoding="utf-8")
        tracker.snapshot_after()

        versions = _versions(session)
        assert [version.source_conversation_id for version in versions] == [conversation.id, conversation.id]
        assert [version.prompt for version in versions] == ["Make this artifact clearer.", "Make this artifact clearer."]


def test_generated_tracker_dedupes_no_change_windows(tmp_path: Path):
    with _session() as session:
        project = Project(name="Demo", path=str(tmp_path / "project"))
        session.add(project)
        session.commit()
        artifacts_root = Path(project.path) / ".anton" / "artifacts"
        folder = _make_artifact(artifacts_root)
        ArtifactVersionService(session, tmp_path / "store").snapshot_artifact(folder, project_id=project.id)

        tracker = GeneratedArtifactCheckpointTracker(session, artifacts_root, store_root=tmp_path / "store")
        assert tracker.snapshot_before() == []
        assert tracker.snapshot_after() == []

        versions = _versions(session)
        assert len(versions) == 1


def test_generated_tracker_captures_new_artifacts_after_generation(tmp_path: Path):
    with _session() as session:
        project = Project(name="Demo", path=str(tmp_path / "project"))
        session.add(project)
        session.commit()
        artifacts_root = Path(project.path) / ".anton" / "artifacts"
        tracker = GeneratedArtifactCheckpointTracker(session, artifacts_root, store_root=tmp_path / "store")

        assert tracker.snapshot_before() == []
        _make_artifact(artifacts_root, slug="new-demo", content="hello")
        after = tracker.snapshot_after()

        versions = _versions(session)
        assert len(after) == 1
        assert len(versions) == 1
        assert versions[0].operation_type == "generated_update"
        assert versions[0].snapshot_role == "single"
        assert versions[0].pre_snapshot_version_id is None


def test_hermes_snapshots_after_consumer_closes_early(tmp_path: Path, monkeypatch):
    engine = get_engine(get_app_settings().database.uri)
    project_path = tmp_path / "project"
    project_path.mkdir()
    project_name = f"hermes-cancel-{uuid4().hex}"

    with Session(engine) as session:
        project = Project(name=project_name, path=str(project_path), is_active=False)
        session.add(project)
        session.commit()
        session.refresh(project)
        conversation = Conversation(topic="Hermes cancellation", project_id=project.id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        _ = conversation.project
        _ = conversation.messages

        def fake_run(self, conversation_id, prompt, history, *, project_path, stream_callback, **kwargs):
            stream_callback("starting")
            time.sleep(0.1)
            artifacts_root = Path(project_path) / ".anton" / "artifacts"
            _make_artifact(artifacts_root, slug="hermes-output", content="created after close")
            return {"type": "message", "content": "done"}

        monkeypatch.setattr(HermesHarness, "_run", fake_run)

        async def run_and_close():
            stream = HermesHarness().stream_response(
                conversation=conversation,
                input=[{"type": "text", "text": "make an artifact"}],
            )
            first = await anext(stream)
            assert first["type"] == "delta"
            await stream.aclose()

        asyncio.run(run_and_close())

        versions = []
        for _ in range(20):
            session.expire_all()
            artifact = session.exec(
                select(Artifact)
                .where(Artifact.project_id == project.id)
                .where(Artifact.slug == "hermes-output")
            ).first()
            versions = []
            if artifact is not None:
                versions = session.exec(
                    select(ArtifactVersion)
                    .where(ArtifactVersion.artifact_id == artifact.id)
                    .order_by(ArtifactVersion.version_number)
                ).all()
            if any(version.operation_type == "generated_update" for version in versions):
                break
            asyncio.run(asyncio.sleep(0.05))

        assert [version.operation_type for version in versions] == ["generated_update"]
