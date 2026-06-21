"""Tests for the AI edit pipeline (propose + accept with OCC).

Mirrors the fixture/setup style of tests/test_artifact_versions.py. The artifact
resolver (``_artifact_from_identifier_or_path`` → ``get_or_create_artifact_for_path``)
only treats a folder as a real artifact when it lives under a registered project's
``.anton/artifacts`` directory, so each test seeds a Project row and points the
artifact version store at a tmp directory.
"""
import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

import cowork.services.artifact_versions as artifact_versions
from cowork.models.artifact import Artifact
from cowork.models.project import Project
from cowork.services.artifact_edits import EditConflict, accept_edit, propose_edit


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ArtifactVersionService's default store root at a tmp directory."""
    store_root = tmp_path / "store"
    monkeypatch.setattr(
        artifact_versions.ArtifactVersionService,
        "_default_store_root",
        lambda self: store_root,
    )
    return store_root


def _seed_artifact(tmp_path: Path, session: Session) -> Path:
    project_dir = tmp_path / "project"
    folder = project_dir / ".anton" / "artifacts" / "demo"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps(
            {"slug": "demo", "name": "Demo Artifact", "description": "A small artifact", "type": "document"}
        ),
        encoding="utf-8",
    )
    (folder / "index.html").write_text("<h1>Hello world</h1>\n", encoding="utf-8")
    session.add(Project(id=uuid4(), name="demo-project", path=str(project_dir)))
    session.commit()
    return folder


def _current_version_id(session: Session, folder: Path) -> str | None:
    artifact = artifact_versions._artifact_from_identifier_or_path(session, artifact_id=None, path=str(folder))
    return str(artifact.current_version_id) if artifact.current_version_id else None


def test_propose_returns_diff_for_valid_old_text(tmp_path: Path, session: Session, store: Path):
    folder = _seed_artifact(tmp_path, session)

    result = propose_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="Hello world",
        new_text="Hello there",
        base_version_id=None,
    )

    assert result["applies"] is True
    assert result["target"] == "index.html"
    assert result["old"] == "Hello world"
    assert result["new"] == "Hello there"
    # Frontend proposeEdit() aliases.
    assert result["oldText"] == "Hello world"
    assert result["newText"] == "Hello there"
    # Dry-run: the live file is unchanged.
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello world</h1>\n"


def test_propose_reports_not_applies_for_missing_old_text(tmp_path: Path, session: Session, store: Path):
    folder = _seed_artifact(tmp_path, session)

    result = propose_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="this text is not present",
        new_text="whatever",
    )

    assert result["applies"] is False
    assert "error" in result


def test_accept_applies_and_advances_current_version(tmp_path: Path, session: Session, store: Path):
    folder = _seed_artifact(tmp_path, session)

    # The artifact starts with no versions; an initial accept (base None) lands.
    assert _current_version_id(session, folder) is None

    result = accept_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="Hello world",
        new_text="Hello there",
        base_version_id=None,
        actor_name="Tester",
    )

    assert result["ok"] is True
    new_version_id = result["versionId"]
    assert new_version_id

    # current_version_id advanced to the new version.
    assert _current_version_id(session, folder) == new_version_id
    # The live file was rewritten.
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello there</h1>\n"
    assert result["changedPaths"] == ["index.html"]
    assert result["version"]["operationType"] == "ai_edit"


def test_accept_with_stale_base_version_conflicts(tmp_path: Path, session: Session, store: Path):
    folder = _seed_artifact(tmp_path, session)

    # First accept establishes a real current version.
    first = accept_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="Hello world",
        new_text="Hello there",
        base_version_id=None,
    )
    current_after_first = first["versionId"]

    # A second accept using a STALE base (some older/other version id) must 409.
    stale_base = str(uuid4())
    assert stale_base != current_after_first

    with pytest.raises(EditConflict) as excinfo:
        accept_edit(
            session,
            path=str(folder),
            target="index.html",
            old_text="Hello there",
            new_text="Hello again",
            base_version_id=stale_base,
        )

    conflict = excinfo.value
    assert conflict.current_version_id == current_after_first
    assert conflict.base_version_id == stale_base
    # The conflicting edit did not mutate the live file or advance the version.
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello there</h1>\n"
    assert _current_version_id(session, folder) == current_after_first
    # 409 body carries the current version for the frontend's merge retry.
    assert conflict.current_version_dict() is not None
    assert conflict.current_version_dict()["versionId"] == current_after_first


def test_accept_with_matching_base_version_succeeds(tmp_path: Path, session: Session, store: Path):
    folder = _seed_artifact(tmp_path, session)

    first = accept_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="Hello world",
        new_text="Hello there",
        base_version_id=None,
    )

    # Re-read the now-current version and use it as the base — CAS should pass.
    second = accept_edit(
        session,
        path=str(folder),
        target="index.html",
        old_text="Hello there",
        new_text="Hello again",
        base_version_id=first["versionId"],
    )

    assert second["ok"] is True
    assert second["previousVersionId"] == first["versionId"]
    assert second["versionId"] != first["versionId"]
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Hello again</h1>\n"
