"""Artifact discovery and identity never cross writable directory links."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cowork.services import artifact_identity, artifact_roots, artifacts
from cowork.services.artifact_identity import ensure_full_id, resolve_artifact_folder
from cowork.services.artifacts import ProjectArtifacts, list_artifacts


FULL_ID = "11111111111141118111111111111111"


def _write_metadata(folder: Path, artifact_id: str = FULL_ID) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps({"id": artifact_id, "createdAt": "2026-08-28"}),
        encoding="utf-8",
    )


def _source(project: Path) -> ProjectArtifacts:
    return ProjectArtifacts(
        base=project / ".anton" / "artifacts",
        project_id="project-1",
        project_name=project.name,
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_desktop_scan_drops_a_symlinked_artifacts_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    linked = project / ".anton" / "artifacts"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(artifact_roots, "_scan_artifact_dirs", lambda: [linked])

    assert artifact_roots.artifacts_sources_for_scan() == []


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_org_discovery_drops_a_symlinked_conversation_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    conversation = project / "conversations" / "22222222-2222-4222-8222-222222222222"
    (conversation / ".anton").mkdir(parents=True)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (conversation / ".anton" / "artifacts").symlink_to(
        outside, target_is_directory=True
    )
    monkeypatch.setattr(artifact_roots, "_org_mode", lambda: True)

    assert artifact_roots._project_artifact_bases(
        str(project), object(), include_other_members=True
    ) == []


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_identity_resolution_refuses_an_anchored_root_symlink(tmp_path):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    outside = tmp_path / "outside-artifacts"
    _write_metadata(outside / "private")
    (project / ".anton" / "artifacts").symlink_to(
        outside, target_is_directory=True
    )
    artifact_identity._clear_identity_indexes()

    assert list_artifacts([_source(project)]) == []
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        resolve_artifact_folder([_source(project)], FULL_ID)


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_symlinked_artifact_folder_is_neither_resolved_nor_migrated(tmp_path):
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    base.mkdir(parents=True)
    outside = tmp_path / "outside-folder"
    legacy = {"id": "a1b2c3d4", "createdAt": "2026-08-28"}
    _write_metadata(outside, legacy["id"])
    before = (outside / "metadata.json").read_bytes()
    linked = base / "planted"
    linked.symlink_to(outside, target_is_directory=True)
    wanted = artifact_identity.resolve_artifact_id(
        legacy["id"], "", legacy["createdAt"]
    )
    artifact_identity._clear_identity_indexes()

    assert list_artifacts([_source(project)]) == []
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        resolve_artifact_folder([_source(project)], wanted)
    with pytest.raises(OSError):
        ensure_full_id(linked, legacy)

    assert (outside / "metadata.json").read_bytes() == before


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_symlinked_trusted_project_anchor_is_refused(tmp_path):
    real_project = tmp_path / "real-project"
    folder = real_project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    assert list_artifacts([_source(real_project)])[0]["slug"] == "draft"

    planted_anchor = tmp_path / "planted-project"
    planted_anchor.symlink_to(real_project, target_is_directory=True)
    planted = ProjectArtifacts(
        base=planted_anchor / ".anton" / "artifacts",
        project_id=None,
        project_name="planted-project",
        trusted_anchor=planted_anchor,
        root_parts=(".anton", "artifacts"),
    )

    assert list_artifacts([planted]) == []


@pytest.mark.skipif(os.name == "nt", reason="file-link threat is POSIX org storage")
def test_listing_does_not_read_a_symlinked_publish_record(tmp_path):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    outside = tmp_path / "outside-published.json"
    outside.write_text(
        json.dumps(
            {
                "index.html": {
                    "published": True,
                    "url": "https://outside.example/private",
                    "mode": "password",
                    "access_password": "secret",
                }
            }
        ),
        encoding="utf-8",
    )
    (folder / ".published.json").symlink_to(outside)

    card = list_artifacts([_source(project)])[0]

    assert card["publishedUrl"] == ""
    assert card["accessMode"] == "public"
    assert "outside.example" in outside.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_folder_swapped_after_index_build_is_refused(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    outside = tmp_path / "outside-folder"
    _write_metadata(outside)
    parked = tmp_path / "parked-folder"
    original = artifact_identity._identity_index
    swapped = False

    def swap_after_build(*args, **kwargs):
        nonlocal swapped
        result = original(*args, **kwargs)
        if not swapped:
            folder.rename(parked)
            folder.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    artifact_identity._clear_identity_indexes()
    monkeypatch.setattr(artifact_identity, "_identity_index", swap_after_build)

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        resolve_artifact_folder([_source(project)], FULL_ID)


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_root_swapped_after_index_build_is_refused(tmp_path, monkeypatch):
    project = tmp_path / "project"
    root = project / ".anton" / "artifacts"
    _write_metadata(root / "draft")
    outside = tmp_path / "outside-artifacts"
    _write_metadata(outside / "private")
    parked = tmp_path / "parked-artifacts"
    original = artifact_identity._identity_index
    swapped = False

    def swap_after_build(*args, **kwargs):
        nonlocal swapped
        result = original(*args, **kwargs)
        if not swapped:
            root.rename(parked)
            root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    artifact_identity._clear_identity_indexes()
    monkeypatch.setattr(artifact_identity, "_identity_index", swap_after_build)

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        resolve_artifact_folder([_source(project)], FULL_ID)


@pytest.mark.skipif(os.name == "nt", reason="directory-link threat is POSIX org storage")
def test_delete_refuses_a_folder_swap_before_removal(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    outside = tmp_path / "outside-folder"
    _write_metadata(outside)
    (outside / "keep.txt").write_text("private", encoding="utf-8")
    parked = tmp_path / "parked-draft"

    def swap_during_unpublish(*_args, **_kwargs):
        folder.rename(parked)
        folder.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(artifacts, "_unpublish_pinned_folder", swap_during_unpublish)

    with pytest.raises(FileNotFoundError, match="changed before deletion"):
        artifacts.delete_artifact_from_source(
            _source(project),
            "draft",
            expected_artifact_id=FULL_ID,
            api_key="key",
            publish_url="https://publish.example",
        )

    assert (outside / "keep.txt").read_text(encoding="utf-8") == "private"
    assert parked.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="file-link threat is POSIX org storage")
def test_delete_does_not_read_or_write_a_symlinked_publish_record(tmp_path):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    outside = tmp_path / "outside-published.json"
    outside_payload = json.dumps(
        {"index.html": {"report_id": "outside-id", "published": True}}
    )
    outside.write_text(outside_payload, encoding="utf-8")
    (folder / ".published.json").symlink_to(outside)

    artifacts.delete_artifact_from_source(
        _source(project),
        "draft",
        expected_artifact_id=FULL_ID,
        api_key="key",
        publish_url="https://publish.example",
    )

    assert not folder.exists()
    assert outside.read_text(encoding="utf-8") == outside_payload


def test_pinned_delete_unpublishes_before_removing_the_folder(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    (folder / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "report_id": "remote-id",
                    "published": True,
                    "url": "https://example.test/remote-id",
                }
            }
        ),
        encoding="utf-8",
    )
    revoked = []
    monkeypatch.setattr(
        artifacts,
        "_unpublish_identifier",
        lambda identifier, _entry, **_kwargs: revoked.append(identifier),
    )

    artifacts.delete_artifact_from_source(
        _source(project),
        "draft",
        expected_artifact_id=FULL_ID,
        api_key="key",
        publish_url="https://publish.example",
    )

    assert revoked == ["remote-id"]
    assert not folder.exists()


def test_pinned_delete_keeps_the_folder_when_unpublish_fails(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "draft"
    _write_metadata(folder)
    (folder / "index.html").write_text("<html></html>", encoding="utf-8")
    (folder / ".published.json").write_text(
        json.dumps(
            {"index.html": {"report_id": "remote-id", "published": True}}
        ),
        encoding="utf-8",
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(artifacts, "_unpublish_identifier", fail)

    with pytest.raises(RuntimeError, match="remote unavailable"):
        artifacts.delete_artifact_from_source(
            _source(project),
            "draft",
            expected_artifact_id=FULL_ID,
            api_key="key",
            publish_url="https://publish.example",
        )

    assert folder.is_dir()
