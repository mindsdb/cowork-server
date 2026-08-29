"""HTTP-originated filesystem selectors are normalized before service use."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import artifacts, comments, project_files
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.artifacts import ProjectArtifacts


def test_desktop_project_path_keeps_local_card_addressing(tmp_path, monkeypatch):
    project = tmp_path / "legacy-desktop-project"
    base = project / ".anton" / "artifacts"
    folder = base / "quarterly-report"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html>report</html>")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": uuid4().hex,
                "slug": folder.name,
                "name": "Quarterly report",
                "type": "html-app",
            }
        )
    )
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name=project.name,
    )
    monkeypatch.setattr(artifacts, "artifacts_sources_for_scan", lambda: [source])

    cards = artifacts._desktop_artifacts_for_project_path(
        SimpleNamespace(scope=LOCAL_SCOPE), str(project)
    )

    assert len(cards) == 1
    assert cards[0]["projectId"] is None
    assert "/drafts/local/" in cards[0]["draftUrl"]


@pytest.mark.parametrize(
    "value",
    [".", "..", "../other", "other/child", r"other\child", "bad\x00name", "x" * 256],
)
def test_legacy_artifact_delete_ref_rejects_non_child_names(value):
    with pytest.raises(HTTPException) as error:
        artifacts._artifact_delete_ref(value)

    assert error.value.status_code == 400


def test_artifact_delete_ref_canonicalizes_uuid():
    value = uuid4()

    ref = artifacts._artifact_delete_ref(str(value))

    assert ref.artifact_id == value.hex
    assert ref.legacy_slug is None


@pytest.mark.parametrize(
    "value",
    [
        "../secret",
        "nested/../../secret",
        "/absolute",
        "nested//file",
        r"nested\..\file",
        r"C:\Windows\win.ini",
        "C:secret.txt",
    ],
)
def test_project_mutation_path_rejects_traversal_components(value):
    with pytest.raises(HTTPException) as error:
        project_files._validated_project_path(value)

    assert error.value.status_code == 400


def test_project_mutation_path_preserves_nested_and_hidden_files():
    path = project_files._validated_project_path("conversations/abc/.anton/anton.md")

    assert path.parts == ("conversations", "abc", ".anton", "anton.md")
    assert path.value == "conversations/abc/.anton/anton.md"


def test_project_mutation_path_rebuilds_every_component_with_basename(monkeypatch):
    real_basename = project_files.os.path.basename
    seen = []

    def record(value):
        seen.append(value)
        return real_basename(value)

    monkeypatch.setattr(project_files.os.path, "basename", record)

    path = project_files._validated_project_path("nested/.anton/notes.md")

    assert path.parts == ("nested", ".anton", "notes.md")
    assert seen == ["nested", ".anton", "notes.md"]


def test_local_comments_id_is_canonical_before_identity_lookup(monkeypatch):
    value = uuid4()
    seen = {}
    monkeypatch.setattr(comments, "_org_mode", lambda: False)
    monkeypatch.setattr(comments, "artifacts_sources_for_scan", lambda: [])

    def resolve(_sources, artifact_id):
        seen["artifact_id"] = artifact_id
        raise FileNotFoundError

    monkeypatch.setattr(comments, "resolve_artifact_folder", resolve)

    assert comments.resolve_comments_route("artifact", str(value)) is None
    assert seen["artifact_id"] == value.hex


def test_invalid_local_comments_id_stops_at_request_boundary(monkeypatch):
    monkeypatch.setattr(comments, "_org_mode", lambda: False)

    with pytest.raises(HTTPException) as error:
        comments._local_report_id_for_request("artifact", "../another-artifact")

    assert error.value.status_code == 404


def test_historical_cloud_comments_key_remains_opaque(monkeypatch):
    monkeypatch.setattr(comments, "_org_mode", lambda: False)

    assert comments._local_report_id_for_request("legacy-user", "report-v1") is None
    assert comments.resolve_comments_route("legacy-user", "report-v1") == (
        "legacy-user",
        "report-v1",
    )


@pytest.mark.asyncio
async def test_status_path_only_selects_a_server_built_card(tmp_path, monkeypatch):
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "safe"
    primary = folder / "index.html"
    folder.mkdir(parents=True)
    primary.write_text("<html></html>", encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": uuid4().hex,
                "slug": folder.name,
                "primary": primary.name,
                "type": "html-app",
            }
        ),
        encoding="utf-8",
    )
    (folder / ".published.json").write_text(
        json.dumps(
            {
                primary.name: {
                    "published": True,
                    "report_id": "report-id",
                    "url": "https://example.test/artifact",
                    "published_mtime": primary.stat().st_mtime + 10,
                }
            }
        ),
        encoding="utf-8",
    )
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(artifacts, "artifacts_sources_for_scan", lambda: [source])

    result = await artifacts.artifact_status(str(primary))
    outside = await artifacts.artifact_status(str(tmp_path / "outside"))
    missing_child = await artifacts.artifact_status(str(folder / "does-not-exist"))

    assert result["publishedUrl"] == "https://example.test/artifact"
    assert result["modified"] is False
    assert outside["publishedUrl"] == ""
    assert missing_child["publishedUrl"] == ""


@pytest.mark.asyncio
async def test_status_lookup_preserves_published_loose_files(tmp_path, monkeypatch):
    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    base.mkdir(parents=True)
    loose = base / "legacy.html"
    loose.write_text("<html></html>", encoding="utf-8")
    (base / ".published.json").write_text(
        json.dumps(
            {
                loose.name: {
                    "published": True,
                    "report_id": "legacy-id",
                    "url": "https://example.test/legacy",
                    "mode": "password",
                    "access_password": "must-not-leak",
                }
            }
        ),
        encoding="utf-8",
    )
    source = ProjectArtifacts(
        base=base,
        project_id=None,
        project_name="project",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(artifacts, "artifacts_sources_for_scan", lambda: [source])

    result = await artifacts.artifact_status(str(loose))

    assert result["publishedUrl"] == "https://example.test/legacy"
    assert result["accessMode"] == "password"
    assert result["accessProtected"] is True
    assert "accessPassword" not in result
