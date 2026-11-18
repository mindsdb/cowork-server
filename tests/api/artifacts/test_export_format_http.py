"""HTTP artifact format exposure and agreement with the export boundary."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services import artifact_permissions
from cowork.services.artifact_export import ExportSource
from cowork.services.artifact_metadata import ArtifactMetadata
from cowork.services.artifact_permissions import artifact_capabilities
from _async_stub import awaitable
from tests.support.artifacts.format_cases import _LOCAL_ARTIFACTS, _local_source, _write_artifact
from tests.support.artifacts.format_fixtures import local_mode as local_mode

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def test_the_local_workspace_response_carries_export_formats(tmp_path, local_mode, monkeypatch):
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        awaitable(lambda *_args: (source, folder, ArtifactMetadata.parse((folder / "metadata.json").read_text()), True)),
    )
    from cowork.server import create_app

    response = TestClient(create_app()).get(f"/api/v1/artifacts/workspace/local/{artifact_id}")

    assert response.status_code == 200, response.text
    assert response.json()["capabilities"]["exportFormats"] == ["html", "docx"]


def test_the_workspace_response_decides_export_formats_where_artifact_capabilities_does(
    tmp_path, local_mode, monkeypatch
):
    """The workspace response already holds its owner's capabilities, and adds
    exportFormats through the helper artifact_capabilities() adds it with. A
    change to how the list is decided then reaches both answers alike."""
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        awaitable(lambda *_args: (source, folder, ArtifactMetadata.parse((folder / "metadata.json").read_text()), True)),
    )
    monkeypatch.setattr(artifact_permissions, "offered_export_formats", lambda **_kwargs: ["docx"])
    from cowork.server import create_app

    response = TestClient(create_app()).get(f"/api/v1/artifacts/workspace/local/{artifact_id}")
    capabilities = artifact_capabilities(
        SimpleNamespace(scope=LOCAL_SCOPE),
        source,
        "report",
        export_source=ExportSource(source_name="report.md", is_backend=False),
    )

    assert response.status_code == 200, response.text
    assert capabilities["exportFormats"] == ["docx"]
    assert response.json()["capabilities"]["exportFormats"] == ["docx"]


def test_a_hard_linked_primary_offers_nothing_anywhere_and_the_route_answers_422(
    tmp_path, local_mode, monkeypatch
):
    """A sync or backup tool can give the primary a second hard link. The
    export's read refuses such a file, so its card and the workspace response
    offer no format, and the route answers 422, as for any format not offered."""
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(
        source.base, "report", {"report.md": "# R\n"}, type="document", primary="report.md"
    )
    os.link(folder / "report.md", tmp_path / "backup.md")
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        cards = [card.to_json() for card in artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])]
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        awaitable(lambda *_args: (source, folder, ArtifactMetadata.parse((folder / "metadata.json").read_text()), True)),
    )
    from cowork.server import create_app

    client = TestClient(create_app())
    response = client.get(f"/api/v1/artifacts/workspace/local/{artifact_id}")
    files_before = sorted(path.name for path in folder.iterdir())

    assert [card["capabilities"]["exportFormats"] for card in cards] == [[]]
    assert response.status_code == 200, response.text
    assert response.json()["capabilities"]["exportFormats"] == []
    for export_format in ("html", "docx"):
        refused = client.post(
            f"/api/v1/artifacts/workspace/local/{artifact_id}/export", json={"format": export_format}
        )
        assert refused.status_code == 422, refused.text
    assert sorted(path.name for path in folder.iterdir()) == files_before


@pytest.mark.parametrize("slug", sorted(_LOCAL_ARTIFACTS))
def test_the_route_accepts_exactly_the_formats_the_card_lists(tmp_path, local_mode, monkeypatch, slug):
    files, metadata, _expected = _LOCAL_ARTIFACTS[slug]
    source = _local_source(tmp_path)
    artifact_id, folder = _write_artifact(source.base, slug, files, **metadata)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        (card,) = [card.to_json() for card in artifacts_ep._artifact_cards(ScopedSession(raw, LOCAL_SCOPE), [source])]
    listed = card["capabilities"]["exportFormats"]
    monkeypatch.setattr(
        workspace_ep,
        "review_artifact_for_request",
        awaitable(lambda *_args: (source, folder, ArtifactMetadata.parse((folder / "metadata.json").read_text()), True)),
    )
    from cowork.server import create_app

    client = TestClient(create_app())
    for export_format in ("html", "docx", "pdf"):
        response = client.post(
            f"/api/v1/artifacts/workspace/local/{artifact_id}/export",
            json={"format": export_format},
        )
        expected = 200 if export_format in listed else 422
        assert response.status_code == expected, (export_format, response.text)
