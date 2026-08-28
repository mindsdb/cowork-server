"""Request selectors choose server catalog entries before artifact I/O."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifacts


def test_project_filtered_cards_discover_roots_with_the_server_catalog_id(
    monkeypatch,
):
    request_id = uuid4()
    # Same UUID value, deliberately a different object representing the value
    # recovered from the tenant-scoped database catalog.
    server_id = UUID(request_id.hex)
    trusted_source = SimpleNamespace(base="/server-owned/artifacts")
    seen = {}
    events = []

    class SanitizedProjectRef(str):
        pass

    real_basename = artifacts.os.path.basename

    def sanitize(project_ref):
        events.append("basename")
        sanitized = SanitizedProjectRef(real_basename(project_ref))
        seen["sanitized"] = sanitized
        return sanitized

    def recover(_session, project_ref):
        events.append("catalog")
        assert project_ref is seen["sanitized"]
        seen["lookup"] = project_ref
        return server_id

    def discover(_session, project_id):
        events.append("roots")
        seen["discovery"] = project_id
        return [trusted_source]

    def list_from_roots(sources):
        events.append("sink")
        seen["card_sources"] = sources
        return []

    monkeypatch.setattr(artifacts.os.path, "basename", sanitize)
    monkeypatch.setattr(artifacts, "scoped_project_id_for_request", recover)
    monkeypatch.setattr(artifacts, "artifacts_sources_for_project", discover)
    monkeypatch.setattr(artifacts, "_list_artifacts", list_from_roots)

    cards = artifacts.artifacts_for_request(
        SimpleNamespace(), project_id=str(request_id)
    )

    assert cards == []
    assert seen["lookup"] == str(request_id)
    assert seen["discovery"] is server_id
    assert seen["card_sources"] == [trusted_source]
    assert events == ["basename", "catalog", "roots", "sink"]


def test_project_path_leaf_sanitization_does_not_accept_a_prefixed_uuid(monkeypatch):
    def unexpected_catalog(*_args):
        raise AssertionError("path-like project value reached the catalog")

    monkeypatch.setattr(
        artifacts, "scoped_project_id_for_request", unexpected_catalog
    )

    with pytest.raises(HTTPException) as error:
        artifacts._scoped_project_sources(
            SimpleNamespace(), f"nested/{uuid4()}"
        )

    assert error.value.status_code == 400


async def test_legacy_delete_scans_only_roots_selected_by_the_server_catalog(
    monkeypatch,
):
    request_id = uuid4()
    server_id = UUID(request_id.hex)
    source = object()
    seen = {}

    monkeypatch.setattr(
        artifacts, "scoped_project_id_for_request", lambda *_args: server_id
    )

    def discover(_session, project_id):
        assert project_id is server_id
        return [source]

    class ScanReached(Exception):
        pass

    def scan(sources, folder_name):
        seen["sources"] = sources
        seen["folder_name"] = folder_name
        raise ScanReached

    monkeypatch.setattr(artifacts, "artifacts_sources_for_project", discover)
    monkeypatch.setattr(artifacts, "_legacy_artifact_for_sources", scan)

    with pytest.raises(ScanReached):
        await artifacts.delete_artifact_for_request(
            SimpleNamespace(), "legacy-report", project_id=str(request_id)
        )

    assert seen == {"sources": [source], "folder_name": "legacy-report"}


async def test_identity_delete_passes_catalog_project_and_canonical_artifact_ids(
    monkeypatch,
):
    request_project_id = uuid4()
    server_project_id = UUID(request_project_id.hex)
    artifact_id = uuid4()
    seen = {}

    monkeypatch.setattr(
        artifacts,
        "scoped_project_id_for_request",
        lambda *_args: server_project_id,
    )
    monkeypatch.setattr(
        artifacts, "artifacts_sources_for_project", lambda *_args: []
    )

    class ResolutionReached(Exception):
        pass

    def resolve(_session, project_ref, requested_artifact_id):
        seen["project_ref"] = project_ref
        seen["artifact_id"] = requested_artifact_id
        raise ResolutionReached

    monkeypatch.setattr(artifact_scope, "review_artifact_for_request", resolve)

    with pytest.raises(ResolutionReached):
        await artifacts.delete_artifact_for_request(
            SimpleNamespace(), str(artifact_id), project_id=str(request_project_id)
        )

    assert seen == {
        "project_ref": str(server_project_id),
        "artifact_id": artifact_id.hex,
    }
