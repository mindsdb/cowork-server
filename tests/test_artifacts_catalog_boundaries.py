"""Request selectors choose server catalog entries before artifact I/O."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifacts
from cowork.services import artifact_identity


async def test_project_route_basename_sanitizes_before_artifact_root_discovery(
    monkeypatch,
):
    request_id = uuid4()
    server_id = UUID(request_id.hex)
    trusted_source = SimpleNamespace(base="/server-owned/artifacts")
    seen = {}
    events = []

    class RequestProjectId:
        def __str__(self):
            events.append("selector")
            return str(request_id)

    real_basename = artifacts.os.path.basename

    def sanitize(project_ref):
        events.append("basename")
        return real_basename(project_ref)

    def discover(_session, project_ref):
        events.append("roots")
        seen["project_ref"] = project_ref
        return server_id, [trusted_source]

    def list_from_roots(sources):
        events.append("sink")
        seen["card_sources"] = sources
        return []

    session = SimpleNamespace()
    seen["session"] = session
    monkeypatch.setattr(artifacts.os.path, "basename", sanitize)
    monkeypatch.setattr(artifacts, "_scoped_project_sources", discover)
    monkeypatch.setattr(artifacts, "_list_artifacts", list_from_roots)

    cards = await artifacts.list_artifacts(
        session,
        project_id=RequestProjectId(),
        project_path=None,
    )

    assert cards == []
    assert seen["project_ref"] == str(request_id)
    assert seen["card_sources"] == [trusted_source]
    assert events == ["selector", "basename", "roots", "sink"]


async def test_desktop_route_reselects_one_source_by_sanitized_name(monkeypatch):
    other = SimpleNamespace(
        base=Path("/server-owned/other/.anton/artifacts"),
        project_id=None,
        project_name="other",
    )
    selected = SimpleNamespace(
        base=Path("/server-owned/project/.anton/artifacts"),
        project_id=None,
        project_name="project",
    )
    events = []

    def discover():
        events.append("roots")
        return [other, selected]

    def list_from_roots(sources):
        events.append(("sink", sources))
        return []

    monkeypatch.setattr(artifacts, "_org_mode", lambda: False)
    monkeypatch.setattr(artifacts, "artifacts_sources_for_scan", discover)
    monkeypatch.setattr(artifacts, "_list_artifacts", list_from_roots)

    cards = await artifacts.list_artifacts(
        SimpleNamespace(),
        project_id=None,
        project_path="/server-owned/project",
    )

    assert cards == []
    assert events == [
        "roots",
        "roots",
        ("sink", [selected]),
    ]


async def test_desktop_route_rejects_unregistered_path_with_same_basename(
    monkeypatch,
):
    source = SimpleNamespace(
        base=Path("/server-owned/project/.anton/artifacts"),
        project_id=None,
        project_name="project",
    )
    monkeypatch.setattr(artifacts, "_org_mode", lambda: False)
    monkeypatch.setattr(artifacts, "artifacts_sources_for_scan", lambda: [source])
    monkeypatch.setattr(
        artifacts,
        "_list_artifacts",
        lambda _sources: pytest.fail("unregistered path reached artifact listing"),
    )

    cards = await artifacts.list_artifacts(
        SimpleNamespace(),
        project_id=None,
        project_path="/attacker-controlled/project",
    )

    assert cards == []


def test_artifact_component_returns_the_basename_sanitizer_value(monkeypatch):
    class SanitizedName(str):
        pass

    clean = SanitizedName("artifacts")
    real_basename = artifact_identity.os.path.basename

    def basename(value):
        result = real_basename(value)
        return clean if result == clean else result

    monkeypatch.setattr(artifact_identity.os.path, "basename", basename)

    assert artifact_identity._component("artifacts") is clean
    with pytest.raises(ValueError, match="component is invalid"):
        artifact_identity._component("../artifacts")


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
