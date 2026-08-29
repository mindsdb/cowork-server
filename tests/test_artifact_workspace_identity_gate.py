"""The workspace routes' path-level identity gate.

The rest of the workspace suite calls the handlers directly, which skips
FastAPI's own parameter validation — so a gate that only exists in the route
signature would look tested while never running. These go over HTTP.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from cowork.api.v1 import artifact_scope
from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture
def client(monkeypatch):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


@pytest.fixture
def no_resolution(monkeypatch):
    """Fail loudly if a request reaches artifact resolution at all.

    That is the point of the gate: a malformed identity must be refused before
    the service layer, the identity index or the filesystem sees the string.
    """
    def explode(*_args, **_kwargs):
        raise AssertionError("resolution reached with an unvalidated identity")

    monkeypatch.setattr("cowork.api.v1.artifact_scope._resolve_in", explode)


@pytest.mark.parametrize(
    "artifact_id",
    [
        "../../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "a1b2c3d4",                              # legacy 8-char id, no workspace
        "not-a-uuid",
        "0123456789abcdef0123456789abcdefff",    # hex, but too long for a UUID
        "",
    ],
)
def test_a_malformed_identity_never_reaches_resolution(client, no_resolution, artifact_id):
    res = client.get(f"/api/v1/artifacts/workspace/local/{artifact_id}")

    assert res.status_code in (404, 422), res.text


def test_both_uuid_spellings_address_the_same_artifact(client, monkeypatch):
    """The dashed and undashed forms are one identity, and the handler receives
    the canonical 32-hex spelling either way — that is what metadata carries."""
    seen: list[str] = []

    def capture(_sources, artifact_id):
        seen.append(artifact_id)
        raise AssertionError("stop here; the id is what this test is about")

    monkeypatch.setattr("cowork.api.v1.artifact_scope._resolve_in", capture)

    dashed = "0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b"
    undashed = dashed.replace("-", "")
    for spelling in (dashed, undashed):
        with pytest.raises(AssertionError, match="stop here"):
            client.get(f"/api/v1/artifacts/workspace/local/{spelling}")

    assert seen == [undashed, undashed]


def test_the_draft_preview_route_is_gated_too(client, no_resolution):
    res = client.get("/api/v1/artifacts/drafts/local/not-a-uuid/index.html")

    assert res.status_code == 422, res.text


def test_project_root_discovery_receives_the_scoped_catalog_id(monkeypatch):
    """The equal request spelling chooses a DB value; it is never reused as
    the UUID handed to artifact-root discovery."""
    server_id = UUID("0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b")
    request_ref = (str(server_id) + "x")[:-1]
    seen = []

    class Projects:
        def __init__(self, _session):
            pass

        def list_projects(self):
            return [SimpleNamespace(id=server_id)]

    monkeypatch.setattr("cowork.services.projects.ProjectService", Projects)
    monkeypatch.setattr(
        artifact_scope,
        "artifacts_sources_for_project",
        lambda _session, project_id, **_kwargs: seen.append(project_id) or [],
    )

    assert artifact_scope._sources_for_project_ref(object(), request_ref) == []
    assert seen == [server_id]
    assert seen[0] is server_id


# ── ?download=1 is read at the route boundary (ENG-2044) ─────────────────────
# test_artifact_draft_pinned_serving.py exercises the download behaviour by
# calling the handler with `download=True`. That proves the header builder, not
# that FastAPI parses the query string into the parameter — a `Query(False)`
# default that leaks the `Query` object into a direct call, or a renamed
# parameter, would leave those tests green while every real request ignored
# the flag. These go over HTTP for the same reason the gate tests above do.

_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture
def served_xlsx(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
    from cowork.services.artifacts import ProjectArtifacts

    project = tmp_path / "project"
    base = project / ".anton" / "artifacts"
    folder = base / "model"
    folder.mkdir(parents=True)
    (folder / "model.xlsx").write_bytes(b"PK\x03\x04sheet")
    source = ProjectArtifacts(
        base=base, project_id=None, project_name="project",
        trusted_anchor=project, root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(
        workspace_ep, "review_artifact_for_request",
        lambda *_args: (source, folder, {"type": "file"}, True),
    )
    return "/api/v1/artifacts/drafts/local/0123456789abcdef0123456789abcdef/model.xlsx"


@pytest.mark.parametrize("flag", ["1", "true"])
def test_download_flag_reaches_the_handler_over_http(client, served_xlsx, flag):
    res = client.get(f"{served_xlsx}?download={flag}")

    assert res.status_code == 200, res.text
    assert res.headers["content-disposition"] == (
        "attachment; filename=\"model.xlsx\"; filename*=UTF-8''model.xlsx"
    )
    assert res.headers["content-type"].startswith(_XLSX)
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.content == b"PK\x03\x04sheet"


@pytest.mark.parametrize("query", ["", "?download=0", "?download=false"])
def test_preview_is_unchanged_without_the_flag(client, served_xlsx, query):
    res = client.get(f"{served_xlsx}{query}")

    assert res.status_code == 200, res.text
    assert "content-disposition" not in res.headers
    assert res.content == b"PK\x03\x04sheet"
