"""HTTP review selectors are validated before filesystem-backed resolution."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, get_scoped_session


PROJECT_ID = UUID("22222222-2222-4222-8222-222222222222")
ARTIFACT_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.fixture
def client(monkeypatch):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


@pytest.mark.parametrize("route", ["draft", "delete"])
def test_session_comes_from_dependency_and_cannot_be_supplied_in_query(
    client, monkeypatch, route,
):
    scoped_session = SimpleNamespace(scope=LOCAL_SCOPE)
    client.app.dependency_overrides[get_scoped_session] = lambda: scoped_session
    seen = []

    def capture(session, *_args, **_kwargs):
        seen.append(session)
        raise HTTPException(status_code=404, detail="No artifact in test catalog")

    async def capture_delete(session, *args, **kwargs):
        return capture(session, *args, **kwargs)

    params = {"session": "../../untrusted"}
    if route == "draft":
        monkeypatch.setattr(workspace_ep, "review_artifact_for_request", capture)
        method = "get"
        path = f"/api/v1/artifacts/drafts/local/{ARTIFACT_ID.hex}/index.html"
        schema_path = "/api/v1/artifacts/drafts/{project_ref}/{artifact_id}/{rel_path}"
    else:
        monkeypatch.setattr(artifacts_ep, "delete_artifact_for_request", capture_delete)
        method = "delete"
        path = f"/api/v1/artifacts/{ARTIFACT_ID.hex}"
        schema_path = "/api/v1/artifacts/{slug}"
        params["project_id"] = str(PROJECT_ID)

    response = client.request(method, path, params=params)

    assert response.status_code == 404, response.text
    assert len(seen) == 1
    assert seen[0] is scoped_session
    operation = client.app.openapi()["paths"][schema_path][method]
    assert all(parameter["name"] != "session" for parameter in operation["parameters"])


@pytest.mark.parametrize(
    "project_ref",
    ["..%5Coutside", "local%00", "C:%5Coutside", "%2e", "%2e%2e"],
)
def test_malformed_draft_project_is_rejected_before_review(
    client, monkeypatch, project_ref,
):
    def unexpected_review(*_args):
        pytest.fail("invalid project reached artifact review resolution")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", unexpected_review)

    response = client.get(
        f"/api/v1/artifacts/drafts/{project_ref}/{ARTIFACT_ID.hex}/index.html"
    )

    assert response.status_code == 400, response.text


@pytest.mark.parametrize("selector", ["project", "artifact"])
@pytest.mark.parametrize("prefix", ["../", "/", "..\\"])
async def test_path_prefixed_uuid_is_rejected_instead_of_selecting_its_basename(
    monkeypatch, selector, prefix,
):
    def unexpected_review(*_args):
        pytest.fail("a path-prefixed identity reached artifact review resolution")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", unexpected_review)
    project_ref = prefix + str(PROJECT_ID) if selector == "project" else "local"
    artifact_id = prefix + ARTIFACT_ID.hex if selector == "artifact" else ARTIFACT_ID.hex

    with pytest.raises(HTTPException) as error:
        await workspace_ep.serve_private_draft(
            project_ref, artifact_id, "index.html",
            SimpleNamespace(query_params={}), SimpleNamespace(),
        )

    assert error.value.status_code == 400


@pytest.mark.parametrize("project_ref", ["local", str(PROJECT_ID), PROJECT_ID.hex])
@pytest.mark.parametrize("artifact_id", [str(ARTIFACT_ID), ARTIFACT_ID.hex])
def test_valid_draft_selectors_reach_review_as_canonical_identities(
    client, monkeypatch, project_ref, artifact_id,
):
    seen = []

    def review(_session, project, artifact):
        seen.append((project, artifact))
        raise HTTPException(status_code=404, detail="No artifact in test catalog")

    monkeypatch.setattr(workspace_ep, "review_artifact_for_request", review)

    response = client.get(
        f"/api/v1/artifacts/drafts/{project_ref}/{artifact_id}/index.html"
    )

    assert response.status_code == 404, response.text
    expected_project = "local" if project_ref == "local" else str(PROJECT_ID)
    assert seen == [(expected_project, ARTIFACT_ID.hex)]


@pytest.mark.parametrize("artifact_id", [str(ARTIFACT_ID), ARTIFACT_ID.hex])
def test_delete_uuid_reaches_review_with_the_scoped_project(
    client, monkeypatch, artifact_id,
):
    seen = []
    monkeypatch.setattr(
        artifacts_ep, "_scoped_project_sources",
        lambda _session, _project_id: (PROJECT_ID, []),
    )

    def review(_session, project, artifact):
        seen.append((project, artifact))
        raise HTTPException(status_code=404, detail="No artifact in test catalog")

    monkeypatch.setattr(artifact_scope, "review_artifact_for_request", review)

    response = client.delete(
        f"/api/v1/artifacts/{artifact_id}", params={"project_id": str(PROJECT_ID)},
    )

    assert response.status_code == 404, response.text
    assert seen == [(str(PROJECT_ID), ARTIFACT_ID.hex)]


def test_delete_legacy_slug_still_uses_the_scoped_directory_catalog(client, monkeypatch):
    sources = [object()]
    seen = []
    monkeypatch.setattr(
        artifacts_ep, "_scoped_project_sources",
        lambda _session, _project_id: (PROJECT_ID, sources),
    )

    def resolve_legacy(authorized_sources, slug):
        seen.append((authorized_sources, slug))
        return None, None

    def unexpected_review(*_args):
        pytest.fail("legacy slug entered the UUID review resolver")

    monkeypatch.setattr(artifacts_ep, "_legacy_artifact_for_sources", resolve_legacy)
    monkeypatch.setattr(artifact_scope, "review_artifact_for_request", unexpected_review)

    response = client.delete(
        "/api/v1/artifacts/quarterly-report", params={"project_id": str(PROJECT_ID)},
    )

    assert response.status_code == 404, response.text
    assert seen == [(sources, "quarterly-report")]


@pytest.mark.parametrize("slug", ["..%5Coutside", "report%00", "%2e", "%2e%2e"])
def test_malformed_delete_slug_is_rejected_before_catalog_resolution(
    client, monkeypatch, slug,
):
    def unexpected_resolution(*_args):
        pytest.fail("invalid artifact selector reached the project catalog")

    monkeypatch.setattr(artifacts_ep, "_scoped_project_sources", unexpected_resolution)

    response = client.delete(
        f"/api/v1/artifacts/{slug}", params={"project_id": str(PROJECT_ID)},
    )

    assert response.status_code == 400, response.text
