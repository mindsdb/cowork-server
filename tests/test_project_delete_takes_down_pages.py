"""Deleting a project takes down its artifacts' published pages first (ENG-4455).

The pages, their access rules and their viewer analytics live upstream; the
project's files were the only thing pointing at them, so a delete that only
removed the files left every page live with nothing to find it by. The
take-down is the artifact delete's (`take_down_artifact_pages`), and a page
that fails to come down aborts the project delete, as it aborts an artifact
delete.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.services import artifacts as artifacts_service

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"

pytestmark = pytest.mark.usefixtures("granted_product_permissions", "cleanup_tmp_projects")


@pytest.fixture()
def org_client(tmp_path) -> Iterator[TestClient]:
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from cowork.server import create_app

        # No `with`: the lifespan's boot migrations would collide with the
        # schema conftest already built.
        client = TestClient(create_app())
        try:
            yield client
        finally:
            client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


@pytest.fixture()
def org() -> str:
    return str(uuid4())


@pytest.fixture()
def upstream(monkeypatch):
    """The take-down's credential and its calls to the publishing service."""
    state = type("Upstream", (), {})()
    state.credentials = 0
    state.taken_down = []
    state.fail_on = set()

    async def credential(scope):
        state.credentials += 1
        return "https://view.example", "turnkey-1"

    def unpublish(identifier, entry, *, api_key, publish_url):
        if identifier in state.fail_on:
            raise RuntimeError("Unpublishing failed: HTTP Error 500")
        state.taken_down.append((identifier, api_key, publish_url))

    monkeypatch.setattr(artifacts_ep, "take_down_credential", credential, raising=False)
    monkeypatch.setattr(artifacts_service, "_unpublish_identifier", unpublish)
    return state


def _headers(org: str, user: str) -> dict[str, str]:
    return {"X-User-Id": user, "X-Organization-Id": org}


def _create_project(client, org: str, user: str = ALICE) -> dict:
    res = client.post("/api/v1/projects/", headers=_headers(org, user), json={"name": f"p-{uuid4().hex[:8]}"})
    assert res.status_code == 201, res.text
    return res.json()


def _artifact(base: Path, slug: str, *, published: dict | None, primary: bool = True) -> Path:
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"slug": slug, "type": "html-app"}))
    if primary:
        (folder / "index.html").write_text("<html></html>")
    if published is not None:
        (folder / ".published.json").write_text(json.dumps({"index.html": published}))
    return folder


def _project_ids(client, org: str, user: str = ALICE) -> set[str]:
    res = client.get("/api/v1/projects/", headers=_headers(org, user))
    assert res.status_code == 200, res.text
    return {p["id"] for p in res.json()}


def test_org_project_delete_takes_down_every_live_page_then_removes_the_project(org_client, org, upstream):
    project = _create_project(org_client, org)
    base = Path(project["path"]) / ".anton" / "artifacts"
    _artifact(base, "live", published={"report_id": "rid-live", "url": "https://view.example/a/1"})
    # The primary file is gone: the page used to be logged and left up.
    _artifact(base, "orphan", published={"report_id": "rid-orphan", "published": True}, primary=False)
    _artifact(base, "down", published={"report_id": "rid-down", "published": False})
    _artifact(base, "draft", published=None)

    res = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, ALICE))

    assert res.status_code == 204, res.text
    assert sorted(upstream.taken_down) == [
        ("rid-live", "turnkey-1", "https://view.example"),
        ("rid-orphan", "turnkey-1", "https://view.example"),
    ]
    assert upstream.credentials == 1
    assert project["id"] not in _project_ids(org_client, org)
    assert not Path(project["path"]).exists()


def test_a_failed_take_down_aborts_the_project_delete(org_client, org, upstream):
    project = _create_project(org_client, org)
    base = Path(project["path"]) / ".anton" / "artifacts"
    first = _artifact(base, "a-first", published={"report_id": "rid-1"})
    second = _artifact(base, "b-second", published={"report_id": "rid-2"})
    upstream.fail_on = {"rid-2"}

    res = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, ALICE))

    assert res.status_code == 502
    assert "b-second" in res.json()["detail"]
    assert "the project was not deleted" in res.json()["detail"]
    assert project["id"] in _project_ids(org_client, org)
    assert first.exists() and second.exists()
    # The page already down is recorded as down, so a retry skips it.
    assert json.loads((first / ".published.json").read_text())["index.html"]["published"] is False
    assert json.loads((second / ".published.json").read_text())["index.html"].get("published", True) is True

    upstream.fail_on = set()
    retry = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, ALICE))

    assert retry.status_code == 204, retry.text
    assert [identifier for identifier, *_ in upstream.taken_down] == ["rid-1", "rid-2"]


def test_a_refused_project_delete_takes_no_page_down(org_client, org, upstream):
    project = _create_project(org_client, org, ALICE)
    _artifact(Path(project["path"]) / ".anton" / "artifacts", "live", published={"report_id": "rid-1"})

    res = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, BOB))

    assert res.status_code == 403, res.text
    assert upstream.taken_down == []
    assert upstream.credentials == 0
    assert project["id"] in _project_ids(org_client, org)


def test_a_project_with_no_live_page_mints_no_credential(org_client, org, upstream):
    project = _create_project(org_client, org)
    _artifact(Path(project["path"]) / ".anton" / "artifacts", "draft", published=None)

    res = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, ALICE))

    assert res.status_code == 204, res.text
    assert upstream.credentials == 0
    assert upstream.taken_down == []


def test_pages_in_conversation_workspaces_come_down_too(org_client, org, upstream):
    project = _create_project(org_client, org)
    workspace = Path(project["path"]) / "conversations" / str(uuid4()) / ".anton" / "artifacts"
    _artifact(workspace, "legacy", published={"report_id": "rid-legacy"})

    res = org_client.delete(f"/api/v1/projects/{project['id']}", headers=_headers(org, ALICE))

    assert res.status_code == 204, res.text
    assert [identifier for identifier, *_ in upstream.taken_down] == ["rid-legacy"]


# -- desktop ------------------------------------------------------------------


def test_desktop_project_delete_takes_down_pages_and_aborts_on_failure(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    from cowork.server import create_app

    client = TestClient(create_app())
    try:
        res = client.post("/api/v1/projects/", json={"name": f"published-{uuid4().hex[:8]}"})
        assert res.status_code == 201, res.text
        project = res.json()
        folder = _artifact(
            Path(project["path"]) / ".anton" / "artifacts", "live", published={"report_id": "rid-desk"},
        )
        upstream.fail_on = {"rid-desk"}

        failed = client.delete(f"/api/v1/projects/{project['id']}")

        assert failed.status_code == 502
        assert folder.exists()

        upstream.fail_on = set()
        done = client.delete(f"/api/v1/projects/{project['id']}")

        assert done.status_code == 204, done.text
        assert [identifier for identifier, *_ in upstream.taken_down] == ["rid-desk"]
        assert not folder.exists()
    finally:
        client.close()
        get_app_settings.cache_clear()


def test_desktop_delete_of_a_chosen_folder_leaves_its_pages_up(tmp_path, monkeypatch, upstream):
    """A folder the user chose stays when its project is deleted, artifacts
    and all, so their published pages stay up with them: nothing is taken
    down and no credential is minted."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    from cowork.server import create_app

    chosen = tmp_path / "Documents" / "reports"
    chosen.mkdir(parents=True)
    # Choosing a folder is a local-only request.
    client = TestClient(create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))
    try:
        res = client.post("/api/v1/projects/", json={"name": f"chosen-{uuid4().hex[:8]}", "path": str(chosen)})
        assert res.status_code == 201, res.text
        project = res.json()
        folder = _artifact(Path(project["path"]) / ".anton" / "artifacts", "live", published={"report_id": "rid-kept"})

        done = client.delete(f"/api/v1/projects/{project['id']}")

        assert done.status_code == 204, done.text
        assert upstream.taken_down == []
        assert folder.exists()
        assert json.loads((folder / ".published.json").read_text())["index.html"]["report_id"] == "rid-kept"
    finally:
        client.close()
        get_app_settings.cache_clear()
