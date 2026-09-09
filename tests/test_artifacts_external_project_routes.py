"""Every artifact action, over HTTP, for a project pointed at a chosen folder.

Listing and serving were made to work for such a project; the rest of the
artifact surface resolved by scanning the projects root and refused it, mostly
as a 404 "Artifact is not in a known artifacts directory" and in three places as
a silent empty result.

These go over the route on purpose. The resolver-level equivalents passed while
the HTTP path was still broken, because the handlers held no session to resolve
a project row with.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cowork.services import publish as publish_service

ARTIFACT_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccc01"


@pytest.fixture()
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(root))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield root
    get_app_settings.cache_clear()


def _client():
    """`base_url` sets `scope["server"]`, `client` the peer; both gates read them."""
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    return TestClient(
        create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )


def _artifact_in(folder: Path, slug: str = "dash", *, artifact_id: str = ARTIFACT_ID):
    """An html-app artifact on disk, shaped as the harness writes one."""
    artifact = folder / ".anton" / "artifacts" / slug
    artifact.mkdir(parents=True)
    (artifact / "index.html").write_text("<html><body>hi</body></html>")
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "id": artifact_id,
                "slug": slug,
                "name": "Adopted dashboard",
                "primary": "index.html",
                "type": "html-app",
            }
        )
    )
    return artifact


def _adopt(client, folder: Path, name: str):
    """Point a project at a folder the user already has, over the wire.

    Names must be unique across the whole suite: the HTTP tests share one
    process-wide database, and adopting refuses a name already taken.
    """
    created = client.post(
        "/api/v1/projects/", json={"name": name, "path": str(folder)}
    )
    assert created.status_code == 201, created.text
    assert created.json()["capabilities"]["directoryIsExternal"] is True
    return created.json()


def _adopted(
    client, tmp_path: Path, name: str, *, slug: str = "dash", artifact_id: str = ARTIFACT_ID
):
    folder = tmp_path / "Documents" / name
    folder.mkdir(parents=True)
    artifact = _artifact_in(folder, slug, artifact_id=artifact_id)
    project = _adopt(client, folder, name)
    return project, folder, artifact


# -- the 404 group -----------------------------------------------------------


def test_preview_reads_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-preview")

    res = client.get(
        "/api/v1/artifacts/preview", params={"path": str(artifact / "index.html")}
    )

    assert res.status_code == 200, res.text
    assert "hi" in res.json()["content"]


def test_preview_mount_serves_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    client = _client()
    project, _folder, artifact = _adopted(client, tmp_path, "routes-mount")

    res = client.post(
        "/api/v1/artifacts/preview-mount", json={"path": str(artifact / "index.html")}
    )

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["kind"] == "static"
    # The serve URL cannot be rediscovered by scanning, so it has to be built
    # from the source's own base and row name.
    assert payload["serveUrl"] == (
        f"/api/v1/artifacts/serve/{project['name']}/dash/index.html"
    )


def test_export_writes_beside_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-export")

    res = client.post(
        "/api/v1/artifacts/export",
        json={"path": str(artifact / "index.html"), "format": "html"},
    )

    assert res.status_code == 200, res.text
    assert Path(res.json()["path"]).exists()


def test_open_accepts_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    """`open` shells out, so the subprocess is stubbed; what is under test is
    that resolution no longer refuses the path before that point."""
    from cowork.api.v1.endpoints import artifacts as artifacts_ep

    opened = []
    monkeypatch.setattr(
        artifacts_ep.subprocess, "run", lambda cmd, **kw: opened.append(cmd)
    )

    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-open")
    primary = artifact / "index.html"

    res = client.post("/api/v1/artifacts/open", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert opened == [["open", str(primary.resolve())]]


def test_reveal_accepts_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    from cowork.services import artifacts as artifacts_service

    revealed = []
    monkeypatch.setattr(
        artifacts_service, "reveal_in_file_manager", lambda p: revealed.append(Path(p))
    )
    from cowork.api.v1.endpoints import artifacts as artifacts_ep

    monkeypatch.setattr(
        artifacts_ep, "reveal_in_file_manager", lambda p: revealed.append(Path(p))
    )

    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-reveal")
    primary = artifact / "index.html"

    res = client.post("/api/v1/artifacts/reveal", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert revealed == [primary.resolve()]


# -- the silent group --------------------------------------------------------


def test_status_keeps_the_published_pill_for_a_chosen_folder(projects_root, tmp_path):
    """This one never 404s, which is why it reads as a UI bug rather than an
    error: it returns the blank default, so the published pill and the modified
    flag vanish every time the window regains focus."""
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-status")
    (artifact / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "report_id": "uuid-status",
                    "url": "https://4nton.ai/a/uuid-status",
                    "last_md5": "old",
                    "published": True,
                    "mode": "public",
                    "published_mtime": 1,
                }
            }
        )
    )

    res = client.get(
        "/api/v1/artifacts/status", params={"path": str(artifact / "index.html")}
    )

    assert res.status_code == 200, res.text
    assert res.json()["publishedUrl"] == "https://4nton.ai/a/uuid-status"


# -- publish, unpublish, delete ----------------------------------------------


def _published_record(artifact: Path, report_id: str = "uuid-pub"):
    (artifact / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "report_id": report_id,
                    "url": f"https://4nton.ai/a/{report_id}",
                    "last_md5": "old",
                    "published": True,
                    "mode": "public",
                    "published_mtime": 1,
                }
            }
        )
    )


def test_publish_resolves_an_artifact_in_a_chosen_folder(
    projects_root, tmp_path, monkeypatch
):
    """The upload is stubbed. What is under test is that the artifact and its
    artifacts base resolve at all, which is where publish refused before."""
    from cowork.api.v1.endpoints import publish as publish_ep

    seen = {}

    def _fake_publish(artifact, *, artifacts_base, api_key, publish_url, **kwargs):
        seen["artifact"] = Path(artifact)
        seen["base"] = Path(artifacts_base)
        return {"url": "https://4nton.ai/a/uuid-pub", "md5": "abc"}

    monkeypatch.setattr(publish_ep, "_publish", _fake_publish)
    monkeypatch.setattr(
        publish_service, "desktop_publish_credential", lambda: ("key", "https://4nton.ai")
    )

    client = _client()
    _project, folder, artifact = _adopted(client, tmp_path, "routes-publish")
    primary = artifact / "index.html"

    res = client.post("/api/v1/publish/", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert seen["artifact"] == primary.resolve()
    assert seen["base"] == (folder / ".anton" / "artifacts").resolve()


def test_unpublish_resolves_an_artifact_in_a_chosen_folder(
    projects_root, tmp_path, monkeypatch
):
    from cowork.api.v1.endpoints import publish as publish_ep

    seen = {}

    def _fake_unpublish(artifact, *, artifacts_base, api_key, publish_url):
        seen["artifact"] = Path(artifact)
        return {"status": "ok"}

    monkeypatch.setattr(publish_ep, "_unpublish", _fake_unpublish)
    monkeypatch.setattr(
        publish_service, "desktop_publish_credential", lambda: ("key", "https://4nton.ai")
    )

    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-unpublish")
    primary = artifact / "index.html"
    _published_record(artifact)

    res = client.request(
        "DELETE", "/api/v1/publish/", params={"path": str(primary)}
    )

    assert res.status_code == 200, res.text
    assert seen["artifact"] == primary.resolve()


def test_delete_by_path_removes_an_artifact_in_a_chosen_folder(
    projects_root, tmp_path, monkeypatch
):
    from cowork.api.v1.endpoints import artifacts as artifacts_ep

    monkeypatch.setattr(
        artifacts_ep, "_delete_artifact", lambda artifact, **kw: shutil.rmtree(artifact)
    )
    monkeypatch.setattr(
        publish_service, "desktop_publish_credential", lambda: ("key", "https://4nton.ai")
    )

    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-delete")

    res = client.request("DELETE", "/api/v1/artifacts/", params={"path": str(artifact)})

    assert res.status_code == 204, res.text
    assert not artifact.exists()


def test_the_publishable_list_includes_a_chosen_folder(projects_root, tmp_path):
    """The publish picker built its list from the scan, so a project pointed at
    a chosen folder simply had nothing to offer."""
    client = _client()
    _adopted(client, tmp_path, "routes-publishable")

    res = client.get("/api/v1/publish/")

    assert res.status_code == 200, res.text
    listed = [Path(a["path"]) for a in res.json()["artifacts"]]
    assert any(p.name == "index.html" and "routes-publishable" in str(p) for p in listed)


def test_the_project_path_list_is_not_empty_for_a_chosen_folder(projects_root, tmp_path):
    """Older desktop builds address the list by path. It returned 200 with an
    empty body, so the project read as having produced nothing at all."""
    client = _client()
    project, folder, _artifact = _adopted(client, tmp_path, "routes-bypath")

    res = client.get("/api/v1/artifacts/", params={"project_path": str(folder)})

    assert res.status_code == 200, res.text
    cards = res.json()
    assert [c["title"] for c in cards] == ["Adopted dashboard"]
    assert cards[0]["serveUrl"] == (
        f"/api/v1/artifacts/serve/{project['name']}/dash/index.html"
    )


def test_the_project_path_list_carries_the_row_name_not_the_basename(
    projects_root, tmp_path
):
    """Adopting a folder whose basename is already taken gives a row name that
    differs from it, and the serve URL has to follow the row rather than the
    directory the artifacts happen to sit in."""
    client = _client()
    taken = client.post("/api/v1/projects/", json={"name": "routes-clash"})
    assert taken.status_code == 201, taken.text

    folder = tmp_path / "elsewhere" / "routes-clash"
    folder.mkdir(parents=True)
    _artifact_in(folder)
    created = client.post(
        "/api/v1/projects/", json={"name": "routes-clash-2", "path": str(folder)}
    )
    assert created.status_code == 201, created.text
    project_name = created.json()["name"]
    assert project_name != folder.name

    res = client.get("/api/v1/artifacts/", params={"project_path": str(folder)})

    assert res.status_code == 200, res.text
    cards = res.json()
    assert len(cards) == 1
    assert cards[0]["serveUrl"] == (
        f"/api/v1/artifacts/serve/{project_name}/dash/index.html"
    )


def test_search_finds_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    """Search held a scoped session and resolved artifact roots without it."""
    client = _client()
    _adopted(client, tmp_path, "routes-search")

    res = client.get("/api/v1/search", params={"q": "Adopted dashboard"})

    assert res.status_code == 200, res.text
    results = res.json()["results"]
    titles = [r.get("title") for r in results if r.get("type") == "artifact"]
    assert "Adopted dashboard" in titles


def test_comments_resolve_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    """The local journal resolves the artifact folder to decide where threads
    live, and refused a chosen folder with a 404 that read as a missing artifact."""
    client = _client()
    comment_artifact_id = "cccccccc-cccc-4ccc-8ccc-cccccccccc02"
    _project, _folder, artifact = _adopted(
        client, tmp_path, "routes-comments", slug="notes", artifact_id=comment_artifact_id
    )

    # "artifact" is the canonical desktop namespace; anything else proxies out.
    res = client.get(
        f"/api/v1/artifact-comments/artifact/{comment_artifact_id}/threads"
    )

    assert res.status_code == 200, res.text
    assert (artifact / ".revisions").is_dir()
