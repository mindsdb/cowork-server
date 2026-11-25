"""Every artifact action, over HTTP, for a project pointed at a chosen folder.

Listing and serving find such a project through its row; the path-addressed
actions resolved by scanning the projects root only and refused it, mostly as a
404 "Artifact is not in a known artifacts directory" and elsewhere as a silent
blank answer.

These go over the route on purpose: a resolver-level test passed while the
route was still broken, because the handler never handed the resolver the
project's row.
"""
from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import anton.core.artifacts.backend_launcher as backend_launcher
import anton.publisher as anton_publisher
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.api.v1.endpoints import publish as publish_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.project import Project
from cowork.server import create_app
from cowork.services import artifact_revisions
from cowork.services import artifacts as artifacts_service
from cowork.services import published_page_ownership
from cowork.services import publish as publish_service
from cowork.services import scratchpad_runtime
from cowork.services.artifact_roots import desktop_adopted_rows

ARTIFACT_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccc01"


@pytest.fixture()
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(root))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield root
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def _remove_adopted_rows():
    """Adopting writes into the process-wide database from tests/conftest.py.

    Left behind, those rows are visible to every later test that resolves
    artifact roots for real.
    """
    uri = get_app_settings().database.uri
    with get_open_session(uri) as s:
        before = {row.id for row in s.exec(select(Project)).all()}
    yield
    with get_open_session(uri) as s:
        for row in s.exec(select(Project)).all():
            if row.id not in before:
                s.delete(row)
        s.commit()


def _client() -> TestClient:
    """`base_url` sets `scope["server"]`, `client` the peer; both gates read them."""
    return TestClient(
        create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )


def _artifact_in(
    folder: Path, slug: str = "dash", *, artifact_id: str = ARTIFACT_ID, extra: dict | None = None
) -> Path:
    """An artifact on disk, shaped as the harness writes one."""
    artifact = folder / ".anton" / "artifacts" / slug
    artifact.mkdir(parents=True)
    (artifact / "index.html").write_text("<html><body>hi</body></html>")
    metadata = {
        "id": artifact_id,
        "slug": slug,
        "name": "Adopted dashboard",
        "primary": "index.html",
        "type": "html-app",
        **(extra or {}),
    }
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    return artifact


def _adopt(client: TestClient, folder: Path, name: str) -> dict:
    """Point a project at a folder the user already has, over the wire.

    Adopting refuses a name already taken in the shared test database, so
    names are unique per test; `_remove_adopted_rows` drops each row after.
    """
    created = client.post("/api/v1/projects/", json={"name": name, "path": str(folder)})
    assert created.status_code == 201, created.text
    assert created.json()["capabilities"]["directoryIsExternal"] is True
    return created.json()


def _adopted(
    client: TestClient, tmp_path: Path, name: str, *, slug: str = "dash", extra: dict | None = None
) -> tuple[dict, Path, Path]:
    folder = tmp_path / "Documents" / name
    folder.mkdir(parents=True)
    artifact = _artifact_in(folder, slug, extra=extra)
    project = _adopt(client, folder, name)
    return project, folder, artifact


def _published_record(artifact: Path, report_id: str = "uuid-pub", mode: str = "public") -> None:
    (artifact / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "report_id": report_id,
                    "url": f"https://4nton.ai/a/{report_id}",
                    "last_md5": "old",
                    "published": True,
                    "mode": mode,
                    "published_mtime": 1,
                }
            }
        )
    )


# -- viewing, opening, revealing, status -------------------------------------


def test_preview_reads_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-preview")

    res = client.get("/api/v1/artifacts/preview", params={"path": str(artifact / "index.html")})

    assert res.status_code == 200, res.text
    assert "hi" in res.json()["content"]


def test_preview_mount_serves_an_artifact_in_a_chosen_folder_under_its_row_name(
    projects_root, tmp_path
):
    client = _client()
    project, _folder, artifact = _adopted(client, tmp_path, "routes-mount")

    res = client.post("/api/v1/artifacts/preview-mount", json={"path": str(artifact / "index.html")})

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["kind"] == "static"
    # The scan cannot rediscover this URL, and the folder's basename is not the
    # segment: it is built from the adopted source's base and row name.
    assert payload["serveUrl"] == f"/api/v1/artifacts/serve/{project['name']}/dash/index.html"


def test_open_accepts_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    """`open` shells out, so the subprocess is stubbed; what is under test is
    that resolution no longer refuses the path before that point."""
    opened = []
    monkeypatch.setattr(artifacts_ep.subprocess, "run", lambda cmd, **kw: opened.append(cmd))
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-open")
    primary = artifact / "index.html"

    res = client.post("/api/v1/artifacts/open", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert opened == [["open", str(primary.resolve())]]


def test_reveal_accepts_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    revealed = []
    # Patched in the endpoint module, which binds the name at import.
    monkeypatch.setattr(artifacts_ep, "reveal_in_file_manager", lambda p: revealed.append(Path(p)))
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-reveal")
    primary = artifact / "index.html"

    res = client.post("/api/v1/artifacts/reveal", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert revealed == [primary.resolve()]


def test_status_keeps_the_published_pill_for_a_chosen_folder(projects_root, tmp_path):
    """A blank status is a 200 too, so this asserts the published fields: the
    pill and the modified flag vanished every time the window regained focus."""
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-status")
    _published_record(artifact, "uuid-status")

    res = client.get("/api/v1/artifacts/status", params={"path": str(artifact / "index.html")})

    assert res.status_code == 200, res.text
    assert res.json()["publishedUrl"] == "https://4nton.ai/a/uuid-status"
    assert res.json()["modified"] is True


def test_a_fullstack_backend_launches_from_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    """Auto-launch answers 200 with a launchError rather than an error status,
    so the failure showed as a preview that never came up. The spawn is
    stubbed; under test is that the chosen folder reaches it as the project
    root."""
    client = _client()
    folder = tmp_path / "Documents" / "routes-fullstack"
    folder.mkdir(parents=True)
    artifact = _artifact_in(folder, "app", extra={"type": "fullstack", "port": 51999})
    _adopt(client, folder, "routes-fullstack")

    monkeypatch.setattr(artifacts_service, "_probe_port", lambda port, **kw: False)
    monkeypatch.setattr(artifacts_service, "_LAUNCHED_BACKENDS", {})
    pool_roots: list[str] = []
    launched: list[Path] = []

    class _Pool:
        def __init__(self, root: str, **_kwargs) -> None:
            pool_roots.append(root)

    async def _launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, **_rest):
        launched.append(artifact_folder)
        return {"port": 51999, "pid": 1}

    monkeypatch.setattr(scratchpad_runtime, "WorkspaceScopedPool", _Pool)
    monkeypatch.setattr(backend_launcher, "launch_artifact_backend", _launch)

    res = client.post("/api/v1/artifacts/preview-mount", json={"path": str(artifact / "index.html")})

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["kind"] == "proxy"
    assert payload["launchError"] == ""
    assert payload["backendRunning"] is True
    assert [Path(root).resolve() for root in pool_roots] == [folder.resolve()]
    assert [path.resolve() for path in launched] == [artifact.resolve()]


def test_preview_assets_of_a_chosen_folder_serve_where_identity_headers_are_trusted(
    projects_root, tmp_path, monkeypatch
):
    """Where identity headers are trusted, each preview asset is checked
    against the caller's artifacts roots. A folder adopted before that was
    turned on keeps its row, so its mount still has to count as the caller's.
    Checked at the service: a trusted-headers request carries a person's
    scope, where no folder can be adopted to set this up over the route."""
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-assets")
    res = client.post("/api/v1/artifacts/preview-mount", json={"path": str(artifact / "index.html")})
    assert res.status_code == 200, res.text
    token = res.json()["token"]
    with get_open_session(get_app_settings().database.uri) as s:
        rows = desktop_adopted_rows(ScopedSession(s, LOCAL_SCOPE))

    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "true")
    get_app_settings.cache_clear()
    assert get_app_settings().trust_identity_headers

    assert artifacts_service.preview_mount_for(token, scope=LOCAL_SCOPE) is None
    assert artifacts_service.preview_mount_for(token, scope=LOCAL_SCOPE, adopted=rows) == artifact.resolve()


# -- export, which moved to an id-addressed route -----------------------------


def test_export_writes_beside_an_artifact_in_a_chosen_folder(projects_root, tmp_path):
    """Coverage, not a regression test: the id-addressed export resolves
    through the project row and already worked for a chosen folder."""
    client = _client()
    project, _folder, artifact = _adopted(
        client, tmp_path, "routes-export", extra={"primary": "report.md", "type": "document"}
    )
    (artifact / "report.md").write_text("# Quarterly\n\nNumbers.\n")

    res = client.post(
        f"/api/v1/artifacts/workspace/{project['id']}/{ARTIFACT_ID}/export", json={"format": "html"}
    )

    assert res.status_code == 200, res.text
    assert (artifact / res.json()["filename"]).is_file()


# -- publish, update, versions, activate, unpublish, delete --------------------


def _with_credential(monkeypatch) -> None:
    """A configured publish credential: without one these routes answer 400
    for every folder, chosen or allocated, before resolving the path."""
    monkeypatch.setattr(publish_service, "desktop_publish_credential", lambda: ("key", "https://4nton.ai"))
    monkeypatch.setattr(
        publish_service, "_resolve_publish_endpoint", lambda settings: ("https://4nton.ai", "key")
    )


def test_publish_resolves_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    """The upload is stubbed; under test is that the artifact and its
    artifacts base resolve, which is where publish refused before."""
    seen = {}

    def _fake_publish(artifact, *, artifacts_base, api_key, publish_url, **kwargs):
        seen["artifact"] = Path(artifact)
        seen["base"] = Path(artifacts_base)
        return {"url": "https://4nton.ai/a/uuid-pub", "md5": "abc"}

    monkeypatch.setattr(publish_ep, "_publish", _fake_publish)
    _with_credential(monkeypatch)
    client = _client()
    _project, folder, artifact = _adopted(client, tmp_path, "routes-publish")
    primary = artifact / "index.html"

    res = client.post("/api/v1/publish/", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert res.json()["url"] == "https://4nton.ai/a/uuid-pub"
    assert seen == {"artifact": primary.resolve(), "base": (folder / ".anton" / "artifacts").resolve()}


def test_update_republishes_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    seen = {}

    def _fake_publish(artifact, *, artifacts_base, api_key, publish_url, access, **kwargs):
        seen["artifact"] = Path(artifact)
        return {"url": "https://4nton.ai/a/uuid-pub", "md5": "new"}

    monkeypatch.setattr(publish_service, "publish_artifact", _fake_publish)
    _with_credential(monkeypatch)
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-update")
    _published_record(artifact)
    primary = artifact / "index.html"

    res = client.post("/api/v1/publish/update", json={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert seen == {"artifact": primary.resolve()}


def test_versions_lists_the_history_of_an_artifact_in_a_chosen_folder(
    projects_root, tmp_path, monkeypatch
):
    asked = []

    def _fake_list_versions(report_id, **kwargs):
        asked.append(report_id)
        return {"report_id": report_id, "current_md5": "v2", "versions": [{"md5": "v1"}, {"md5": "v2"}]}

    monkeypatch.setattr(anton_publisher, "list_versions", _fake_list_versions)
    _with_credential(monkeypatch)
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-versions")
    _published_record(artifact, "uuid-versions")

    res = client.get("/api/v1/publish/versions", params={"path": str(artifact / "index.html")})

    assert res.status_code == 200, res.text
    assert asked == ["uuid-versions"]
    assert [v["md5"] for v in res.json()["versions"]] == ["v2", "v1"]


def test_activate_rolls_back_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    def _fake_activate(report_id, md5, **kwargs):
        return {"current_md5": md5, "view_url": f"https://4nton.ai/a/{report_id}"}

    monkeypatch.setattr(anton_publisher, "activate_version", _fake_activate)
    monkeypatch.setattr(published_page_ownership, "publication_owner", lambda *a, **kw: nullcontext())
    _with_credential(monkeypatch)
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-activate")
    _published_record(artifact, "uuid-activate")

    res = client.post(
        "/api/v1/publish/activate", json={"path": str(artifact / "index.html"), "md5": "v1"}
    )

    assert res.status_code == 200, res.text
    assert res.json()["url"] == "https://4nton.ai/a/uuid-activate"
    record = json.loads((artifact / ".published.json").read_text())
    assert record["index.html"]["last_md5"] == "v1"


def test_unpublish_resolves_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    seen = {}

    def _fake_unpublish(artifact, *, artifacts_base, api_key, publish_url, **kwargs):
        seen["artifact"] = Path(artifact)
        return {"status": "ok"}

    monkeypatch.setattr(publish_ep, "_unpublish", _fake_unpublish)
    _with_credential(monkeypatch)
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-unpublish")
    _published_record(artifact)
    primary = artifact / "index.html"

    res = client.request("DELETE", "/api/v1/publish/", params={"path": str(primary)})

    assert res.status_code == 200, res.text
    assert seen == {"artifact": primary.resolve()}


def test_delete_by_path_removes_an_artifact_in_a_chosen_folder(projects_root, tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts_ep, "_delete_artifact", lambda artifact, **kw: shutil.rmtree(artifact))
    _with_credential(monkeypatch)
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-delete")

    res = client.request("DELETE", "/api/v1/artifacts/", params={"path": str(artifact)})

    assert res.status_code == 204, res.text
    assert not artifact.exists()


def test_a_refresh_keeps_revisions_whose_comments_live_in_the_cloud_in_a_chosen_folder(
    projects_root, tmp_path
):
    """A scheduled refresh may replace an artifact's head revision only while
    its comments live in the local journal. Shared with restricted access, a
    chosen folder's comments live in the cloud, and the check that says so
    used to re-resolve the folder by scan and read it as never published."""
    client = _client()
    _project, _folder, artifact = _adopted(client, tmp_path, "routes-refresh")
    (artifact / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "report_id": "uuid-restricted",
                    "url": "https://4nton.ai/a/uuid-restricted",
                    "published": True,
                    "mode": "restricted",
                    "lambda_artifact_key": f"artifact/{ARTIFACT_ID}",
                }
            }
        )
    )

    assert artifact_revisions._comments_live_in_the_cloud(artifact, LOCAL_SCOPE) is True
