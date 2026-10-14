"""Artifacts on a deployment that trusts identity headers: each person's own.

Two people, A and B, sign in to one stack. Every artifact route takes the
request's scope and resolves paths only inside that person's own projects, so B
reaches none of A's artifacts by path, by project name, by preview token or by
artifact id: list, read, status, export, preview, delete by path, serve,
search, comments and publish all answer 404, blank or absent for B, and work
for A. A deployment that keeps projects shared does what today's stack does,
and the desktop and hosted paths are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlmodel import Session

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.services.projects import ProjectService

TOKEN = "partition-artifacts-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
BEARER = {"Authorization": f"Bearer {TOKEN}"}
A = {**BEARER, "X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT}
B = {**BEARER, "X-User-Id": SUB_B, "X-Organization-Id": DEPLOYMENT}


def _person(*, partition: str, sub: str) -> TenantScope:
    return TenantScope(
        personal=True,
        org_id=partition,
        user_id=sub,
        personal_resources=ALL_PERSONAL_RESOURCES,
    )


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    """A trusted-identity stack whose stores all live under the test's folder."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    monkeypatch.delenv("ANTON_COWORK_STATE_DIR", raising=False)
    get_app_settings.cache_clear()
    yield monkeypatch
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        raw.commit()


def _general(scope: TenantScope) -> Path:
    """The person's own `general` project directory, provisioned as a request would."""
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        project = ProjectService(ScopedSession(raw, scope)).ensure_general_for_scope()
        assert project is not None
        return Path(project.path)


def _artifact(project_dir: Path, slug: str, *, title: str, published_url: str = "") -> Path:
    folder = project_dir / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    primary = folder / "index.html"
    primary.write_text(f"<html><body>{title}</body></html>", encoding="utf-8")
    (folder / "report.md").write_text(f"# {title}\n", encoding="utf-8")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": uuid4().hex,
                "slug": slug,
                "name": title,
                "type": "html-app",
                "primary": "index.html",
            }
        ),
        encoding="utf-8",
    )
    if published_url:
        (folder / ".published.json").write_text(
            json.dumps(
                {
                    "index.html": {
                        "report_id": "report-1",
                        "url": published_url,
                        "published": True,
                        "published_mtime": primary.stat().st_mtime + 10,
                    }
                }
            ),
            encoding="utf-8",
        )
    return folder


def _artifact_id(folder: Path) -> str:
    return json.loads((folder / "metadata.json").read_text())["id"]


def _client() -> TestClient:
    # No `with`: skipping the lifespan skips boot migrations and workers.
    return TestClient(cowork_server.create_app())


@pytest.fixture
def two_people(stack):
    """A owns `general/dash` (published) and `general/notes`; B owns an empty
    `general` of their own, so a same-name lookup in B's partition finds B's."""
    a_general = _general(ALICE)
    _general(BOB)
    dash = _artifact(
        a_general, "dash", title="Quarterly forecast", published_url="https://view.example/r/1"
    )
    notes = _artifact(a_general, "notes", title="Planning notes")
    return dash, notes


# ── reading another person's artifact ────────────────────────────────────────


def test_list_shows_only_the_callers_own_artifacts(two_people):
    client = _client()

    mine = client.get("/api/v1/artifacts/", headers=A)
    theirs = client.get("/api/v1/artifacts/", headers=B)

    assert mine.status_code == 200 and theirs.status_code == 200
    assert {card["slug"] for card in mine.json()} == {"dash", "notes"}
    assert theirs.json() == []


def test_preview_and_status_by_path_reach_only_the_owner(two_people):
    dash, _notes = two_people
    primary = str(dash / "index.html")
    client = _client()

    assert client.get("/api/v1/artifacts/preview", params={"path": primary}, headers=B).status_code == 404
    preview = client.get("/api/v1/artifacts/preview", params={"path": primary}, headers=A)
    assert preview.status_code == 200
    assert "Quarterly forecast" in preview.json()["content"]

    theirs = client.get("/api/v1/artifacts/status", params={"path": primary}, headers=B)
    mine = client.get("/api/v1/artifacts/status", params={"path": primary}, headers=A)
    assert theirs.status_code == 200 and theirs.json()["publishedUrl"] == ""
    assert mine.json()["publishedUrl"] == "https://view.example/r/1"


def test_export_by_path_reaches_only_the_owner(two_people):
    dash, _notes = two_people
    body = {"path": str(dash / "report.md"), "format": "html"}
    client = _client()

    assert client.post("/api/v1/artifacts/export", json=body, headers=B).status_code == 404
    assert not (dash / "report.html").exists()
    mine = client.post("/api/v1/artifacts/export", json=body, headers=A)
    assert mine.status_code == 200
    assert (dash / "report.html").is_file()


def test_preview_mount_and_its_assets_reach_only_the_owner(two_people):
    dash, _notes = two_people
    body = {"path": str(dash / "index.html")}
    client = _client()

    assert client.post("/api/v1/artifacts/preview-mount", json=body, headers=B).status_code == 404
    mounted = client.post("/api/v1/artifacts/preview-mount", json=body, headers=A)
    assert mounted.status_code == 200
    token = mounted.json()["token"]

    asset = f"/api/v1/artifacts/preview-asset/{token}/index.html"
    # The token is a hash of A's path; holding it gives B nothing.
    assert client.get(asset, headers=B).status_code == 404
    assert client.get(asset, headers=A).status_code == 200


class _Upstream:
    status_code = 200
    headers = {"content-type": "application/json"}

    async def aiter_raw(self):
        yield b"{}"

    async def aclose(self):
        pass


def test_a_backend_preview_token_forwards_only_for_the_owner(two_people, monkeypatch):
    """The proxy token is a hash of A's artifact root, as stable as the asset
    one, and A's backend runs with A's connection secrets: holding the token
    must not forward B's request to it."""
    from cowork.services import artifacts, preview_proxy

    dash, _notes = two_people
    meta = json.loads((dash / "metadata.json").read_text())
    (dash / "metadata.json").write_text(json.dumps({**meta, "port": 4242}))
    forwarded: list[str] = []

    class _Client:
        def build_request(self, method, url, headers, content):
            forwarded.append(url)
            return (method, url)

        async def send(self, upstream_req, stream):
            return _Upstream()

    async def running(_root, port, *, scope):
        return True, "already_running", port

    monkeypatch.setattr(artifacts, "_ensure_backend_running", running)
    monkeypatch.setattr(preview_proxy, "get_proxy_client", _Client)
    client = _client()

    mounted = client.post("/api/v1/artifacts/preview-mount", json={"path": str(dash / "index.html")}, headers=A)
    assert mounted.status_code == 200 and mounted.json()["kind"] == "proxy"
    url = f"/api/v1/artifacts/proxy/{mounted.json()['token']}/data"

    assert client.get(url, headers=B).status_code == 404
    assert forwarded == []
    assert client.get(url, headers=A).status_code == 200
    assert forwarded == ["http://127.0.0.1:4242/data"]


def test_serve_by_project_name_reads_the_callers_own_project(two_people):
    url = "/api/v1/artifacts/serve/general/dash/index.html"
    client = _client()

    # B has a `general` too, and it holds no `dash`.
    assert client.get(url, headers=B).status_code == 404
    mine = client.get(url, headers=A)
    assert mine.status_code == 200
    assert b"Quarterly forecast" in mine.content


def test_delete_by_path_reaches_only_the_owner(two_people, monkeypatch):
    _dash, notes = two_people
    monkeypatch.setattr(
        "cowork.services.publish.desktop_publish_credential",
        lambda: ("publish-key", "https://publish.example"),
    )
    client = _client()

    refused = client.delete("/api/v1/artifacts/", params={"path": str(notes)}, headers=B)
    assert refused.status_code == 404
    assert notes.is_dir()

    assert client.delete("/api/v1/artifacts/", params={"path": str(notes)}, headers=A).status_code == 204
    assert not notes.exists()


def test_open_and_reveal_are_refused_in_a_personal_scope(two_people):
    dash, _notes = two_people
    body = {"path": str(dash / "index.html")}
    client = _client()

    # Refused before the path resolves, for the owner as for anyone else.
    for headers in (A, B):
        assert client.post("/api/v1/artifacts/open", json=body, headers=headers).status_code == 403
        assert client.post("/api/v1/artifacts/reveal", json=body, headers=headers).status_code == 403


def test_search_lists_only_the_callers_own_artifacts(two_people):
    client = _client()

    def artifact_titles(headers):
        response = client.get("/api/v1/search", params={"q": "Quarterly"}, headers=headers)
        assert response.status_code == 200
        return [r["title"] for r in response.json()["results"] if r["type"] == "artifact"]

    assert artifact_titles(A) == ["Quarterly forecast"]
    assert artifact_titles(B) == []


def test_comments_resolve_only_the_callers_own_artifact(two_people):
    dash, _notes = two_people
    url = f"/api/v1/artifact-comments/artifact/{_artifact_id(dash)}/threads"
    client = _client()

    assert client.get(url, headers=B).status_code == 404
    assert client.post(url, json={"text": "mine now"}, headers=B).status_code == 404
    assert client.get(url, headers=A).status_code == 200
    assert not (dash / ".revisions" / "comments.json").exists()


# ── publish ──────────────────────────────────────────────────────────────────


def test_publish_routes_resolve_only_the_callers_own_artifact(two_people, monkeypatch):
    dash, _notes = two_people
    primary = str(dash / "index.html")
    monkeypatch.setattr(
        "cowork.services.publish._resolve_publish_endpoint",
        lambda _settings: ("https://publish.example", "publish-key"),
    )
    monkeypatch.setattr(
        "anton.publisher.list_versions",
        lambda report_id, **_kw: {"report_id": report_id, "current_md5": "m1", "versions": []},
    )
    client = _client()

    for method, url, kwargs in (
        ("get", "/api/v1/publish/versions", {"params": {"path": primary}}),
        ("delete", "/api/v1/publish/", {"params": {"path": primary}}),
        ("post", "/api/v1/publish/update", {"json": {"path": primary}}),
        ("post", "/api/v1/publish/activate", {"json": {"path": primary, "md5": "m1"}}),
        ("post", "/api/v1/publish/", {"json": {"path": primary}}),
    ):
        response = getattr(client, method)(url, headers=B, **kwargs)
        assert response.status_code == 404, (method, url, response.text)

    mine = client.get("/api/v1/publish/versions", params={"path": primary}, headers=A)
    assert mine.status_code == 200
    assert mine.json()["reportId"] == "report-1"


def test_publish_history_and_publishable_list_are_the_callers_own(two_people):
    from cowork.services import publish

    dash, _notes = two_people
    publish._save_state(
        {"publish_history": [{"artifact": str(dash / "index.html"), "url": "https://view.example/r/1"}]},
        scope=ALICE,
    )
    client = _client()

    mine = client.get("/api/v1/publish/", headers=A).json()
    theirs = client.get("/api/v1/publish/", headers=B).json()

    assert [item["url"] for item in mine["history"]] == ["https://view.example/r/1"]
    assert str(dash / "index.html") in {item["path"] for item in mine["artifacts"]}
    assert theirs["history"] == []
    assert theirs["artifacts"] == []


def test_the_agent_publish_tool_resolves_the_askers_own_artifact(two_people):
    """The agent's tools resolve paths in the turn's bound scope."""
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.harnesses.anton_harness import tools

    dash, _notes = two_people
    primary = str(dash / "index.html")

    with use_settings_scope(ALICE):
        assert tools._published_state(primary)["url"] == "https://view.example/r/1"
    with use_settings_scope(BOB):
        assert tools._published_state(primary) == {"report_id": "", "url": "", "published": False}


def test_the_operator_view_lists_no_persons_artifacts(two_people):
    """A call with no person (audit mode) is the operator view, LOCAL_SCOPE. It
    reads every project row unfiltered, and a person's project sits outside the
    unkeyed root, but a row with an org_id is never an adopted folder, so the
    operator view's list does not reach into anyone's partition. (Rows other
    modules leave behind can list their own artifacts here.)"""
    dash, notes = two_people
    client = _client()

    listed = client.get("/api/v1/artifacts/", headers=BEARER)

    assert listed.status_code == 200
    folders = {Path(card["folder"]).resolve() for card in listed.json()}
    assert dash.resolve() not in folders and notes.resolve() not in folders
    assert not any(PARTITION_A in str(folder) for folder in folders)


# ── live backends ───────────────────────────────────────────────────────────


async def test_two_peoples_same_slug_backends_keep_running(stack, monkeypatch):
    """anton's launcher keys the dict it is handed by slug and reaps the entry
    it finds there, so one dict for the whole server would stop A's backend the
    moment B previews an artifact with the same slug."""
    from cowork.services import artifacts

    a_app = _artifact(_general(ALICE), "app", title="A app")
    b_app = _artifact(_general(BOB), "app", title="B app")
    launched: list[int] = []
    reaped: list[int] = []

    async def fake_launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, ds_env, health_timeout):
        # What anton's launcher does with the dict: reap, then track by slug.
        previous = tracked_backends.pop(slug, None)
        if previous is not None:
            reaped.append(previous["pid"])
        pid = 1000 + len(launched)
        launched.append(pid)
        tracked_backends[slug] = {"proc": None, "port": pid, "pid": pid, "log_path": ""}
        return {"port": pid, "pid": pid}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    monkeypatch.setattr(artifacts, "_probe_port", lambda _port, **_kw: False)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    monkeypatch.setattr(artifacts, "_BACKEND_LAUNCH_LOCKS", {})

    a_running = await artifacts._ensure_backend_running(a_app, 1, scope=ALICE)
    b_running = await artifacts._ensure_backend_running(b_app, 1, scope=BOB)

    assert a_running[:2] == (True, "launched") and b_running[:2] == (True, "launched")
    assert reaped == []
    roots = {str(a_app.parents[2].resolve()), str(b_app.parents[2].resolve())}
    assert set(artifacts._LAUNCHED_BACKENDS) == roots
    assert all(set(backends) == {"app"} for backends in artifacts._LAUNCHED_BACKENDS.values())
    assert len(artifacts._BACKEND_LAUNCH_LOCKS) == 2


class _FakePad:
    """anton's LocalScratchpadRuntime as the launcher reads it, recording its folder."""

    def __init__(self, name: str, *, workspace_path: Path, **_coding) -> None:
        self.workspace = workspace_path
        self._venv_python = str(workspace_path / ".anton" / "scratchpad-venvs" / name / "bin" / "python")

    def _ensure_venv(self) -> None:
        pass


@pytest.fixture
def fake_pads(monkeypatch):
    from cowork.services import scratchpad_runtime

    monkeypatch.setattr(scratchpad_runtime, "_pads", {})
    monkeypatch.setattr(
        scratchpad_runtime,
        "_resolve_coding",
        lambda **coding: tuple(coding.values()),
    )
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _FakePad)
    return scratchpad_runtime


async def test_two_peoples_same_slug_backends_get_their_own_venvs(stack, monkeypatch, fake_pads):
    """anton's launcher asks the pool for the slug's venv, then installs the
    artifact's requirements into that slug's pad. A pool keyed by slug alone
    would hand B's backend A's pad, inside A's project."""
    from cowork.services import artifacts

    a_app = _artifact(_general(ALICE), "app", title="A app")
    b_app = _artifact(_general(BOB), "app", title="B app")
    seen: dict[Path, tuple[str, Path]] = {}

    async def fake_launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, ds_env, health_timeout):
        # What anton's launcher asks of the pool: the venv, then the pad.
        venv_python = await scratchpad_pool.venv_python(slug)
        pad = await scratchpad_pool.get_or_create(slug)
        seen[artifact_folder] = (venv_python, pad.workspace)
        return {"port": 40000 + len(seen), "pid": len(seen)}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    monkeypatch.setattr(artifacts, "_probe_port", lambda _port, **_kw: False)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    monkeypatch.setattr(artifacts, "_BACKEND_LAUNCH_LOCKS", {})

    await artifacts._ensure_backend_running(a_app, 1, scope=ALICE)
    await artifacts._ensure_backend_running(b_app, 1, scope=BOB)

    a_root, b_root = a_app.parents[2].resolve(), b_app.parents[2].resolve()
    assert seen[a_app][1] == a_root
    assert seen[b_app][1] == b_root
    assert seen[b_app][0].startswith(str(b_root))
    assert set(fake_pads._pads) == {a_root, b_root}


def test_the_scratchpad_cap_counts_each_project_on_its_own(tmp_path, monkeypatch, fake_pads):
    monkeypatch.setattr(fake_pads, "MAX_PADS", 1)
    a_project, b_project = tmp_path / "a", tmp_path / "b"

    assert fake_pads.get_or_create("app", workspace_path=str(a_project)).workspace == a_project.resolve()
    assert fake_pads.get_or_create("other", workspace_path=str(b_project)).workspace == b_project.resolve()
    with pytest.raises(RuntimeError, match="Maximum concurrent scratchpads"):
        fake_pads.get_or_create("other", workspace_path=str(a_project))


class _LiveProc:
    returncode = None


async def test_another_persons_backend_on_the_port_is_not_the_callers(stack, monkeypatch):
    """Ports are reused: by the time B previews, the port in B's metadata.json
    can belong to A's backend. Only a backend this server launched for B's
    project and slug, on that port, counts as B's running backend."""
    from cowork.services import artifacts

    a_app = _artifact(_general(ALICE), "app", title="A app")
    b_app = _artifact(_general(BOB), "app", title="B app")
    launched: list[Path] = []

    async def fake_launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, ds_env, health_timeout):
        port = 5000 + len(launched)
        launched.append(artifact_folder)
        tracked_backends[slug] = {"proc": _LiveProc(), "port": port, "pid": port, "log_path": ""}
        return {"port": port, "pid": port}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    # Every port answers, as A's backend would on the port B's metadata names.
    monkeypatch.setattr(artifacts, "_probe_port", lambda _port, **_kw: True)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    monkeypatch.setattr(artifacts, "_BACKEND_LAUNCH_LOCKS", {})

    a_first = await artifacts._ensure_backend_running(a_app, 5000, scope=ALICE)
    b_open = await artifacts._ensure_backend_running(b_app, 5000, scope=BOB)
    a_again = await artifacts._ensure_backend_running(a_app, 5000, scope=ALICE)

    assert a_first == (True, "launched", 5000)
    assert b_open == (True, "launched", 5001)
    assert a_again == (True, "already_running", 5000)
    assert launched == [a_app, b_app]


async def test_desktop_counts_any_listener_as_the_artifacts_backend(monkeypatch, tmp_path):
    """The agent's own launch_backend tool starts backends this server never
    tracks, so on the desktop a listener on the artifact's port is its own."""
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import artifacts

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    try:
        app = _artifact(tmp_path / "projects" / "proj", "app", title="Desktop app")
        monkeypatch.setattr(
            "anton.core.artifacts.backend_launcher.launch_artifact_backend",
            lambda **_kw: pytest.fail("the agent's backend is already running"),
        )
        monkeypatch.setattr(artifacts, "_probe_port", lambda _port, **_kw: True)
        monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})

        result = await artifacts._ensure_backend_running(app, 4242, scope=LOCAL_SCOPE)
    finally:
        get_app_settings.cache_clear()

    assert result == (True, "already_running", 4242)


@pytest.mark.parametrize("listening", [False, True])
async def test_a_backend_launches_only_inside_the_callers_projects(stack, monkeypatch, listening):
    """Refused before the port is probed: a live port is no reason to report
    another person's artifact as running."""
    from cowork.services import artifacts

    a_app = _artifact(_general(ALICE), "app", title="A app")
    monkeypatch.setattr(
        "anton.core.artifacts.backend_launcher.launch_artifact_backend",
        lambda **_kw: pytest.fail("B must not start A's backend"),
    )
    monkeypatch.setattr(artifacts, "_probe_port", lambda _port, **_kw: listening)

    running, detail, _port = await artifacts._ensure_backend_running(a_app, 1, scope=BOB)

    assert running is False
    assert detail == "Artifact is not in a registered project."


# ── shared projects, desktop and hosted ─────────────────────────────────────


def test_shared_projects_keep_todays_shared_artifacts(stack):
    """Projects left off COWORK_PERSONAL_RESOURCES stay in the unkeyed root,
    where every person reads every artifact, as on today's stack."""
    stack.setenv("COWORK_PERSONAL_RESOURCES", "")
    get_app_settings.cache_clear()
    root = Path(get_app_settings().project.root_dir)
    dash = _artifact(root / "shared-proj", "dash", title="Shared forecast")
    client = _client()

    for headers in (A, B):
        preview = client.get(
            "/api/v1/artifacts/preview", params={"path": str(dash / "index.html")}, headers=headers
        )
        assert preview.status_code == 200
        served = client.get("/api/v1/artifacts/serve/shared-proj/dash/index.html", headers=headers)
        assert served.status_code == 200


def test_desktop_previews_the_unkeyed_projects_root(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    try:
        dash = _artifact(tmp_path / "home" / "projects" / "proj", "dash", title="Desktop forecast")
        client = _client()

        preview = client.get(
            "/api/v1/artifacts/preview", params={"path": str(dash / "index.html")}, headers=BEARER
        )
        assert preview.status_code == 200
        mounted = client.post(
            "/api/v1/artifacts/preview-mount", json={"path": str(dash / "index.html")}, headers=BEARER
        )
        assert mounted.status_code == 200
        asset = client.get(
            f"/api/v1/artifacts/preview-asset/{mounted.json()['token']}/index.html", headers=BEARER
        )
        assert asset.status_code == 200
    finally:
        get_app_settings.cache_clear()


def test_hosted_search_lists_no_artifacts(org_deployment, monkeypatch, tmp_path):
    """Hosted search has never listed artifacts. The scan of an org's own
    projects root would find this one, so the endpoint skips it explicitly."""
    from cowork.api.v1.endpoints.search import search_cowork
    from cowork.services.artifact_roots import artifacts_sources_for_scan

    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    org_scope = TenantScope(org_mode=True, org_id=org, user_id=SUB_A)
    _artifact(tmp_path / "shared" / org / "projects" / "proj", "dash", title="Quarterly forecast")
    assert artifacts_sources_for_scan(scope=org_scope), "the scan itself does see the org's tree"

    import asyncio

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        found = asyncio.run(search_cowork(ScopedSession(raw, org_scope), q="Quarterly", limit=25))

    assert [r for r in found["results"] if r["type"] == "artifact"] == []
