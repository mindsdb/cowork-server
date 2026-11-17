"""Artifact discovery for a project pointed at a folder the user chose.

The desktop scan only sees direct children of the projects root, so an adopted
folder is invisible to it: the agent writes artifacts into the user's own
folder and they then disappear from the artifact list and do not serve.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectService,
)
from _linked_folders import SECRET, LoggedRefusal, linked_folder_refusals, replace_with_link

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


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


@pytest.fixture()
def engine(projects_root):
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    with Session(eng) as seed:
        seed.add(
            Project(
                id=GENERAL_PROJECT_ID,
                name=GENERAL_PROJECT,
                path=str(projects_root / "general"),
                is_active=True,
            )
        )
        seed.commit()
    return eng


def _scoped(engine) -> ScopedSession:
    return ScopedSession(Session(engine), LOCAL_SCOPE)


def _artifacts_dir(folder: Path) -> Path:
    target = folder / ".anton" / "artifacts"
    target.mkdir(parents=True)
    return target


# -- discovery ---------------------------------------------------------------


def test_an_adopted_folder_is_a_discoverable_artifacts_source(engine, tmp_path):
    from cowork.services.artifact_roots import artifacts_sources_for_scope

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    base = _artifacts_dir(folder)
    svc = ProjectService(_scoped(engine))
    svc.create_project("notes", path=folder)

    session = _scoped(engine)
    bases = {Path(s.base).resolve() for s in artifacts_sources_for_scope(session)}
    assert base.resolve() in bases


def test_the_source_carries_the_row_name_not_the_folder_basename(engine, tmp_path):
    """The project's name is what the user typed, not the folder's basename,
    so the two differ whenever they name the project something else. The scan
    reports the basename, which addresses the wrong project."""
    from cowork.services.artifact_roots import artifacts_sources_for_scope

    folder = tmp_path / "notes"
    folder.mkdir()
    base = _artifacts_dir(folder)
    adopting = ProjectService(_scoped(engine))
    project = adopting.create_project("Reports", path=folder)
    assert project.name != folder.name

    session = _scoped(engine)
    match = next(
        s
        for s in artifacts_sources_for_scope(session)
        if Path(s.base).resolve() == base.resolve()
    )
    assert match.project_name == project.name


def test_an_adopted_folder_without_an_artifacts_dir_is_not_a_source(engine, tmp_path):
    from cowork.services.artifact_roots import artifacts_sources_for_scope

    folder = tmp_path / "plain"
    folder.mkdir()
    svc = ProjectService(_scoped(engine))
    svc.create_project("plain", path=folder)

    session = _scoped(engine)
    anchors = {Path(s.trusted_anchor).resolve() for s in artifacts_sources_for_scope(session)}
    assert folder.resolve() not in anchors


def test_an_in_root_project_is_still_found_by_the_scan(engine):
    """The scan is unchanged; adopted folders are added to it, not swapped in."""
    from cowork.services.artifact_roots import artifacts_sources_for_scope

    svc = ProjectService(_scoped(engine))
    project = svc.create_project("reports")
    base = _artifacts_dir(Path(project.path))

    session = _scoped(engine)
    bases = {Path(s.base).resolve() for s in artifacts_sources_for_scope(session)}
    assert base.resolve() in bases


# -- serving -----------------------------------------------------------------


def test_the_serve_url_uses_the_row_name(projects_root, tmp_path):
    from cowork.services.artifacts import serve_url_for

    base = _artifacts_dir(tmp_path / "notes")
    (base / "report").mkdir()
    target = base / "report" / "index.html"
    target.write_text("<p>x</p>")

    url = serve_url_for(target, artifacts_base=base, project_name="notes-2", scope=LOCAL_SCOPE)
    assert url == "/api/v1/artifacts/serve/notes-2/report/index.html"


def test_the_serve_route_resolves_an_adopted_folder(engine, tmp_path):
    from cowork.services.artifacts import _project_artifacts_base

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    base = _artifacts_dir(folder)
    svc = ProjectService(_scoped(engine))
    project = svc.create_project("notes", path=folder)

    session = _scoped(engine)
    resolved = _project_artifacts_base(project.name, session)
    assert resolved is not None
    assert resolved.resolve() == base.resolve()


def test_an_unknown_project_name_resolves_to_nothing(engine):
    from cowork.services.artifacts import _project_artifacts_base

    session = _scoped(engine)
    assert _project_artifacts_base("no-such-project", session) is None


def test_a_traversal_attempt_still_resolves_to_nothing(engine):
    from cowork.services.artifacts import _project_artifacts_base

    session = _scoped(engine)
    for attempt in ["..", ".", "a/b", "a\\b", ""]:
        assert _project_artifacts_base(attempt, session) is None


def test_a_listed_card_from_an_adopted_folder_serves_under_the_row_name(
    engine, tmp_path
):
    """End to end: the agent writes into the user's folder, and the card the
    grid renders carries a serve URL addressed by the project's row name."""
    import json

    from cowork.services.artifact_roots import artifacts_sources_for_scope
    from cowork.services.artifacts import list_artifacts

    folder = tmp_path / "notes"
    folder.mkdir()
    base = _artifacts_dir(folder)
    adopting = ProjectService(_scoped(engine))
    project = adopting.create_project("Reports", path=folder)

    artifact = base / "dash"
    artifact.mkdir()
    (artifact / "index.html").write_text("<html></html>")
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "slug": "dash",
                "name": "Dashboard",
                "primary": "index.html",
                "type": "html-app",
            }
        )
    )

    session = _scoped(engine)
    cards = [card.to_json() for card in list_artifacts(artifacts_sources_for_scope(session))]
    card = next(c for c in cards if c["title"] == "Dashboard")
    assert card["serveUrl"] == (
        f"/api/v1/artifacts/serve/{project.name}/dash/index.html"
    )
    assert project.name != folder.name


def test_the_artifacts_route_lists_an_adopted_folder(projects_root, tmp_path):
    """Through HTTP, not through the resolver. The unit tests above passed
    while the route's local branch still called the filesystem scan
    directly, so none of this was reachable from the route.
    """
    import json

    from fastapi.testclient import TestClient

    from cowork.server import create_app

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    artifact = folder / ".anton" / "artifacts" / "dash"
    artifact.mkdir(parents=True)
    (artifact / "index.html").write_text("<html></html>")
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "slug": "dash",
                "name": "Adopted dashboard",
                "primary": "index.html",
                "type": "html-app",
            }
        )
    )

    # base_url sets scope["server"], which the chosen-folder gate reads.
    client = TestClient(
        create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )
    created = client.post(
        "/api/v1/projects/",
        json={"name": "adopted-dash-project", "path": str(folder)},
    )
    assert created.status_code == 201, created.text
    project_name = created.json()["name"]
    project_id = created.json()["id"]

    unfiltered = client.get("/api/v1/artifacts/")
    assert unfiltered.status_code == 200, unfiltered.text
    card = next(
        c for c in unfiltered.json() if c["title"] == "Adopted dashboard"
    )
    assert card["serveUrl"] == f"/api/v1/artifacts/serve/{project_name}/dash/index.html"

    # The rail addresses artifacts by project id, which is a different
    # resolver, and it must agree.
    by_id = client.get(f"/api/v1/artifacts/?project_id={project_id}")
    assert by_id.status_code == 200, by_id.text
    by_id_card = next(
        c for c in by_id.json() if c["title"] == "Adopted dashboard"
    )
    assert by_id_card["serveUrl"] == card["serveUrl"]

    served = client.get(card["serveUrl"])
    assert served.status_code == 200, served.text
    # Every HTML preview now carries the storage shim (see preview_html.py);
    # this route is not what's under test here, only that it resolves at all.
    assert "anton-preview" in served.text


# -- serving through links and stale rows ------------------------------------


def _local_client():
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    # base_url sets scope["server"], which the chosen-folder gate reads.
    return TestClient(
        create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )


class _Adopted(BaseModel):
    id: str
    name: str


def _adopt(client, folder: Path, name: str) -> _Adopted:
    created = client.post("/api/v1/projects/", json={"name": name, "path": str(folder)})
    assert created.status_code == 201, created.text
    return _Adopted.model_validate(created.json())


def test_serve_refuses_an_adopted_folders_artifacts_linked_to_its_anton(projects_root, tmp_path, caplog):
    folder = tmp_path / "Documents" / "notes"
    (folder / ".anton").mkdir(parents=True)
    (folder / ".anton" / ".env").write_text(SECRET)
    client = _local_client()
    adopted = _adopt(client, folder, "linked-artifacts")
    replace_with_link(folder / ".anton" / "artifacts", folder / ".anton")

    with caplog.at_level("WARNING", logger="cowork.services.artifacts"):
        served = client.get(f"/api/v1/artifacts/serve/{adopted.name}/.env")

    assert SECRET not in served.text, (served.status_code, served.text[:200])
    assert served.status_code == 404
    assert linked_folder_refusals(caplog.records) == [
        LoggedRefusal(event="artifact_folder_refused", reason="linked_folder", project_id=adopted.id)
    ]


def test_serve_refuses_an_adopted_folders_anton_linked_elsewhere(projects_root, tmp_path):
    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    page = elsewhere / "artifacts" / "dash" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>outside the project</html>")
    client = _local_client()
    name = _adopt(client, folder, "linked-anton").name
    replace_with_link(folder / ".anton", elsewhere)

    served = client.get(f"/api/v1/artifacts/serve/{name}/dash/index.html")

    assert "outside the project" not in served.text, (served.status_code, served.text[:200])
    assert served.status_code == 404


@pytest.mark.parametrize(
    "folder_state",
    [
        "no-artifacts",
        "dangling-link",
        pytest.param(
            "unreadable-parent",
            marks=pytest.mark.skipif(
                os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root"
            ),
        ),
    ],
)
def test_serve_does_not_take_a_root_folder_of_the_same_name_for_an_adopted_project(
    projects_root, tmp_path, folder_state, caplog
):
    """The scan lists every directory in the projects root, so falling through
    to it would serve a leftover folder under the adopted project's name. A
    dangling link at the adopted folder's path is not a stale row either, nor
    is a folder whose parent lstat can't read. None of these is a linked
    `.anton`, so none logs the linked-folder warning."""
    import shutil

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    client = _local_client()
    name = _adopt(client, folder, "same-name").name
    if folder_state == "dangling-link":
        shutil.rmtree(folder)
        folder.symlink_to(tmp_path / "nowhere", target_is_directory=True)
    else:
        shutil.rmtree(folder / ".anton", ignore_errors=True)
    page = projects_root / name / ".anton" / "artifacts" / "dash" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>leftover in the root</html>")
    if folder_state == "unreadable-parent":
        folder.parent.chmod(0)
    try:
        with caplog.at_level("WARNING", logger="cowork.services.artifacts"):
            served = client.get(f"/api/v1/artifacts/serve/{name}/dash/index.html")
    finally:
        folder.parent.chmod(0o755)

    assert "leftover in the root" not in served.text, (served.status_code, served.text[:200])
    assert served.status_code == 404
    assert linked_folder_refusals(caplog.records) == []


def test_serve_reads_an_adopted_folder_reached_through_a_link(projects_root, tmp_path):
    """Only `.anton` and `.anton/artifacts` must be real directories; the
    adopted folder itself may be an alias. Adoption stores the resolved
    path, so the folder moves after it and a link takes its stored place."""
    import shutil

    stored = tmp_path / "Documents" / "notes"
    page = stored / ".anton" / "artifacts" / "dash" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>through the alias</html>")
    client = _local_client()
    name = _adopt(client, stored, "aliased").name
    real = tmp_path / "real" / "notes"
    real.parent.mkdir(parents=True)
    shutil.move(stored, real)
    stored.symlink_to(real, target_is_directory=True)

    served = client.get(f"/api/v1/artifacts/serve/{name}/dash/index.html")

    assert served.status_code == 200, served.text
    assert "through the alias" in served.text


def test_serve_finds_a_stale_rows_project_in_the_projects_root(projects_root, tmp_path):
    """A row created before the projects root moved points at a folder that is
    gone; the project now lives under the new root, where the scan finds it.
    Passes before the fix too: a regression guard."""
    import shutil

    folder = tmp_path / "old-root" / "moved"
    folder.mkdir(parents=True)
    client = _local_client()
    name = _adopt(client, folder, "moved").name
    shutil.rmtree(folder)
    page = projects_root / name / ".anton" / "artifacts" / "dash" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>moved with the root</html>")

    served = client.get(f"/api/v1/artifacts/serve/{name}/dash/index.html")

    assert served.status_code == 200, served.text
    assert "moved with the root" in served.text
