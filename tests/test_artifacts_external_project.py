"""Artifact discovery for a project pointed at a folder the user chose.

The desktop scan only sees direct children of the projects root, so an adopted
folder is invisible to it: the agent writes artifacts into the user's own
folder and they then disappear from the artifact list and do not serve.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectService,
)


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


# -- path resolution ---------------------------------------------------------


def test_a_path_inside_an_adopted_folder_resolves_with_a_session(engine, tmp_path):
    """The path-addressed surface, which cannot use `artifacts_sources_for_project`
    because it is given a filesystem path and no project id."""
    from cowork.services.artifacts import resolve_artifact_path

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    artifact = _artifacts_dir(folder) / "dash"
    artifact.mkdir()
    primary = artifact / "index.html"
    primary.write_text("<html></html>")
    ProjectService(_scoped(engine)).create_project("notes", path=folder)

    resolved = resolve_artifact_path(str(primary), session=_scoped(engine))
    assert resolved == primary.resolve()


def test_the_same_path_is_refused_without_a_session(engine, tmp_path):
    """Callers that hold no session keep the old behavior, so nothing that
    resolves by scan today starts resolving by row behind their back."""
    from cowork.services.artifacts import resolve_artifact_path

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    artifact = _artifacts_dir(folder) / "dash"
    artifact.mkdir()
    primary = artifact / "index.html"
    primary.write_text("<html></html>")
    ProjectService(_scoped(engine)).create_project("notes", path=folder)

    with pytest.raises(FileNotFoundError):
        resolve_artifact_path(str(primary))


def test_the_artifact_root_climb_stops_at_an_adopted_container(engine, tmp_path):
    """The container set bounds the climb. It is empty for an adopted folder
    without the widening, and a `metadata.json` the user happens to keep in
    their own folder then becomes the artifact root — handing every sibling
    file in that folder to whatever reads the root."""
    from cowork.services.artifacts import _artifact_root_for

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    # A file of the user's own, above the artifacts dir. An unbounded climb
    # walks up to it; a bounded one never leaves `.anton/artifacts/`.
    (folder / "metadata.json").write_text("{}")
    base = _artifacts_dir(folder)
    loose = base / "stray"
    loose.mkdir()
    primary = loose / "index.html"
    primary.write_text("<html></html>")
    ProjectService(_scoped(engine)).create_project("notes", path=folder)

    root = _artifact_root_for(primary, session=_scoped(engine))
    assert root == loose.resolve()
    assert root != folder.resolve()


# -- serving -----------------------------------------------------------------


def test_the_serve_url_uses_the_row_name(projects_root, tmp_path):
    from cowork.services.artifacts import serve_url_for

    base = _artifacts_dir(tmp_path / "notes")
    (base / "report").mkdir()
    target = base / "report" / "index.html"
    target.write_text("<p>x</p>")

    url = serve_url_for(target, artifacts_base=base, project_name="notes-2")
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
    cards = list_artifacts(artifacts_sources_for_scope(session))
    card = next(c for c in cards if c["title"] == "Dashboard")
    assert card["serveUrl"] == (
        f"/api/v1/artifacts/serve/{project.name}/dash/index.html"
    )
    assert project.name != folder.name


def test_the_artifacts_route_lists_an_adopted_folder(projects_root, tmp_path):
    """Through HTTP, not through the resolver. The unit tests above passed
    while the local branch of `artifact_sources_for_request` still called the
    filesystem scan directly, so none of this was reachable from the route.
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
    assert served.text == "<html></html>"


def test_published_state_still_returns_the_blank_default_when_the_database_errors(
    engine, tmp_path, monkeypatch
):
    """Its contract is to answer for any path, never to raise. Resolving through
    the project row put a database read in the way, and the agent's own
    `publish_or_preview` calls this with no guard of its own.

    The path has to resolve, or the run never reaches the read under test.
    """
    from cowork.services import publish as publish_service
    from cowork.services.publish import published_owner_state, published_state

    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    artifact = _artifacts_dir(folder) / "dash"
    artifact.mkdir()
    primary = artifact / "index.html"
    primary.write_text("<html></html>")
    (artifact / "metadata.json").write_text("{}")
    ProjectService(_scoped(engine)).create_project("notes", path=folder)

    def _locked(_session=None):
        raise RuntimeError("database is locked")

    # Patched in `publish`, which binds the name at import; patching it in
    # `artifacts` would leave this binding pointing at the real one, and the
    # resolution above still has to succeed.
    monkeypatch.setattr(publish_service, "_artifact_dirs_for_scope", _locked)

    assert published_state(str(primary), _scoped(engine)) == {
        "report_id": "",
        "url": "",
        "published": False,
    }
    assert published_owner_state(str(primary), _scoped(engine)) == {}


def test_a_relative_path_in_two_roots_is_reported_as_ambiguous(engine, tmp_path):
    """The one behavior change an allocated project can see. Widening the root
    set widens this ambiguity: a relative path that matched one root before can
    match two once any folder is adopted, and the caller is told to send an
    absolute path rather than being handed an arbitrary one of them."""
    from cowork.services.artifacts import resolve_artifact_path

    # The service holds the session; letting it go detaches the row.
    svc = ProjectService(_scoped(engine))
    allocated = svc.create_project("reports")
    allocated_dash = _artifacts_dir(Path(allocated.path)) / "dash"
    allocated_dash.mkdir()
    (allocated_dash / "index.html").write_text("<html></html>")

    adopted = tmp_path / "Documents" / "old-work"
    adopted.mkdir(parents=True)
    adopted_dash = _artifacts_dir(adopted) / "dash"
    adopted_dash.mkdir()
    (adopted_dash / "index.html").write_text("<html></html>")
    ProjectService(_scoped(engine)).create_project("old-work", path=adopted)

    with pytest.raises(ValueError, match="multiple project artifact roots"):
        resolve_artifact_path("dash/index.html", session=_scoped(engine))

    # Without the adopted root in play the same relative path still resolves.
    assert resolve_artifact_path("dash/index.html") == (
        allocated_dash / "index.html"
    ).resolve()
