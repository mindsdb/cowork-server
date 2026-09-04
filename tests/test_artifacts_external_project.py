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
    """Adopting a folder while its name is already taken gives a row named
    `notes-2` over a directory named `notes`. The scan reports the basename,
    which addresses the wrong project."""
    from cowork.services.artifact_roots import artifacts_sources_for_scope

    folder = tmp_path / "notes"
    folder.mkdir()
    base = _artifacts_dir(folder)
    svc = ProjectService(_scoped(engine))
    svc.create_project("notes")
    adopting = ProjectService(_scoped(engine))
    project = adopting.create_project("notes", path=folder)
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
    svc = ProjectService(_scoped(engine))
    svc.create_project("notes")
    adopting = ProjectService(_scoped(engine))
    project = adopting.create_project("notes", path=folder)

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
