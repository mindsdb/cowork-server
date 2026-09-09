"""The seeded `general` row after the projects root moves.

Changing the projects root left the seeded row on its original path, so every
turn kept writing under the old root, and removing the old directory made the
project unprovisionable: `ensure_dir_exists` refuses a path that no longer
matches the current root.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
)
from cowork.services.artifacts import list_artifacts
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectService,
)


def _point_at(monkeypatch, root: Path) -> None:
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(root))
    get_app_settings.cache_clear()


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    """An old and a current projects root, settings pointed at the old one."""
    from cowork.common.settings.app_settings import get_app_settings

    old = tmp_path / "old_projects"
    new = tmp_path / "new_projects"
    (old / GENERAL_PROJECT).mkdir(parents=True)
    new.mkdir()
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    _point_at(monkeypatch, old)
    yield old, new
    get_app_settings.cache_clear()


@pytest.fixture()
def engine(roots):
    old, _ = roots
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    with Session(eng) as seed:
        seed.add(
            Project(
                id=GENERAL_PROJECT_ID,
                name=GENERAL_PROJECT,
                path=str(old / GENERAL_PROJECT),
                is_active=True,
            )
        )
        seed.commit()
    return eng


def _scoped(engine) -> ScopedSession:
    return ScopedSession(Session(engine), LOCAL_SCOPE)


def _write_artifact(base: Path, slug: str, title: str) -> None:
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": str(uuid4()),
                "slug": slug,
                "name": title,
                "primary": "index.html",
                "type": "html-app",
            }
        )
    )


def _titles(session) -> tuple[set[str], set[str]]:
    """Titles from the project-scoped listing and the unparameterized one.

    The rail calls the first and the panel the second; the ticket asks for them
    to agree after a root change.
    """
    by_id = list_artifacts(
        artifacts_sources_for_project(session, GENERAL_PROJECT_ID)
    )
    unfiltered = list_artifacts(artifacts_sources_for_scope(session))
    return {c["title"] for c in by_id}, {c["title"] for c in unfiltered}


def test_the_row_follows_the_current_root(roots, engine, monkeypatch):
    old, new = roots
    _write_artifact(
        old / GENERAL_PROJECT / ".anton" / "artifacts", "old-dash", "Old dashboard"
    )
    _point_at(monkeypatch, new)

    project = ProjectService(_scoped(engine)).ensure_general_for_scope()

    assert project is not None
    assert Path(project.path).parent.resolve() == new.resolve()
    assert Path(project.path).is_dir()


def test_an_artifact_written_after_the_move_reaches_both_listings(
    roots, engine, monkeypatch
):
    """Asserting only that the two listings match passes vacuously while both
    are empty, which is the state on either side of the fix. Write into the
    re-pointed directory first."""
    old, new = roots
    _write_artifact(
        old / GENERAL_PROJECT / ".anton" / "artifacts", "old-dash", "Old dashboard"
    )
    _point_at(monkeypatch, new)

    project = ProjectService(_scoped(engine)).ensure_general_for_scope()
    _write_artifact(
        Path(project.path) / ".anton" / "artifacts", "new-dash", "New dashboard"
    )

    by_id, unfiltered = _titles(_scoped(engine))
    assert by_id == {"New dashboard"}
    assert unfiltered == {"New dashboard"}


def test_the_listings_agree_when_both_roots_hold_artifacts(roots, engine, monkeypatch):
    """Reachable by pointing the variable at a copy of the projects folder, or
    by flipping it back after a re-point. Before the fix the row stayed on the
    old root and the two listings disagreed."""
    old, new = roots
    _write_artifact(
        old / GENERAL_PROJECT / ".anton" / "artifacts", "old-dash", "Old dashboard"
    )
    _write_artifact(
        new / GENERAL_PROJECT / ".anton" / "artifacts", "new-dash", "New dashboard"
    )
    _point_at(monkeypatch, new)

    ProjectService(_scoped(engine)).ensure_general_for_scope()

    by_id, unfiltered = _titles(_scoped(engine))
    assert by_id == {"New dashboard"}
    assert unfiltered == {"New dashboard"}
    # Left behind on purpose: the row moves, the bytes stay.
    assert (old / GENERAL_PROJECT / ".anton" / "artifacts" / "old-dash").is_dir()


def test_a_scaffold_only_old_directory_still_re_points(roots, engine, monkeypatch):
    """A turn recreates `skills/` under the row's path on every run, so "the
    directory holds something" is never false once the project has been used."""
    old, new = roots
    (old / GENERAL_PROJECT / "skills").mkdir(parents=True, exist_ok=True)
    _point_at(monkeypatch, new)

    project = ProjectService(_scoped(engine)).ensure_general_for_scope()

    assert Path(project.path).parent.resolve() == new.resolve()


def test_the_default_guard_still_declines_a_populated_directory(
    roots, engine, monkeypatch
):
    """Org mode relies on the default: swapping a populated path for an empty
    directory would strand that organization's work."""
    old, new = roots
    populated = old / GENERAL_PROJECT
    _write_artifact(
        populated / ".anton" / "artifacts", "old-dash", "Old dashboard"
    )
    _point_at(monkeypatch, new)

    session = _scoped(engine)
    project = session.get(Project, GENERAL_PROJECT_ID)
    ProjectService(session)._repoint_if_stale(project)

    assert Path(project.path).resolve() == populated.resolve()
