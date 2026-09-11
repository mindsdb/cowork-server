"""Deleting a project must not delete a folder the user chose.

This is the property the whole feature rests on: the row is Cowork's, the
folder is the user's. It already held, but only as a side effect of the stored
path differing from the one re-derived from the project's name, so it is
pinned here rather than left to be re-derived by the next reader.

`delete_project` keeps returning a bool. Three call sites test it for
truthiness (`endpoints/projects.py` at the create rollback, the 404 branch,
and the org path), so a richer return would silently kill the 404.
"""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

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


def _svc(engine) -> ProjectService:
    return ProjectService(ScopedSession(Session(engine), LOCAL_SCOPE))


def test_the_chosen_folder_and_its_contents_survive(engine, tmp_path):
    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    (folder / "draft.md").write_text("keep me")
    (folder / "sub").mkdir()
    (folder / "sub" / "deep.txt").write_text("me too")

    created = _svc(engine)
    project = created.create_project("notes", path=folder)

    deleting = _svc(engine)
    assert deleting.delete_project(project.id) is True

    assert folder.is_dir()
    assert (folder / "draft.md").read_text() == "keep me"
    assert (folder / "sub" / "deep.txt").read_text() == "me too"


def test_the_row_is_gone_even_though_the_folder_stays(engine, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    created = _svc(engine)
    project = created.create_project("notes", path=folder)

    deleting = _svc(engine)
    deleting.delete_project(project.id)

    assert _svc(engine).get_project_by_name_or_none("notes") is None


def test_an_allocated_directory_is_still_removed(engine):
    """The other half of the contract: a directory Cowork made is Cowork's."""
    created = _svc(engine)
    project = created.create_project("notes")
    allocated = Path(project.path)
    assert allocated.is_dir()

    deleting = _svc(engine)
    deleting.delete_project(project.id)

    assert not allocated.exists()


def test_keeping_the_folder_is_not_logged_as_a_warning(engine, tmp_path, caplog):
    """It is the expected outcome now, and a warning on every delete of an
    adopted folder trains operators to ignore the one that means something."""
    folder = tmp_path / "notes"
    folder.mkdir()
    created = _svc(engine)
    project = created.create_project("notes", path=folder)

    deleting = _svc(engine)
    with caplog.at_level(logging.INFO, logger="cowork.services.projects"):
        deleting.delete_project(project.id)

    records = [r for r in caplog.records if "leaving it in place" in r.message]
    assert [r.levelno for r in records] == [logging.INFO]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_missing_project_still_returns_falsy(engine):
    """The 404 branch and the create rollback both test this for truthiness."""
    assert _svc(engine).delete_project(uuid4()) is False
