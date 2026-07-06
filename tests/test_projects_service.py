"""Projects can point at a user-chosen folder (ENG-384).

These lock in the semantics that flow depends on: creating a project at an
existing folder must not disturb its contents, renaming must never move the
directory (the path may be a folder the user owns outside the projects root),
and instructions live on the Project record.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.schemas.projects import ProjectCreateRequest
from cowork.services.projects import ProjectService


@pytest.fixture
def session(tmp_path, monkeypatch):
    # Keep skill reconciliation away from the developer's real skills dir.
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    get_app_settings.cache_clear()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def test_create_at_existing_folder_preserves_contents(session, tmp_path):
    existing = tmp_path / "my-notes"
    existing.mkdir()
    (existing / "notes.txt").write_text("keep me")

    project = ProjectService(session).create_project(
        "My Notes", path=existing, instructions="Organise my files"
    )

    assert project.path == str(existing)
    assert project.instructions == "Organise my files"
    assert (existing / "notes.txt").read_text() == "keep me"


def test_create_without_path_uses_projects_root(session):
    project = ProjectService(session).create_project("plain-project")

    path = Path(project.path)
    assert path.is_dir()
    assert path.parent == Path(get_app_settings().project.root_dir)
    assert project.instructions is None


def test_rename_never_moves_the_directory(session, tmp_path):
    existing = tmp_path / "owned-folder"
    existing.mkdir()
    svc = ProjectService(session)
    project = svc.create_project("Rename Me", path=existing)

    renamed = svc.update_project(project.id, name="Renamed Now")

    assert renamed.name == "Renamed-Now"
    assert renamed.path == str(existing)
    assert existing.is_dir()


def test_rename_to_same_name_is_a_noop(session, tmp_path):
    existing = tmp_path / "stable-folder"
    existing.mkdir()
    svc = ProjectService(session)
    project = svc.create_project("Stable Name", path=existing)

    renamed = svc.update_project(project.id, name="Stable Name")

    assert renamed.name == "Stable-Name"
    assert renamed.path == str(existing)


def test_instructions_update_and_clear(session):
    svc = ProjectService(session)
    project = svc.create_project("instructed", instructions="first")

    assert svc.update_project(project.id, instructions="second").instructions == "second"
    assert svc.update_project(project.id, instructions="").instructions is None


def test_create_request_rejects_relative_paths():
    assert ProjectCreateRequest(name="p", path="/abs/dir").path == Path("/abs/dir")
    assert ProjectCreateRequest(name="p", path="~/dir").path == Path.home() / "dir"
    assert ProjectCreateRequest(name="p").path is None

    with pytest.raises(ValidationError):
        ProjectCreateRequest(name="p", path="relative/dir")
