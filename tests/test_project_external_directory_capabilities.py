"""What a client is told about a project pointed at a folder the user chose.

Renaming moves the directory, which is only defined inside the projects root,
so the capability has to say so instead of advertising an action that fails
with an internal message. The same flag tells the delete confirmation that the
folder is kept.
"""
from __future__ import annotations

import logging
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


def _svc(engine) -> ProjectService:
    return ProjectService(ScopedSession(Session(engine), LOCAL_SCOPE))


# -- the predicate -----------------------------------------------------------


def test_an_adopted_folder_reads_as_external(engine, tmp_path):
    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    svc = _svc(engine)
    project = svc.create_project("notes", path=folder)
    assert svc.directory_is_external(project) is True


def test_an_allocated_directory_does_not(engine):
    svc = _svc(engine)
    project = svc.create_project("notes")
    assert svc.directory_is_external(project) is False


def test_the_seeded_general_project_does_not(engine):
    svc = _svc(engine)
    project = svc.get_project(GENERAL_PROJECT_ID)
    assert svc.directory_is_external(project) is False


# -- rename ------------------------------------------------------------------


def test_renaming_an_adopted_folder_is_refused_with_a_usable_message(
    engine, tmp_path
):
    """Not the internal "not a direct child of a trusted projects root" the
    directory move would otherwise raise."""
    folder = tmp_path / "Documents" / "notes"
    folder.mkdir(parents=True)
    svc = _svc(engine)
    project = svc.create_project("notes", path=folder)

    renaming = _svc(engine)
    with pytest.raises(ValueError, match="folder you chose"):
        renaming.stage_project_update(
            project.id,
            resolved_name="renamed",
            is_active=None,
            display_label="renamed",
        )


def test_the_refusal_leaves_the_folder_untouched(engine, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "draft.md").write_text("keep me")
    svc = _svc(engine)
    project = svc.create_project("notes", path=folder)

    renaming = _svc(engine)
    with pytest.raises(ValueError):
        renaming.stage_project_update(
            project.id,
            resolved_name="renamed",
            is_active=None,
            display_label="renamed",
        )

    assert folder.is_dir()
    assert (folder / "draft.md").read_text() == "keep me"


def test_an_allocated_project_still_renames(engine, projects_root):
    svc = _svc(engine)
    project = svc.create_project("notes")

    renaming = _svc(engine)
    updated, stage = renaming.stage_project_update(
        project.id,
        resolved_name="renamed",
        is_active=None,
        display_label="renamed",
    )
    assert updated.name == "renamed"
    assert stage is not None
    renaming.rollback_project_rename(stage)


def test_a_failed_rename_restore_names_the_project_on_its_log_lines(
    engine, monkeypatch, owned_logger, caplog
):
    """The directory moved, the skill rewrite failed, and moving the directory
    back failed too. Both lines carry the project id as a record attribute."""
    from cowork.services import projects as projects_module
    from cowork.services.skills import SkillService

    svc = _svc(engine)
    project_id = svc.create_project("notes").id
    original_rename = ProjectService._rename_in_root
    renames: list[tuple[Path, Path]] = []

    def move_then_fail_the_restore(self, old, new):
        renames.append((old, new))
        if len(renames) > 1:
            raise OSError("directory restore failed")
        original_rename(self, old, new)

    def fail_the_rewrites(self, rewrites):
        raise RuntimeError("skill rewrite failed")

    monkeypatch.setattr(ProjectService, "_rename_in_root", move_then_fail_the_restore)
    monkeypatch.setattr(SkillService, "apply_project_reference_rewrites", fail_the_rewrites)
    logged = owned_logger(projects_module.__name__)
    with pytest.raises(RuntimeError, match="skill rewrite failed"):
        _svc(engine).stage_project_update(
            project_id,
            resolved_name="renamed",
            is_active=None,
            display_label="renamed",
        )

    assert len(renames) == 2  # the move, then the failed move back
    records = [
        r for r in caplog.records
        if r.name == logged.logger.name and r.levelno == logging.ERROR
    ]
    assert [r.getMessage() for r in records] == [
        "Could not restore the directory for the project",
        "Could not fully restore the project after rename staging failed",
    ]
    assert [r.project_id for r in records] == [str(project_id)] * 2
    assert logged.output().count(f"[Project:{project_id}]: ") == 2


def test_a_label_only_change_is_allowed_on_an_adopted_folder(engine, tmp_path):
    """`display_name` is not the directory, so nothing moves and there is
    nothing to refuse."""
    folder = tmp_path / "notes"
    folder.mkdir()
    svc = _svc(engine)
    project = svc.create_project("notes", path=folder)

    updating = _svc(engine)
    updated, stage = updating.stage_project_update(
        project.id,
        resolved_name=None,
        is_active=None,
        display_label="My notes",
    )
    assert updated.display_name == "My notes"
    assert stage is None
