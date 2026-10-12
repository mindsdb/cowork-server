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
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
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


DRIVER_TEXT = "server closed the connection unexpectedly"


class _DatabaseOutage:
    """Fails every statement once `down` is set, as a dropped connection does,
    and records each statement tried while down."""

    def __init__(self, engine) -> None:
        self.down = False
        self.statements: list[str] = []
        event.listen(engine, "before_cursor_execute", self._refuse)

    def _refuse(self, conn, cursor, statement, parameters, context, executemany) -> None:
        if self.down:
            self.statements.append(statement)
            raise OperationalError(statement, parameters, Exception(DRIVER_TEXT))


@pytest.mark.parametrize("database_down", [False, True], ids=["database-up", "database-down"])
def test_a_failed_rename_restore_names_the_project_on_its_log_lines(
    engine, monkeypatch, owned_logger, caplog, database_down
):
    """The directory moved, the skill rewrite failed, and moving the directory
    back failed too. Both lines carry the project id as a record attribute,
    also when the database is gone: the rollback expired the project, so the
    id comes from the stage, never from a reload."""
    from cowork.services import projects as projects_module
    from cowork.services.skills import SkillService

    svc = _svc(engine)
    project_id = svc.create_project("notes").id
    original_rename = ProjectService._rename_in_root
    renames: list[tuple[Path, Path]] = []
    outage = _DatabaseOutage(engine)

    def move_then_fail_the_restore(self, old, new):
        renames.append((old, new))
        if len(renames) > 1:
            raise OSError("directory restore failed")
        original_rename(self, old, new)

    def fail_the_rewrites(self, rewrites):
        outage.down = database_down
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
    assert outage.statements == []
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


def test_a_failed_commit_restore_logs_the_project_without_querying_the_database(
    engine, monkeypatch, owned_logger, caplog
):
    """The commit failed because the database went away, and moving the
    directory back failed too. The rollback expired the project, so reading
    its id would query the database that just failed, and that query's error
    would replace the commit's. Both lines are written, nothing is queried,
    and the commit's own error reaches the caller."""
    from cowork.services import projects as projects_module

    svc = _svc(engine)
    project_id = svc.create_project("notes").id
    updating = _svc(engine)
    project, stage = updating.stage_project_update(
        project_id,
        resolved_name="renamed",
        is_active=None,
        display_label="renamed",
    )
    assert stage is not None and stage.directory_moved
    outage = _DatabaseOutage(engine)
    commit_error = OperationalError("COMMIT", {}, Exception(DRIVER_TEXT))

    def commit_fails():
        outage.down = True
        raise commit_error

    def fail_the_restore(self, old, new):
        raise OSError("directory restore failed")

    monkeypatch.setattr(updating.session, "commit", commit_fails)
    monkeypatch.setattr(ProjectService, "_rename_in_root", fail_the_restore)
    logged = owned_logger(projects_module.__name__)
    with pytest.raises(OperationalError) as raised:
        updating.commit_staged_project_update(project, stage)

    assert raised.value is commit_error
    assert outage.statements == []
    records = [
        r for r in caplog.records
        if r.name == logged.logger.name and r.levelno == logging.ERROR
    ]
    # The commit error is in each line's exception chain, so the database
    # filter replaces both messages; the call site and the id remain.
    assert [r.funcName for r in records] == ["rollback_project_rename", "_commit_staged_project_update"]
    assert [r.project_id for r in records] == [str(project_id)] * 2
    output = logged.output()
    assert output.count(f"[Project:{project_id}]: Database operation failed") == 2
    assert DRIVER_TEXT not in output


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
