"""Pointing a project at a folder the user already has.

A project directory is normally allocated inside the projects root, and
several subsystems find projects by scanning that root. Adopting an outside
folder is therefore deliberately narrow: desktop only, an existing directory
only, never inside the root, and never a folder another project claims.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.schemas.projects import ProjectCreateRequest
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectPathNotAllowedError,
    ProjectService,
)

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


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


def _svc(engine, scope=LOCAL_SCOPE) -> ProjectService:
    return ProjectService(ScopedSession(Session(engine), scope))


def _folder(base: Path, name: str) -> Path:
    target = base / name
    target.mkdir(parents=True)
    return target


# -- the desktop-only gate ---------------------------------------------------


def test_org_mode_refuses_a_chosen_folder(engine, monkeypatch, tmp_path):
    """Refused before any filesystem access. The path below does not exist, so
    an implementation that statted first would raise the wrong error, and on a
    shared deployment that stat is an existence oracle for a server path."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    svc = _svc(engine, TenantScope(org_mode=True, org_id=ORG, user_id="u1"))
    with pytest.raises(ProjectPathNotAllowedError):
        svc.create_project("notes", path=tmp_path / "definitely-absent")


def test_a_server_bound_off_loopback_refuses_a_chosen_folder(
    engine, monkeypatch, tmp_path
):
    """The peer address cannot carry this decision. The container runs uvicorn
    with `--forwarded-allow-ips "*"`, so X-Forwarded-For rewrites
    request.client and a remote caller can present 127.0.0.1. What it cannot
    present is the address the server was told to bind."""
    monkeypatch.setenv("COWORK_SERVER_HOST", "0.0.0.0")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    folder = _folder(tmp_path, "notes")
    with pytest.raises(ProjectPathNotAllowedError, match="listening only on"):
        _svc(engine).create_project("notes", path=folder)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.1.1"])
def test_a_loopback_bound_server_allows_a_chosen_folder(
    engine, monkeypatch, tmp_path, host
):
    monkeypatch.setenv("COWORK_SERVER_HOST", host)
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    folder = _folder(tmp_path, f"notes-{host.replace(':', '-')}")
    svc = _svc(engine)
    project = svc.create_project(f"notes-{host.replace(':', '-')}", path=folder)
    assert Path(project.path) == folder.resolve()


def test_a_server_on_a_routable_address_refuses_it(engine, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_SERVER_HOST", "10.0.0.5")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    folder = _folder(tmp_path, "notes")
    with pytest.raises(ProjectPathNotAllowedError, match="listening only on"):
        _svc(engine).create_project("notes", path=folder)


# -- what counts as a folder -------------------------------------------------


def test_a_missing_directory_is_refused(engine, tmp_path):
    with pytest.raises(ValueError, match="existing local folder"):
        _svc(engine).create_project("notes", path=tmp_path / "absent")


def test_a_file_is_not_a_folder(engine, tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("x")
    with pytest.raises(ValueError, match="existing local folder"):
        _svc(engine).create_project("notes", path=target)


def test_a_folder_inside_the_projects_root_is_refused(engine, projects_root):
    """Inside the root a chosen folder can equal the path `delete_project`
    re-derives from a name, and it would then be rmtree'd as though Cowork had
    allocated it."""
    inside = _folder(projects_root, "notes")
    with pytest.raises(ValueError, match="outside the Cowork projects directory"):
        _svc(engine).create_project("notes", path=inside)


def test_the_projects_root_itself_is_refused(engine, projects_root):
    projects_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="outside the Cowork projects directory"):
        _svc(engine).create_project("notes", path=projects_root)


def test_a_folder_another_project_already_uses_is_refused(engine, tmp_path):
    folder = _folder(tmp_path, "shared")
    _svc(engine).create_project("first", path=folder)
    with pytest.raises(ValueError, match="already uses this folder"):
        _svc(engine).create_project("second", path=folder)


# -- the happy path ----------------------------------------------------------


def test_the_row_points_at_the_chosen_folder(engine, tmp_path, projects_root):
    folder = _folder(tmp_path / "Documents", "notes")
    # The service holds the session; letting it go collects it and the row
    # detaches, because create_project commits without refreshing.
    svc = _svc(engine)
    project = svc.create_project("Notes", path=folder)
    assert Path(project.path) == folder.resolve()
    assert not (projects_root / project.name).exists()


def test_the_users_own_files_are_left_alone(engine, tmp_path):
    folder = _folder(tmp_path, "notes")
    (folder / "draft.md").write_text("keep me")
    _svc(engine).create_project("notes", path=folder)
    assert (folder / "draft.md").read_text() == "keep me"


def test_a_taken_name_is_refused_rather_than_bumped(engine, tmp_path):
    """`name` is the lookup key, the URL segment and the folder basename, and
    `get_project_by_name` is a `.first()` on an unordered select. The allocated
    path can bump to `-2` because `mkdir` arbitrates a concurrent pair;
    adoption creates nothing, so it refuses instead."""
    first_svc = _svc(engine)
    first_svc.create_project("notes", path=_folder(tmp_path, "a"))
    with pytest.raises(ValueError, match="already exists"):
        _svc(engine).create_project("notes", path=_folder(tmp_path, "b"))


def test_a_taken_name_is_refused_against_an_allocated_project(engine, tmp_path):
    allocated = _svc(engine)
    allocated.create_project("notes")
    with pytest.raises(ValueError, match="already exists"):
        _svc(engine).create_project("notes", path=_folder(tmp_path, "b"))


def test_the_name_is_what_the_user_typed_not_the_folder(engine, tmp_path):
    """The two are independent, so an adopted folder's basename still differs
    from the row name whenever the user names the project something else.
    Every scan-based subsystem had assumed they were the same string."""
    svc = _svc(engine)
    project = svc.create_project("My Reports", path=_folder(tmp_path, "notes"))
    assert project.name == "My-Reports"


def test_a_folder_is_still_optional(engine, projects_root):
    svc = _svc(engine)
    project = svc.create_project("notes")
    assert Path(project.path) == projects_root / project.name
    assert (projects_root / project.name).is_dir()


# -- the wire ----------------------------------------------------------------


@pytest.mark.parametrize("raw", ["notes", "./notes", "../notes", "."])
def test_a_relative_path_is_rejected_by_the_schema(raw):
    with pytest.raises(ValidationError):
        ProjectCreateRequest(name="n", path=raw)


@pytest.mark.parametrize("raw", ["~", "~/Documents", "~root/x", "~nosuchuser/x"])
def test_a_tilde_path_is_rejected_rather_than_expanded(raw):
    """expanduser consults the passwd database, so expanding here would let a
    caller tell a real account from a missing one before the service has
    refused the request at all."""
    with pytest.raises(ValidationError):
        ProjectCreateRequest(name="n", path=raw)


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_no_folder_chosen_reads_as_none(raw):
    assert ProjectCreateRequest(name="n", path=raw).path is None


def test_an_omitted_path_is_none():
    assert ProjectCreateRequest(name="n").path is None


def _loopback_client():
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    return TestClient(create_app(), client=("127.0.0.1", 54321))


def test_a_missing_folder_is_a_client_error_not_a_crash(projects_root, tmp_path):
    """The create endpoint carried no `ValueError` mapping, so every rejection
    added here would otherwise reach the client as a 500."""
    res = _loopback_client().post(
        "/api/v1/projects/",
        json={"name": "notes", "path": str(tmp_path / "absent")},
    )
    assert res.status_code == 400, res.text


def test_a_folder_can_be_chosen_from_loopback(projects_root, tmp_path):
    # The app database is shared across the HTTP tests in this suite, and an
    # adopted folder now refuses a name that is already taken.
    folder = _folder(tmp_path, "chosen-from-loopback")
    res = _loopback_client().post(
        "/api/v1/projects/",
        json={"name": "chosen-from-loopback", "path": str(folder)},
    )
    assert res.status_code == 201, res.text
    assert Path(res.json()["path"]) == folder.resolve()
    assert res.json()["capabilities"]["directoryIsExternal"] is True
    assert res.json()["capabilities"]["canRename"] is False


def test_a_non_loopback_caller_cannot_choose_a_folder(projects_root, tmp_path):
    """`tenancy_mode` is local on a self-host deployment that binds 0.0.0.0.
    Without this, a chosen path plus the project-file endpoints is read and
    write anywhere the server user can reach."""
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    folder = _folder(tmp_path, "refused-remotely")
    remote = TestClient(create_app(), client=("203.0.113.7", 44321))
    res = remote.post(
        "/api/v1/projects/", json={"name": "refused-remotely", "path": str(folder)}
    )
    assert res.status_code == 403, res.text


def test_a_non_loopback_caller_can_still_create_a_normal_project(projects_root):
    """The gate is on the chosen folder, not on project creation."""
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    remote = TestClient(create_app(), client=("203.0.113.7", 44321))
    res = remote.post("/api/v1/projects/", json={"name": "remote-no-folder"})
    assert res.status_code == 201, res.text
