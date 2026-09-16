"""Pointing a project at a folder the user already has.

A project directory is normally allocated inside the projects root, and
several subsystems find projects by scanning that root. Adopting an outside
folder is therefore deliberately narrow: desktop only, an existing directory
only, never inside the root, and never a folder another project claims.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.schemas.projects import ProjectCreateRequest
import cowork.services.projects as projects_module
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectNameLockBusyError,
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


@pytest.fixture()
def race_engine(projects_root, tmp_path):
    """A file-backed engine, so two sessions get two real connections.

    On the shared in-memory `StaticPool` engine both ride one connection and
    would see each other's uncommitted rows, which is the opposite of the
    isolation these tests need.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'race.db'}")
    SQLModel.metadata.create_all(engine)
    # `dev_setup` writes this row on every install, so a projects table
    # without it is a state no deployment is ever in.
    with Session(engine) as seed:
        seed.add(
            Project(
                id=GENERAL_PROJECT_ID,
                name=GENERAL_PROJECT,
                path=str(projects_root / "general"),
                is_active=True,
            )
        )
        seed.commit()
    return engine


class _Ran:
    """A thread and what its call did, so "returned" is distinguishable
    from "raised". Starts on construction."""

    def __init__(self, target):
        self.returned = threading.Event()
        self.value = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._body, args=(target,))
        self._thread.start()

    def _body(self, target) -> None:
        try:
            self.value = target()
        except BaseException as exc:
            self.error = exc
        finally:
            self.returned.set()

    def _joined(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)
        assert not self._thread.is_alive(), "the thread never finished"

    def settled(self, timeout: float = 10):
        """Insist the call returned, and hand back what it returned."""
        self._joined(timeout)
        if self.error is not None:
            raise self.error
        return self.value

    def raised(self, timeout: float = 10) -> BaseException:
        """Insist the call failed, and hand back why."""
        self._joined(timeout)
        assert self.error is not None, f"expected a failure, got {self.value!r}"
        return self.error


def _park_inside_the_lock(svc) -> tuple[threading.Event, threading.Event]:
    """Stall `svc` mid-create, after its name check and before its commit.

    `_unique_display_name` is the next call after the name is settled and is
    inside the locked region, so patching it holds the name without touching
    the code under test.
    """
    parked = threading.Event()
    release = threading.Event()
    real_display = svc._unique_display_name

    def park(*args, **kwargs):
        parked.set()
        assert release.wait(timeout=10)
        return real_display(*args, **kwargs)

    svc._unique_display_name = park
    return parked, release


def _rows_named(engine, name: str) -> list[Project]:
    with Session(engine) as check:
        return list(check.exec(sa.select(Project).where(Project.name == name)).all())


# The lock covers the name namespace, not the adoption path, so each of these
# is a different pair racing for one name. Unserialised, all three commit two
# rows called `notes` -- and `name` is the lookup key, with
# `get_project_by_name` a `.first()` on an unordered select, so one of the two
# projects becomes unreachable and its files resolve to the other's directory.
# SQLite serialises the writes but never re-evaluates the reads.
#
# Each test waits a beat on the loser to prove it did not proceed. That wait
# is the assertion: "was serialised" is a negative, and the row count that
# follows only catches the regression if the loser got as far as its own read
# while the winner was parked.


def test_two_concurrent_adoptions_cannot_commit_the_same_name(race_engine, tmp_path):
    first = _svc(race_engine)
    second = _svc(race_engine)
    parked, release = _park_inside_the_lock(first)

    winner = _Ran(lambda: first.create_project("notes", path=_folder(tmp_path, "a")))
    assert parked.wait(timeout=10), "the first adoption never settled its name"

    loser = _Ran(lambda: second.create_project("notes", path=_folder(tmp_path, "b")))
    finished_while_held = loser.returned.wait(timeout=1)
    release.set()
    winner.settled()
    error = loser.raised()
    assert not finished_while_held, "the second adoption ran while the first held the name"
    # Adoption refuses a taken name rather than bumping it: it creates nothing,
    # so `<root>/notes-2` would not describe the folder it points at.
    assert isinstance(error, ValueError)
    assert "already exists" in str(error)
    assert len(_rows_named(race_engine, "notes")) == 1


def test_an_allocated_create_cannot_take_the_name_an_adoption_holds(
    race_engine, tmp_path
):
    """The pair the argument-keyed guard missed.

    `mkdir` settles one allocated create against another, and nothing at all
    against an adoption: a chosen folder is refused inside the projects root,
    so `<root>/notes` is still free while the adoption holds the name.
    """
    adopting = _svc(race_engine)
    allocating = _svc(race_engine)
    parked, release = _park_inside_the_lock(adopting)

    winner = _Ran(
        lambda: adopting.create_project("notes", path=_folder(tmp_path, "chosen"))
    )
    assert parked.wait(timeout=10), "the adoption never settled its name"

    loser = _Ran(lambda: allocating.create_project("notes"))
    finished_while_held = loser.returned.wait(timeout=1)
    release.set()
    winner.settled()
    allocated = loser.settled()
    assert not finished_while_held, "the allocated create ran while the adoption held the name"
    # Bumped, not refused: an allocated create owns its directory, so taking
    # the next free name is a real outcome rather than a lost folder.
    assert allocated.name == "notes-2"
    assert len(_rows_named(race_engine, "notes")) == 1


def test_a_rename_cannot_take_the_name_an_adoption_holds(race_engine, tmp_path):
    """A local rename resolves a name through the same unguarded read."""
    adopting = _svc(race_engine)
    renaming = _svc(race_engine)
    existing = renaming.create_project("scratch")
    parked, release = _park_inside_the_lock(adopting)

    winner = _Ran(
        lambda: adopting.create_project("notes", path=_folder(tmp_path, "chosen"))
    )
    assert parked.wait(timeout=10), "the adoption never settled its name"

    loser = _Ran(lambda: renaming.update_project(existing.id, name="notes"))
    finished_while_held = loser.returned.wait(timeout=1)
    release.set()
    winner.settled()
    renamed = loser.settled()
    assert not finished_while_held, "the rename ran while the adoption held the name"
    assert renamed.name == "notes-2"
    assert len(_rows_named(race_engine, "notes")) == 1


def test_an_is_active_toggle_does_not_queue_behind_a_create(race_engine, tmp_path):
    """No name is allocated, so nothing should serialise."""
    adopting = _svc(race_engine)
    toggling = _svc(race_engine)
    existing = toggling.create_project("scratch")
    parked, release = _park_inside_the_lock(adopting)

    winner = _Ran(
        lambda: adopting.create_project("notes", path=_folder(tmp_path, "chosen"))
    )
    assert parked.wait(timeout=10), "the adoption never settled its name"

    toggle = _Ran(lambda: toggling.update_project(existing.id, is_active=True))
    assert toggle.returned.wait(timeout=10), "the toggle blocked on the name lock"
    # Not just "returned": `_Ran` records a raise as a return, so the outcome
    # has to be asserted or a toggle that fails outright reads as a pass.
    assert toggle.settled().is_active is True
    release.set()
    winner.settled()


def test_a_name_that_cannot_be_locked_in_time_is_a_retryable_failure(
    race_engine, tmp_path, monkeypatch
):
    """The locked region stats chosen paths, and a dead mount blocks there.

    Bounded so one hung stat cannot wedge every create and rename in the
    process. Not a `ValueError`: the endpoints map that to 400, and nothing
    about the request is wrong.
    """
    monkeypatch.setattr(projects_module, "_NAME_LOCK_TIMEOUT_SECONDS", 0.05)
    holding = _svc(race_engine)
    waiting = _svc(race_engine)
    parked, release = _park_inside_the_lock(holding)

    winner = _Ran(lambda: holding.create_project("notes", path=_folder(tmp_path, "a")))
    assert parked.wait(timeout=10), "the first adoption never settled its name"

    blocked = _Ran(lambda: waiting.create_project("other", path=_folder(tmp_path, "b")))
    error = blocked.raised()
    assert isinstance(error, ProjectNameLockBusyError)
    assert not isinstance(error, ValueError)
    release.set()
    winner.settled()
    assert len(_rows_named(race_engine, "other")) == 0


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


def _client(base_url: str = "http://127.0.0.1:26866", peer: str = "127.0.0.1"):
    """`base_url` sets `scope["server"]`, the accepted socket's local address,
    which is what the chosen-folder gate reads. `client` sets the peer, which
    `require_local` reads."""
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    return TestClient(create_app(), base_url=base_url, client=(peer, 54321))


def _loopback_client():
    return _client()


def test_a_missing_folder_is_a_client_error_not_a_crash(projects_root, tmp_path):
    """The create endpoint carried no `ValueError` mapping, so every rejection
    added here would otherwise reach the client as a 500."""
    res = _loopback_client().post(
        "/api/v1/projects/",
        json={"name": "notes", "path": str(tmp_path / "absent")},
    )
    assert res.status_code == 400, res.text


def test_a_busy_name_lock_is_a_503_not_a_crash(projects_root, tmp_path, monkeypatch):
    """Sibling of the test above, same failure shape.

    `ProjectNameLockBusyError` is deliberately not a `ValueError`, so the
    endpoint's 400 mapping does not catch it and it needs its own. Without one
    a bounded acquire reaches the client as a 500, which reads as a bug in the
    request rather than as "retry".
    """
    monkeypatch.setattr(projects_module, "_NAME_LOCK_TIMEOUT_SECONDS", 0.05)
    assert projects_module._LOCAL_NAME_LOCK.acquire(timeout=5)
    try:
        res = _loopback_client().post(
            "/api/v1/projects/",
            json={"name": "notes", "path": str(_folder(tmp_path, "chosen"))},
        )
    finally:
        projects_module._LOCAL_NAME_LOCK.release()
    assert res.status_code == 503, res.text


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
    folder = _folder(tmp_path, "refused-remotely")
    res = _client(peer="203.0.113.7").post(
        "/api/v1/projects/", json={"name": "refused-remotely", "path": str(folder)}
    )
    assert res.status_code == 403, res.text


def test_a_request_that_did_not_arrive_over_loopback_is_refused(
    projects_root, tmp_path
):
    """The peer address is forgeable: the image runs uvicorn with
    `--forwarded-allow-ips "*"`, so X-Forwarded-For rewrites request.client.
    The socket the request actually landed on is not, and in the container
    that socket is the published one, not loopback."""
    folder = _folder(tmp_path, "arrived-off-loopback")
    forging = _client(base_url="http://172.17.0.2:9010", peer="127.0.0.1")
    res = forging.post(
        "/api/v1/projects/",
        json={"name": "arrived-off-loopback", "path": str(folder)},
        headers={"Host": "localhost"},
    )
    assert res.status_code == 403, res.text


def test_the_configured_host_setting_does_not_decide_it(
    projects_root, tmp_path, monkeypatch
):
    """The image's CMD passes `--host 0.0.0.0` on argv and sets no
    COWORK_SERVER_HOST, so the setting reads its loopback default inside a
    container that is published to the world. It must not be the gate."""
    monkeypatch.setenv("COWORK_SERVER_HOST", "127.0.0.1")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    folder = _folder(tmp_path, "setting-says-loopback")
    res = _client(base_url="http://172.17.0.2:9010").post(
        "/api/v1/projects/",
        json={"name": "setting-says-loopback", "path": str(folder)},
        headers={"Host": "localhost"},
    )
    assert res.status_code == 403, res.text


def test_a_non_loopback_caller_can_still_create_a_normal_project(projects_root):
    """The gate is on the chosen folder, not on project creation."""
    res = _client(peer="203.0.113.7").post(
        "/api/v1/projects/", json={"name": "remote-no-folder"}
    )
    assert res.status_code == 201, res.text
