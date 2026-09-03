"""The projects root cannot be redirected between check and use.

_project_path proves containment, but the proof expires when it returns: the
caller's mkdir/rename/rmtree makes the kernel walk the path again. The agent's
pod mounts its own org subtree read-write, so it can replace `projects` with a
symlink into another org in that window. These tests perform exactly that swap
and assert the operation still lands in the right org, or refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.paths import dir_rmdir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
import cowork.services.projects as projects_module
from cowork.services.projects import ProjectService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


@pytest.fixture()
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture()
def svc(shared):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    scope = TenantScope(org_mode=True, org_id=ORG_A, user_id="u")
    return ProjectService(ScopedSession(Session(engine), scope))


def _swap_root_for_symlink_to(root: Path, victim: Path) -> None:
    """What the agent does in the race window: replace its own `projects`
    directory with a symlink pointing into another organization."""
    victim.mkdir(parents=True, exist_ok=True)
    if root.exists():
        for child in root.iterdir():
            child.rmdir() if child.is_dir() else child.unlink()
        root.rmdir()
    root.symlink_to(victim, target_is_directory=True)


def test_mkdir_refuses_a_root_swapped_for_a_symlink(svc, shared):
    root = shared / ORG_A / "projects"
    victim = shared / ORG_B / "projects"
    root.mkdir(parents=True, exist_ok=True)
    path = svc._project_path("mine")  # checked while root is real
    _swap_root_for_symlink_to(root, victim)  # ... then swapped

    with pytest.raises(OSError):  # O_NOFOLLOW on the root open
        svc._mkdir_in_root(path)
    assert not (victim / "mine").exists(), "must not create in the other org"


def test_rmtree_refuses_a_root_swapped_for_a_symlink(svc, shared):
    root = shared / ORG_A / "projects"
    victim = shared / ORG_B / "projects"
    root.mkdir(parents=True, exist_ok=True)
    (victim / "theirs").mkdir(parents=True)
    (victim / "theirs" / "keep.txt").write_text("other org's data")
    path = svc._project_path("theirs")
    _swap_root_for_symlink_to(root, victim)

    with pytest.raises(OSError):
        svc._rmtree_in_root(path)
    assert (
        victim / "theirs" / "keep.txt"
    ).exists(), "must not delete another org's data"


def test_rename_keeps_source_pinned_when_root_is_swapped(
    svc,
    shared,
    monkeypatch,
):
    root = shared / ORG_A / "projects"
    detached = shared / ORG_A / "detached-projects"
    victim = shared / ORG_B / "projects"
    (root / "mine").mkdir(parents=True)
    (root / "mine" / "owner.txt").write_text("org a")
    (victim / "mine").mkdir(parents=True)
    (victim / "mine" / "owner.txt").write_text("org b")
    original_rename = projects_module.dir_rename

    def swap_before_rename(source, source_name, destination, destination_name):
        root.rename(detached)
        root.symlink_to(victim, target_is_directory=True)
        original_rename(source, source_name, destination, destination_name)

    monkeypatch.setattr(projects_module, "dir_rename", swap_before_rename)

    svc._rename_in_root(
        svc._project_path("mine"),
        svc._project_path("renamed"),
    )

    assert (detached / "renamed" / "owner.txt").read_text() == "org a"
    assert (victim / "mine" / "owner.txt").read_text() == "org b"


def test_rename_between_legacy_and_scoped_roots(svc):
    legacy = Path(get_app_settings().project.root_dir) / "legacy"
    scoped = svc._project_path("renamed")
    legacy.mkdir(parents=True)
    (legacy / "keep.txt").write_text("legacy content")

    svc._rename_in_root(legacy, scoped)
    assert (scoped / "keep.txt").read_text() == "legacy content"

    svc._rename_in_root(scoped, legacy)
    assert (legacy / "keep.txt").read_text() == "legacy content"


def test_rename_helper_rejects_nested_source_and_destination_names(tmp_path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "source").mkdir()
    (root / "source").mkdir()

    with projects_module.pinned_dir(root) as pinned:
        with pytest.raises(ValueError, match="direct-child names"):
            projects_module.dir_rename(
                pinned,
                "nested/source",
                pinned,
                "moved",
            )
        with pytest.raises(ValueError, match="direct-child names"):
            projects_module.dir_rename(
                pinned,
                "source",
                pinned,
                "nested/moved",
            )

    assert (root / "nested" / "source").is_dir()
    assert (root / "source").is_dir()


def test_rmtree_helper_rejects_a_nested_name(tmp_path):
    root = tmp_path / "root"
    target = root / "nested" / "target"
    target.mkdir(parents=True)

    with projects_module.pinned_dir(root) as pinned:
        with pytest.raises(ValueError, match="direct-child name"):
            projects_module.dir_rmtree(pinned, "nested/target")

    assert target.is_dir()


def test_rmdir_helper_rejects_a_nested_name(tmp_path):
    root = tmp_path / "root"
    target = root / "nested" / "target"
    target.mkdir(parents=True)

    with projects_module.pinned_dir(root) as pinned:
        with pytest.raises(ValueError, match="direct-child name"):
            dir_rmdir(pinned, "nested/target")

    assert target.is_dir()


def test_rmdir_helper_removes_a_direct_child(tmp_path):
    root = tmp_path / "root"
    target = root / "target"
    target.mkdir(parents=True)

    with projects_module.pinned_dir(root) as pinned:
        dir_rmdir(pinned, "target")

    assert not target.exists()


def test_nested_path_is_refused_even_below_a_real_root(svc, shared):
    """_child_name is the second half of the guarantee: dir_fd only resolves
    one component safely, so a nested path must never reach it."""
    root = shared / ORG_A / "projects"
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="direct child"):
        svc._mkdir_in_root(root / "a" / "b")


def test_normal_operations_still_work(svc, shared):
    path = svc._project_path("ordinary")
    svc._mkdir_in_root(path)
    assert path.is_dir()
    svc._rmtree_in_root(path)
    assert not path.exists()
