"""The project file listing is bounded.

A project directory Cowork allocated holds the agent's own output. Once a
project can point at a folder the user chose, the same listing can be asked to
walk a repository or a home directory, and it used to materialise every path
beneath it with a stat and a resolve each before returning anything.

Two properties matter beyond boundedness. The walk must treat symlinks the way
the recursive glob it replaced did, because every desktop project carries
`skills/<slug>` directory symlinks. And the cap must count files the caller may
actually see, or a subtree they may not spends the whole listing.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import cowork.api.v1.endpoints.project_files as pf
from cowork.api.v1.endpoints.project_files import (
    _MAX_EXAMINED_ENTRIES,
    _MAX_LISTED_FILES,
    _iter_project_files,
    _WalkBudget,
)
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectService,
)


def _names(base: Path) -> list[str]:
    return sorted(
        p.relative_to(base).as_posix()
        for p in _iter_project_files(base, _WalkBudget())
    )


# -- what the walk yields ----------------------------------------------------


def test_a_small_tree_is_returned_whole(tmp_path):
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("y")

    assert _names(tmp_path) == ["notes.md", "sub/deep.txt"]


def test_directories_are_not_yielded_as_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("y")

    assert _names(tmp_path) == ["sub/a.txt"]


def test_it_matches_the_recursive_glob_it_replaced(tmp_path):
    """The old listing was `sorted(base.rglob("*"))` with directories skipped
    by the caller. For a tree with no symlinks the two must agree exactly."""
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "b.txt").write_text("y")
    (tmp_path / "d" / "e" / "c.txt").write_text("z")

    glob_result = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if not p.is_dir()
    )
    assert _names(tmp_path) == glob_result


# -- symlinks ----------------------------------------------------------------


def test_a_symlinked_directory_is_neither_listed_nor_descended(tmp_path):
    """`rglob` yielded the link and the caller skipped it as a directory, so
    nothing under it was ever listed. Descending is what would spend the
    budget on a project's own `skills/<slug>` links."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "theirs.txt").write_text("not mine")
    base = tmp_path / "base"
    base.mkdir()
    (base / "mine.txt").write_text("mine")
    (base / "skills").mkdir()
    (base / "skills" / "a-skill").symlink_to(outside, target_is_directory=True)

    assert _names(base) == ["mine.txt"]


def test_a_skills_symlink_does_not_consume_the_budget(tmp_path):
    """The regression this guards: a desktop project with a linked skill tree
    returned nothing, because every walked entry resolved outside `base` and
    was discarded after the cap had been paid."""
    linked = tmp_path / "canonical-skill"
    linked.mkdir()
    for i in range(_MAX_LISTED_FILES + 10):
        (linked / f"s{i:05d}.md").write_text("x")
    base = tmp_path / "base"
    (base / "skills").mkdir(parents=True)
    (base / "skills" / "a-skill").symlink_to(linked, target_is_directory=True)
    (base / "report.md").write_text("mine")

    assert _names(base) == ["report.md"]


def test_a_symlink_loop_terminates(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "real.txt").write_text("x")
    (inner / "loop").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "top.txt").write_text("y")

    found = _names(tmp_path)

    assert found == ["inner/real.txt", "top.txt"]
    assert len(found) == len(set(found))


# -- the bounds --------------------------------------------------------------


def test_the_examined_ceiling_stops_the_walk_and_says_so(tmp_path, monkeypatch):
    """A stopped walk must not report a complete listing. Only the file cap
    used to set the flag, so hitting this ceiling looked like "that is all
    the files there are"."""
    monkeypatch.setattr(pf, "_MAX_EXAMINED_ENTRIES", 10)
    for i in range(40):
        (tmp_path / f"f{i:03d}.txt").write_text("x")

    budget = _WalkBudget()
    found = list(_iter_project_files(tmp_path, budget))

    assert budget.exhausted is True
    assert len(found) <= 10


def test_a_completed_walk_does_not_report_exhaustion(tmp_path):
    (tmp_path / "a.txt").write_text("x")

    budget = _WalkBudget()
    list(_iter_project_files(tmp_path, budget))

    assert budget.exhausted is False


def test_an_unreadable_directory_is_skipped_not_fatal(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.txt").write_text("y")
    blocked.chmod(0o000)
    try:
        found = _names(tmp_path)
    finally:
        blocked.chmod(0o755)

    assert "ok.txt" in found


# -- the endpoint ------------------------------------------------------------


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


def _list_files(engine, project_name: str) -> dict:
    from cowork.api.v1.endpoints.project_files import list_project_files

    return list_project_files(
        project_name, ScopedSession(Session(engine), LOCAL_SCOPE), None
    )


def test_an_untruncated_response_carries_no_truncated_key(engine, projects_root):
    svc = ProjectService(ScopedSession(Session(engine), LOCAL_SCOPE))
    project = svc.create_project("notes")
    (Path(project.path) / "a.md").write_text("x")

    response = _list_files(engine, project.name)

    assert "truncated" not in response
    assert "a.md" in {f["path"] for f in response["files"]}


def test_a_truncated_response_says_so(engine, projects_root):
    svc = ProjectService(ScopedSession(Session(engine), LOCAL_SCOPE))
    project = svc.create_project("notes")
    for i in range(_MAX_LISTED_FILES + 5):
        (Path(project.path) / f"f{i:05d}.txt").write_text("x")

    response = _list_files(engine, project.name)

    assert response["truncated"] is True
    # The synthetic instructions row is inserted after the walk, so it is not
    # one of the capped entries.
    assert len(response["files"]) == _MAX_LISTED_FILES + 1
