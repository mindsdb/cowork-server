"""Skill distribution into a project pointed at a folder the user chose.

Skills are distributed by scanning the projects root, so an adopted folder is
invisible to it and every skill that works elsewhere silently does nothing
there. The harness creates `skills/` in the folder regardless, so the user
sees an empty directory appear and no skills in it.

Selection still never trusts a caller-supplied path — see
`test_skill_links.py::test_reconcile_project_selects_its_directory_from_inventory`.
An adopted folder's basename is not its name, so its caller passes the name.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cowork.services.skill_links as sl


@pytest.fixture()
def canonical(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    (root / "safe-skill").mkdir(parents=True)
    monkeypatch.setattr(sl, "_canon_root", lambda: root)
    return root


@pytest.fixture()
def links(monkeypatch):
    made: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        sl, "_ensure_symlink", lambda link, target: made.append((link, target))
    )
    monkeypatch.setattr(sl, "_remove_link", lambda link: None)
    return made


def _skill(projects=()):
    return SimpleNamespace(name="safe-skill", enabled=True, projects=list(projects))


def test_a_global_skill_reaches_an_adopted_folder(tmp_path, canonical, links, monkeypatch):
    adopted = tmp_path / "Documents" / "notes"
    adopted.mkdir(parents=True)
    monkeypatch.setattr(sl, "_project_dirs", lambda: [])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes-2": adopted})

    sl.reconcile_skill_links(_skill())

    assert links == [(adopted / "skills" / "safe-skill", canonical / "safe-skill")]


def test_a_targeted_skill_matches_the_row_name_not_the_folder(
    tmp_path, canonical, links, monkeypatch
):
    """`metadata.projects` holds row names. The folder is called `notes` while
    the project is `notes-2`, so matching on the folder would miss it."""
    adopted = tmp_path / "notes"
    adopted.mkdir()
    monkeypatch.setattr(sl, "_project_dirs", lambda: [])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes-2": adopted})

    sl.reconcile_skill_links(_skill(projects=["notes-2"]))

    assert links == [(adopted / "skills" / "safe-skill", canonical / "safe-skill")]


def test_reconcile_project_links_into_an_adopted_folder_by_name(
    tmp_path, canonical, links, monkeypatch
):
    adopted = tmp_path / "notes"
    adopted.mkdir()
    monkeypatch.setattr(sl, "_project_dirs", lambda: [])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes-2": adopted})

    sl.reconcile_project(adopted, [_skill()], project_name="notes-2")

    assert links == [(adopted / "skills" / "safe-skill", canonical / "safe-skill")]


def test_an_adopted_folder_never_receives_another_projects_links(
    tmp_path, canonical, links, monkeypatch
):
    """The collision this feature introduces: an adopted `~/Documents/notes`
    shares its basename with the in-root project `notes`. Selecting on the
    basename would send the in-root project's links into the user's folder."""
    in_root = tmp_path / "projects" / "notes"
    in_root.mkdir(parents=True)
    adopted = tmp_path / "Documents" / "notes"
    adopted.mkdir(parents=True)
    monkeypatch.setattr(sl, "_project_dirs", lambda: [in_root])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes-2": adopted})

    sl.reconcile_project(in_root, [_skill()], project_name="notes")

    assert links == [(in_root / "skills" / "safe-skill", canonical / "safe-skill")]


def test_a_scanned_project_keeps_precedence_over_a_colliding_name(
    tmp_path, canonical, links, monkeypatch
):
    in_root = tmp_path / "projects" / "notes"
    in_root.mkdir(parents=True)
    stray = tmp_path / "elsewhere" / "notes"
    stray.mkdir(parents=True)
    monkeypatch.setattr(sl, "_project_dirs", lambda: [in_root])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes": stray})

    assert [p.path for p in sl._all_projects()] == [in_root]


def test_remove_drops_the_link_from_an_adopted_folder(tmp_path, monkeypatch):
    adopted = tmp_path / "notes"
    adopted.mkdir()
    removed: list[Path] = []
    monkeypatch.setattr(sl, "_project_dirs", lambda: [])
    monkeypatch.setattr(sl, "_external_project_dirs", lambda: {"notes-2": adopted})
    monkeypatch.setattr(sl, "_remove_link", lambda link: removed.append(link))

    sl.remove_skill_links("safe-skill")

    assert removed == [adopted / "skills" / "safe-skill"]


def test_a_db_failure_degrades_to_the_scan(tmp_path, monkeypatch):
    """Reading adopted folders must never abort distribution: without it the
    behaviour is what it was before adopted folders existed."""
    def boom():
        raise RuntimeError("no database here")

    monkeypatch.setattr(sl, "_project_dirs", lambda: [tmp_path / "projects" / "a"])
    monkeypatch.setattr("cowork.db.session.get_open_session", boom)

    assert [p.name for p in sl._all_projects()] == ["a"]
