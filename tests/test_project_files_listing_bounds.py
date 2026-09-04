"""The project file listing is bounded.

A project directory Cowork allocated holds the agent's own output. Once a
project can point at a folder the user chose, the same listing can be asked to
walk a repository or a home directory, and it used to materialise every path
beneath it with a stat and a resolve each before returning anything.
"""
from __future__ import annotations

from pathlib import Path

from cowork.api.v1.endpoints.project_files import (
    _MAX_LISTED_FILES,
    _walk_project_files,
)


def test_a_small_tree_is_returned_whole(tmp_path):
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("y")

    found, truncated = _walk_project_files(tmp_path)

    assert not truncated
    assert {p.name for p in found} == {"notes.md", "deep.txt"}


def test_directories_are_not_listed_as_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("y")

    found, _ = _walk_project_files(tmp_path)

    assert [p.name for p in found] == ["a.txt"]


def test_the_walk_stops_at_the_cap(tmp_path):
    for i in range(_MAX_LISTED_FILES + 50):
        (tmp_path / f"f{i:05d}.txt").write_text("x")

    found, truncated = _walk_project_files(tmp_path)

    assert truncated
    assert len(found) == _MAX_LISTED_FILES


def test_a_truncated_listing_keeps_the_shallow_files(tmp_path):
    """A depth-first walk would spend the cap inside the first big directory
    and never reach the files the user actually put at the top."""
    heavy = tmp_path / "node_modules"
    heavy.mkdir()
    for i in range(_MAX_LISTED_FILES):
        (heavy / f"dep{i:05d}.js").write_text("x")
    (tmp_path / "report.md").write_text("mine")

    found, truncated = _walk_project_files(tmp_path)

    assert truncated
    assert Path(tmp_path / "report.md") in found


def test_a_symlink_loop_terminates(tmp_path):
    """A folder the user chose can contain one. The previous recursive glob
    followed it until the cap, listing the same files over and over."""
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "real.txt").write_text("x")
    (inner / "loop").symlink_to(tmp_path, target_is_directory=True)

    found, truncated = _walk_project_files(tmp_path)

    assert not truncated
    assert [p.name for p in found] == ["real.txt"]


def test_an_unreadable_directory_is_skipped_not_fatal(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "hidden.txt").write_text("y")
    blocked.chmod(0o000)
    try:
        found, truncated = _walk_project_files(tmp_path)
    finally:
        blocked.chmod(0o755)

    assert not truncated
    assert "ok.txt" in {p.name for p in found}
