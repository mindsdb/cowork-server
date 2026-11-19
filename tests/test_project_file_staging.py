"""The two halves of a project file write: a fresh temporary entry, then its bytes.

An import holds many temporaries before any of them is moved into place, so
creating one and filling it are separate from replacing the target.
"""

from __future__ import annotations

import os
import stat

import pytest

from cowork.api.v1.endpoints import project_files


def test_a_temporary_entry_is_new_empty_and_named_with_the_prefix(tmp_path):
    with project_files.pinned_dir(tmp_path) as root:
        name, fd = project_files._create_project_temp(root, ".cowork-import-", 0o640)
        try:
            project_files._write_all(fd, b"bytes")
        finally:
            os.close(fd)

    assert name.startswith(".cowork-import-") and name.endswith(".tmp")
    assert (tmp_path / name).read_bytes() == b"bytes"
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o640 & ~_umask()


def test_two_temporary_entries_never_share_a_name(tmp_path):
    with project_files.pinned_dir(tmp_path) as root:
        first, fd_a = project_files._create_project_temp(root, ".cowork-import-", 0o644)
        second, fd_b = project_files._create_project_temp(root, ".cowork-import-", 0o644)
        os.close(fd_a)
        os.close(fd_b)

    assert first != second
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([first, second])


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_a_temporary_entry_is_never_opened_through_a_planted_link(tmp_path, monkeypatch):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    planted = ".cowork-import-" + "0" * 24 + ".tmp"
    (tmp_path / planted).symlink_to(victim)
    monkeypatch.setattr(project_files.secrets, "token_hex", lambda _n: "0" * 24)

    with project_files.pinned_dir(tmp_path) as root, pytest.raises(FileExistsError):
        project_files._create_project_temp(root, ".cowork-import-", 0o644)

    assert victim.read_text() == "keep"


def test_a_write_that_makes_no_progress_is_an_error(tmp_path, monkeypatch):
    with project_files.pinned_dir(tmp_path) as root:
        _name, fd = project_files._create_project_temp(root, ".cowork-import-", 0o644)
        monkeypatch.setattr(project_files.os, "write", lambda *_args: 0)
        try:
            with pytest.raises(OSError, match="no progress"):
                project_files._write_all(fd, b"bytes")
        finally:
            monkeypatch.undo()
            os.close(fd)


def test_created_lists_only_the_folders_this_call_made(tmp_path):
    (tmp_path / "a").mkdir()
    created: list[tuple[str, ...]] = []

    with project_files.pinned_dir(tmp_path) as root:
        with project_files._opened_pinned_descendant(root, ("a", "b", "c"), create=True, created=created):
            pass
        again: list[tuple[str, ...]] = []
        with project_files._opened_pinned_descendant(root, ("a", "b", "c"), create=True, created=again):
            pass

    assert created == [("a", "b"), ("a", "b", "c")]
    assert again == []
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_descending_without_create_records_nothing(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    created: list[tuple[str, ...]] = []

    with project_files.pinned_dir(tmp_path) as root:
        with project_files._opened_pinned_descendant(root, ("a", "b"), create=False, created=created) as pinned:
            assert pinned.path == tmp_path / "a" / "b"

    assert created == []


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
