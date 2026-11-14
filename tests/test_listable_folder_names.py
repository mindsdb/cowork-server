"""The folders of an artifacts root the artifacts list would open
(held_artifact_roots.listable_folder_names), which a comparison side's carry
holds and times, and whether a root holds one (holds_a_folder), which the
held-root backfill asks and which stops at the first folder. A root the list
could show nothing from holds none; any other read error raises, so a root
the server could not read is never taken for an empty one.
"""
from __future__ import annotations

import errno
import os
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from cowork.services import held_artifact_roots
from cowork.services.held_artifact_roots import holds_a_folder, listable_folder_names
from test_artifact_ownership import make_project, project_root, write_artifact

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def _entry(name: str, *, error: int | None = None) -> SimpleNamespace:
    """A folder entry of a scandir, whose type read fails with `error`."""

    def is_dir(*, follow_symlinks):
        if error is not None:
            raise OSError(error, os.strerror(error))
        return True

    return SimpleNamespace(name=name, is_symlink=lambda: False, is_dir=is_dir)


def _scandir_yields(monkeypatch, *entries: SimpleNamespace) -> None:
    @contextmanager
    def scandir(_root):
        yield iter(entries)

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", scandir)


def test_the_names_are_the_folders_the_list_would_open(tmp_path):
    """A folder with no metadata.json is among them: the caller reads what
    each holds. A dot-named folder, a file and a link are not."""
    source = project_root(make_project(tmp_path, None))
    write_artifact(source.base, "chart")
    (source.base / "no-metadata").mkdir()
    (source.base / ".hidden").mkdir()
    (source.base / "loose.txt").write_text("")
    os.symlink(source.base / "chart", source.base / "linked")

    assert sorted(listable_folder_names(source)) == ["chart", "no-metadata"]
    assert holds_a_folder(source) is True


def test_a_missing_root_and_a_linked_root_hold_none(tmp_path):
    missing = project_root(make_project(tmp_path, None))
    shutil.rmtree(missing.base)
    linked = project_root(make_project(tmp_path, None))
    elsewhere = tmp_path / "elsewhere"
    write_artifact(elsewhere, "chart")
    shutil.rmtree(linked.base)
    os.symlink(elsewhere, linked.base)

    for source in (missing, linked):
        assert listable_folder_names(source) == []
        assert holds_a_folder(source) is False


@pytest.mark.parametrize(("error", "raises"), [(errno.EACCES, False), (errno.EIO, True)], ids=["may-not-read", "eio"])
def test_a_root_the_server_may_not_read_holds_none_and_any_other_error_raises(
    tmp_path, monkeypatch, error, raises
):
    source = project_root(make_project(tmp_path, None))
    write_artifact(source.base, "chart")

    @contextmanager
    def scandir(_root):
        raise OSError(error, os.strerror(error))
        yield

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", scandir)

    if raises:
        with pytest.raises(OSError):
            listable_folder_names(source)
        with pytest.raises(OSError):
            holds_a_folder(source)
        return
    assert listable_folder_names(source) == []
    assert holds_a_folder(source) is False


def test_an_entry_whose_type_cannot_be_read_raises_a_listing_but_not_after_a_folder(tmp_path, monkeypatch):
    """listable_folder_names reads every entry, so an EIO on any of them
    raises: a list of part of the root is never taken for all of it.
    holds_a_folder stops at the first folder, so an EIO on a later entry is
    never read and the held-root backfill holds the root; an EIO before any
    folder still raises."""
    source = project_root(make_project(tmp_path, None))
    _scandir_yields(monkeypatch, _entry("chart"), _entry("stalled", error=errno.EIO))

    with pytest.raises(OSError):
        listable_folder_names(source)
    assert holds_a_folder(source) is True

    _scandir_yields(monkeypatch, _entry("stalled", error=errno.EIO), _entry("chart"))

    with pytest.raises(OSError):
        holds_a_folder(source)


def test_an_entry_the_server_may_not_read_is_left_out(tmp_path, monkeypatch):
    source = project_root(make_project(tmp_path, None))
    _scandir_yields(monkeypatch, _entry("chart"), _entry("private", error=errno.EACCES))

    assert listable_folder_names(source) == ["chart"]
    assert holds_a_folder(source) is True
