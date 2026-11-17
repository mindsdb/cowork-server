"""Writing an import into a project: every file lands, or the project is left as it was.

These run on a real temporary folder. Failures are injected at the filesystem
helpers the import engine calls, never by mocking the engine itself.
"""

from __future__ import annotations

import os
import time

import pytest

from cowork.api.v1.endpoints import project_files
from cowork.api.v1.endpoints.project_files import (
    _ImportConflict,
    _ImportDiskError,
    _ImportFileInUse,
    _ImportRestoreFailed,
    _ProjectImport,
)


def _tree(base) -> dict[str, bytes]:
    """Every file below ``base`` with its bytes, and every folder as ``b"<dir>"``."""
    found: dict[str, bytes] = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        found[relative] = b"<dir>" if path.is_dir() and not path.is_symlink() else path.read_bytes()
    return found


def _feed(data: bytes):
    def push(sink):
        sink(data)

    return push


def _import(base, files: dict[str, bytes], **options) -> None:
    targets = [tuple(path.split("/")) for path in files]
    with project_files.pinned_dir(base) as root:
        with _ProjectImport(root, targets, **options) as writer:
            writer.check()
            for index, data in enumerate(files.values()):
                writer.stage(index, _feed(data))
            writer.commit()


def _case_insensitive(base) -> bool:
    probe = base / "CaseProbe"
    probe.write_text("x")
    try:
        return (base / "caseprobe").exists()
    finally:
        probe.unlink()


def test_files_and_nested_folders_land_with_their_bytes(tmp_path):
    _import(tmp_path, {"a.txt": b"a", "Reports/q3.xlsx": b"xlsx", "Reports/Old/q2.docx": b"docx"})

    assert _tree(tmp_path) == {
        "Reports": b"<dir>",
        "Reports/Old": b"<dir>",
        "Reports/Old/q2.docx": b"docx",
        "Reports/q3.xlsx": b"xlsx",
        "a.txt": b"a",
    }


def test_reimport_replaces_and_leaves_no_temporary_or_backup(tmp_path):
    _import(tmp_path, {"Reports/q3.xlsx": b"old", "keep.txt": b"keep"})

    _import(tmp_path, {"Reports/q3.xlsx": b"new"})

    assert _tree(tmp_path) == {"Reports": b"<dir>", "Reports/q3.xlsx": b"new", "keep.txt": b"keep"}


def test_a_staging_failure_leaves_the_project_as_it_was(tmp_path):
    (tmp_path / "two.txt").write_bytes(b"old two")
    before = _tree(tmp_path)
    targets = [("one.txt",), ("two.txt",), ("New", "three.txt"), ("New", "four.txt"), ("five.txt",)]

    def broken(sink):
        sink(b"partial")
        raise OSError(28, "No space left on device")

    with project_files.pinned_dir(tmp_path) as root, pytest.raises(_ImportDiskError) as raised:
        with _ProjectImport(root, targets) as writer:
            writer.check()
            writer.stage(0, _feed(b"1"))
            writer.stage(1, _feed(b"2"))
            writer.stage(2, broken)

    assert raised.value.path == "New/three.txt"
    assert _tree(tmp_path) == before


def test_an_error_from_the_feed_itself_propagates_and_rolls_back(tmp_path):
    class Refused(Exception):
        pass

    def refuse(_sink):
        raise Refused()

    with project_files.pinned_dir(tmp_path) as root, pytest.raises(Refused):
        with _ProjectImport(root, [("Sub", "a.txt")]) as writer:
            writer.check()
            writer.stage(0, refuse)

    assert _tree(tmp_path) == {}


def test_a_move_failing_mid_commit_puts_every_file_back(tmp_path, monkeypatch):
    for name in ("one.txt", "two.txt"):
        (tmp_path / name).write_bytes(f"old {name}".encode())
    before = _tree(tmp_path)
    real_replace = project_files.dir_replace
    moves: list[tuple[str, str]] = []

    def failing_replace(parent, source, destination):
        moves.append((source, destination))
        if destination == "two.txt" and source.endswith(".tmp"):
            raise OSError(5, "Input/output error")
        real_replace(parent, source, destination)

    monkeypatch.setattr(project_files, "dir_replace", failing_replace)
    files = {"one.txt": b"1", "two.txt": b"2", "New/three.txt": b"3", "four.txt": b"4", "five.txt": b"5"}

    with pytest.raises(_ImportDiskError) as raised:
        _import(tmp_path, files)

    assert raised.value.path == "two.txt"
    assert _tree(tmp_path) == before


def test_a_restore_that_fails_keeps_the_old_copy_and_names_it(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_bytes(b"old one")
    real_replace = project_files.dir_replace

    def failing_replace(parent, source, destination):
        if destination == "two.txt" and source.endswith(".tmp"):
            raise OSError(5, "Input/output error")
        if destination == "one.txt" and source.endswith(".bak"):
            raise OSError(5, "Input/output error")
        real_replace(parent, source, destination)

    monkeypatch.setattr(project_files, "dir_replace", failing_replace)

    with pytest.raises(_ImportRestoreFailed) as raised:
        _import(tmp_path, {"one.txt": b"new one", "two.txt": b"2"})

    kept = raised.value.kept
    assert raised.value.path == "one.txt"
    assert kept is not None and kept.startswith(".cowork-import-") and kept.endswith(".bak")
    assert kept in str(raised.value)
    assert (tmp_path / kept).read_bytes() == b"old one"
    assert not (tmp_path / "two.txt").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_a_new_file_that_cannot_be_removed_on_rollback_is_named(tmp_path, monkeypatch):
    real_replace = project_files.dir_replace
    real_unlink = project_files.dir_unlink

    def failing_replace(parent, source, destination):
        if destination == "two.txt":
            raise OSError(5, "Input/output error")
        real_replace(parent, source, destination)

    def failing_unlink(parent, name):
        if name == "one.txt":
            raise OSError(5, "Input/output error")
        real_unlink(parent, name)

    monkeypatch.setattr(project_files, "dir_replace", failing_replace)
    monkeypatch.setattr(project_files, "dir_unlink", failing_unlink)

    with pytest.raises(_ImportRestoreFailed) as raised:
        _import(tmp_path, {"one.txt": b"1", "two.txt": b"2"})

    assert raised.value.path == "one.txt"
    assert raised.value.kept is None


def test_a_case_variant_of_an_existing_file_is_never_treated_as_its_old_copy(tmp_path):
    (tmp_path / "Report.xlsx").write_bytes(b"theirs")

    if _case_insensitive(tmp_path):
        with pytest.raises(_ImportConflict) as raised:
            _import(tmp_path, {"report.xlsx": b"mine"})
        assert "differs only in case" in str(raised.value)
        assert _tree(tmp_path) == {"Report.xlsx": b"theirs"}
    else:
        _import(tmp_path, {"report.xlsx": b"mine"})
        assert _tree(tmp_path) == {"Report.xlsx": b"theirs", "report.xlsx": b"mine"}


def test_a_name_that_only_casefolds_alike_never_replaces_the_existing_file(tmp_path):
    (tmp_path / "Straße.docx").write_bytes(b"theirs")
    folds_onto_it = (tmp_path / "Strasse.docx").exists()

    if folds_onto_it:
        with pytest.raises(_ImportConflict):
            _import(tmp_path, {"Strasse.docx": b"mine"})
        assert _tree(tmp_path) == {"Straße.docx": b"theirs"}
    else:
        _import(tmp_path, {"Strasse.docx": b"mine"})
        assert _tree(tmp_path) == {"Straße.docx": b"theirs", "Strasse.docx": b"mine"}


@pytest.mark.skipif(os.name == "nt", reason="hard links on the desktop are a POSIX concern")
def test_reimporting_a_hardlinked_file_replaces_only_that_name(tmp_path):
    (tmp_path / "report.xlsx").write_bytes(b"old")
    os.link(tmp_path / "report.xlsx", tmp_path / "copy.xlsx")

    _import(tmp_path, {"report.xlsx": b"new"})

    assert (tmp_path / "report.xlsx").read_bytes() == b"new"
    assert (tmp_path / "copy.xlsx").read_bytes() == b"old"


@pytest.mark.parametrize(
    ("files", "existing", "fragment"),
    [
        ({"a.txt": b"1", "A.TXT": b"2"}, None, "would both be saved as"),
        ({"Café.txt": b"1", "Café.txt": b"2"}, None, "would both be saved as"),
        ({"Report": b"1", "report/x.txt": b"2"}, None, "both a file and a folder"),
        ({"Reports/a.txt": b"1", "reports/b.txt": b"2"}, None, "would be one folder"),
        ({"Caf\u00e9/a.txt": b"1", "Cafe\u0301/b.txt": b"2"}, None, "would be one folder"),
        ({"Docs": b"1"}, "folder", "is a folder"),
        ({"Docs/a.txt": b"1"}, "file", "needs a folder there"),
    ],
)
def test_conflicts_are_refused_before_anything_is_written(tmp_path, files, existing, fragment):
    if existing == "folder":
        (tmp_path / "Docs").mkdir()
    elif existing == "file":
        (tmp_path / "Docs").write_bytes(b"a file")
    before = _tree(tmp_path)

    with pytest.raises(_ImportConflict) as raised:
        _import(tmp_path, files)

    assert fragment in str(raised.value)
    assert _tree(tmp_path) == before


def test_a_target_more_than_32_folders_deep_is_refused(tmp_path):
    deep = "/".join(["d"] * 32) + "/f.txt"

    with pytest.raises(_ImportConflict) as raised:
        _import(tmp_path, {deep: b"x"})

    assert "32 folders deep" in str(raised.value)
    assert _tree(tmp_path) == {}


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_a_link_anywhere_on_the_path_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "project"
    base.mkdir()
    (base / "Linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(_ImportConflict) as raised:
        _import(base, {"Linked/a.txt": b"x"})

    assert "is a link" in str(raised.value)
    assert list(outside.iterdir()) == []


def test_created_folders_are_removed_only_when_empty_deepest_first(tmp_path, monkeypatch):
    real_replace = project_files.dir_replace

    def failing_replace(parent, source, destination):
        if destination == "b.txt":
            raise OSError(5, "Input/output error")
        real_replace(parent, source, destination)

    monkeypatch.setattr(project_files, "dir_replace", failing_replace)
    (tmp_path / "Keep").mkdir()

    with pytest.raises(_ImportDiskError):
        _import(tmp_path, {"New/Deeper/a.txt": b"a", "Keep/Made/b.txt": b"b"})

    assert _tree(tmp_path) == {"Keep": b"<dir>"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_reimported_file_keeps_its_old_mode(tmp_path):
    (tmp_path / "run.sh").write_bytes(b"old")
    os.chmod(tmp_path / "run.sh", 0o750)

    _import(tmp_path, {"run.sh": b"new"})

    assert (tmp_path / "run.sh").read_bytes() == b"new"
    assert (tmp_path / "run.sh").stat().st_mode & 0o777 == 0o750


def test_a_posix_permission_error_is_a_disk_error_not_a_file_in_use(tmp_path, monkeypatch):
    def refuse(parent, source, destination):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(project_files, "dir_replace", refuse)

    with pytest.raises(_ImportDiskError):
        _import(tmp_path, {"a.txt": b"x"}, windows=False)


def test_on_windows_a_held_file_is_in_use_after_brief_retries(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_bytes(b"old")
    sleeps: list[float] = []
    real_replace = project_files.dir_replace

    def held(parent, source, destination):
        if destination == "a.txt" and source.endswith(".tmp"):
            raise PermissionError(13, "The process cannot access the file")
        real_replace(parent, source, destination)

    monkeypatch.setattr(project_files, "dir_replace", held)

    with pytest.raises(_ImportFileInUse) as raised:
        _import(tmp_path, {"a.txt": b"new"}, windows=True, sleep=sleeps.append)

    assert raised.value.path == "a.txt"
    assert len(sleeps) == 2
    assert _tree(tmp_path) == {"a.txt": b"old"}


def test_on_windows_a_briefly_held_temporary_is_moved_after_a_retry(tmp_path, monkeypatch):
    real_replace = project_files.dir_replace
    refusals = iter([True])

    def flaky(parent, source, destination):
        if source.endswith(".tmp") and next(refusals, False):
            raise PermissionError(13, "held by a scanner")
        real_replace(parent, source, destination)

    monkeypatch.setattr(project_files, "dir_replace", flaky)

    _import(tmp_path, {"a.txt": b"new"}, windows=True, sleep=lambda _s: None)

    assert _tree(tmp_path) == {"a.txt": b"new"}


def test_on_windows_a_path_too_long_is_refused_before_anything_is_written(tmp_path):
    long_name = "n" * 200 + ".txt"

    with pytest.raises(_ImportConflict) as raised:
        _import(tmp_path, {long_name: b"x"}, windows=True)

    assert "too long" in str(raised.value)
    assert _tree(tmp_path) == {}


def test_a_stale_import_temporary_is_swept_and_a_fresh_one_kept(tmp_path):
    stale = tmp_path / ".cowork-import-stale.tmp"
    fresh = tmp_path / ".cowork-import-fresh.tmp"
    backup = tmp_path / ".cowork-import-old.bak"
    for path in (stale, fresh, backup):
        path.write_bytes(b"left over")
    two_hours_ago = time.time() - 2 * 60 * 60
    os.utime(stale, (two_hours_ago, two_hours_ago))
    os.utime(backup, (two_hours_ago, two_hours_ago))

    _import(tmp_path, {"a.txt": b"x"})

    assert not stale.exists()
    assert fresh.exists()
    assert backup.exists()


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace an open file")
def test_on_windows_a_file_open_elsewhere_refuses_the_import_and_changes_nothing(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"old a")
    (tmp_path / "b.xlsx").write_bytes(b"old b")
    before = _tree(tmp_path)

    with open(tmp_path / "b.xlsx", "rb"):
        with pytest.raises(_ImportFileInUse) as raised:
            _import(tmp_path, {"a.txt": b"new a", "b.xlsx": b"new b"}, sleep=lambda _s: None)

    assert raised.value.path == "b.xlsx"
    assert _tree(tmp_path) == before
