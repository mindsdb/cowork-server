"""A cloud turn's copy of its project's files (stage_project_files).

The pod mounts its conversation's folder and not the project, so before each
cloud turn the server copies the project's ordinary files into
``conversations/<id>/project-files/``. The pod can write that folder: the copy
never follows a link it planted, never trusts a copy it could have changed,
and keeps what it copied in a record outside the folder. A copy that can't
finish empties the folder, so the agent never reads a file the project no
longer has.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
import sys
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import cowork.services.files as files
import cowork.services.project_file_view as project_file_view
from cowork.db.scoped import TenantScope
from cowork.services.files import (
    NOT_COPIED_FILENAME,
    NotCopiedFiles,
    NotCopiedReason,
    ProjectFileCopyLimits,
    ProjectFilesOutcome,
    stage_project_files,
)

ORG = "7a1c2d3e-4f5a-4b6c-8d7e-9f0a1b2c3d81"
USER = "8b2d3e4f-5a6b-4c7d-9e8f-0a1b2c3d4e81"
SCOPE = TenantScope(org_mode=True, org_id=ORG, user_id=USER)
MEMBER = {"X-User-Id": USER, "X-Organization-Id": ORG}
LIMITS = ProjectFileCopyLimits(max_file_bytes=1000, max_files=50, max_total_bytes=10_000, seconds=30.0)

posix_only = pytest.mark.skipif(os.name == "nt", reason="org mode runs on POSIX only")
not_root = pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0, reason="root ignores folder modes")


def projects(org_id: str = ORG) -> Path:
    """The organization's projects store, where the server keeps a project's
    folder: ``<shared>/<org>/projects``."""
    from cowork.services.projects import projects_root

    return projects_root(scope=TenantScope(org_mode=True, org_id=org_id, user_id=USER))


@pytest.fixture
def project(org_deployment) -> Path:
    """A project folder of the organization, named for this test alone."""
    folder = projects() / f"research-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    return folder


def write(root: Path, rel: str, content: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def stage(project: Path, conv: UUID, **limits):
    return stage_project_files(project, conv, scope=SCOPE, limits=LIMITS._replace(**limits))


def copy_of(project: Path, conv: UUID) -> Path:
    return project / "conversations" / str(conv) / "project-files"


def copied(project: Path, conv: UUID) -> dict[str, str]:
    """Every regular file in the conversation's copy but the not-copied list,
    by its path below the copy, with its text."""
    root = copy_of(project, conv)
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != NOT_COPIED_FILENAME
    }


def listing(project: Path, conv: UUID) -> NotCopiedFiles | None:
    path = copy_of(project, conv) / NOT_COPIED_FILENAME
    if not path.exists():
        return None
    return NotCopiedFiles.model_validate_json(path.read_bytes())


def not_copied(project: Path, conv: UUID) -> dict[str, NotCopiedReason]:
    listed = listing(project, conv)
    return {} if listed is None else {entry.path: entry.reason for entry in listed.files}


@posix_only
def test_the_projects_files_are_copied_with_their_folders(project):
    write(project, "chapter-1.md", "The porter kept the keys.")
    write(project, "data/sales.csv", "a,b\n1,2\n")
    conv = uuid4()

    result = stage(project, conv)

    assert copied(project, conv) == {"chapter-1.md": "The porter kept the keys.", "data/sales.csv": "a,b\n1,2\n"}
    assert (result.files, result.bytes, result.copied, result.not_copied, result.failed) == (2, 33, 2, 0, False)
    assert not_copied(project, conv) == {}
    assert oct((copy_of(project, conv) / "chapter-1.md").stat().st_mode & 0o777) == oct(0o600)


@posix_only
def test_only_ordinary_files_are_copied(project, tmp_path):
    """Hidden entries, the server's folders, memory, caches, OS files and
    anything that is not a regular file with one name stay out."""
    write(project, "keep.txt", "keep")
    for rel in (
        ".hidden.txt", ".anton/anton.md", ".anton/memory/rules.md", f"conversations/{uuid4()}/notes.md",
        "skills/house-style/SKILL.md", "memory/notes.md", "data/memory/notes.md", "__pycache__/m.pyc",
        "node_modules/x.js", "desktop.ini", "Thumbs.db",
    ):
        write(project, rel)
    outside = write(tmp_path, "outside/secret.txt", "secret")
    (project / "link.txt").symlink_to(outside)
    (project / "linked-folder").symlink_to(outside.parent, target_is_directory=True)
    os.link(write(project, "twice.txt"), project / "second-name.txt")
    os.mkfifo(project / "pipe")
    conv = uuid4()

    result = stage(project, conv)

    assert copied(project, conv) == {"keep.txt": "keep"}
    assert result.files == 1 and not result.failed


@posix_only
def test_names_with_a_colon_or_a_backslash_route_and_are_copied(project):
    """The shared store and the upload route take such names, so a project
    whose only file has one still routes a cloud turn to the agent, and the
    agent gets the file."""
    from cowork.services.files import project_holds_files

    write(project, "Q3: results.csv", "q3")
    write(project, "back\\slash.csv", "b")
    conv = uuid4()

    assert project_holds_files(project, cloud=True)
    result = stage(project, conv)

    assert copied(project, conv) == {"Q3: results.csv": "q3", "back\\slash.csv": "b"}
    assert (result.files, result.failed) == (2, False)


@posix_only
def test_an_unchanged_file_is_not_copied_again(project):
    write(project, "notes.md", "first")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv) / "notes.md"
    before = copy.stat()

    result = stage(project, conv)

    after = copy.stat()
    assert (result.files, result.copied) == (1, 0)
    assert (after.st_ino, after.st_ctime_ns) == (before.st_ino, before.st_ctime_ns)


@posix_only
def test_a_changed_file_is_rewritten_in_place(project):
    """A same-size edit is copied again into the same inode, so a pod that
    holds the copy open keeps a valid handle."""
    source = write(project, "notes.md", "first")
    conv = uuid4()
    stage(project, conv)
    inode = (copy_of(project, conv) / "notes.md").stat().st_ino
    source.write_text("third")

    result = stage(project, conv)

    assert result.copied == 1
    assert copied(project, conv) == {"notes.md": "third"}
    assert (copy_of(project, conv) / "notes.md").stat().st_ino == inode


@posix_only
def test_a_copy_the_pod_changed_and_dated_back_is_copied_again(project):
    """The pod can rewrite its copy and set its size and modification time
    back. The change time it can't set, and the record of what was copied is
    out of its reach, so the next turn copies the project's file again."""
    write(project, "notes.md", "honest")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv) / "notes.md"
    staged = copy.stat()
    copy.write_text("forged")
    os.utime(copy, ns=(staged.st_atime_ns, staged.st_mtime_ns))
    assert (copy.stat().st_size, copy.stat().st_mtime_ns) == (staged.st_size, staged.st_mtime_ns)

    result = stage(project, conv)

    assert result.copied == 1
    assert copied(project, conv) == {"notes.md": "honest"}


@posix_only
def test_a_copy_the_pod_wrote_into_while_it_was_copied_is_copied_again(project, monkeypatch):
    """The pod can write its copy while the server is still copying it. The
    server reads each copy back and records it only when it holds the bytes
    the server wrote, so the pod's bytes are never trusted as unchanged on a
    later turn."""
    source = "abcdefghijklmnopqrstuvwxyz"
    write(project, "notes.md", source)
    conv = uuid4()
    monkeypatch.setattr(files, "_COPY_CHUNK_BYTES", 10)
    real_write_all = files.write_all
    writes: list[int] = []

    def pod_writes_between_chunks(fd, data):
        real_write_all(fd, data)
        writes.append(len(data))
        if len(writes) == 1:
            pod = os.open(copy_of(project, conv) / "notes.md", os.O_WRONLY)
            try:
                os.pwrite(pod, b"EVIL", 0)
            finally:
                os.close(pod)

    monkeypatch.setattr(files, "write_all", pod_writes_between_chunks)
    stage(project, conv)
    monkeypatch.setattr(files, "write_all", real_write_all)
    assert copied(project, conv)["notes.md"].startswith("EVIL")

    result = stage(project, conv)

    assert result.copied == 1
    assert copied(project, conv) == {"notes.md": source}


@posix_only
def test_the_record_lives_outside_the_conversations_folder(project):
    write(project, "notes.md")
    conv = uuid4()
    stage(project, conv)

    record = files.copy_record_path(SCOPE, conv)
    conversation = project / "conversations" / str(conv)
    assert record.is_file()
    assert conversation not in record.parents and project not in record.parents


@posix_only
def test_a_removed_file_and_its_empty_folder_leave_the_copy(project):
    write(project, "keep.md")
    gone = write(project, "old/gone.md")
    conv = uuid4()
    stage(project, conv)
    gone.unlink()
    gone.parent.rmdir()

    result = stage(project, conv)

    assert copied(project, conv) == {"keep.md": "x"}
    assert not (copy_of(project, conv) / "old").exists()
    assert result.files == 1


@posix_only
def test_whatever_the_pod_adds_to_the_copy_is_removed(project):
    write(project, "keep.md")
    conv = uuid4()
    stage(project, conv)
    write(copy_of(project, conv), "planted.py", "import os")
    write(copy_of(project, conv), ".hidden/planted.md")

    stage(project, conv)

    assert sorted(p.name for p in copy_of(project, conv).iterdir()) == ["keep.md"]


@posix_only
@pytest.mark.parametrize("level", ["folder", "subfolder", "file"])
def test_a_planted_link_is_removed_and_never_followed(project, tmp_path, level):
    """The pod can swap the copy folder, a folder in it or a copied file for
    a link to somewhere else on the shared store. Staging removes the link,
    never its target, and writes a real file in its place."""
    write(project, "data/notes.md", "project text")
    conv = uuid4()
    stage(project, conv)
    write(project, "data/notes.md", "edited text")
    victim = write(tmp_path, "victim/data/notes.md", "victim text")
    copy = copy_of(project, conv)
    planted = {"folder": copy, "subfolder": copy / "data", "file": copy / "data" / "notes.md"}[level]
    target = {"folder": victim.parent.parent, "subfolder": victim.parent, "file": victim}[level]
    if planted.is_dir():
        files.shutil.rmtree(planted)
    else:
        planted.unlink()
    planted.symlink_to(target, target_is_directory=target.is_dir())

    result = stage(project, conv)

    assert not result.failed
    assert victim.read_text() == "victim text"
    assert not planted.is_symlink()
    assert copied(project, conv) == {"data/notes.md": "edited text"}


@posix_only
def test_a_hard_link_to_a_copy_is_never_written_through(project, tmp_path):
    write(project, "notes.md", "first")
    conv = uuid4()
    stage(project, conv)
    write(project, "notes.md", "second")
    elsewhere = copy_of(project, conv).parent / "kept-by-the-pod.md"
    os.link(copy_of(project, conv) / "notes.md", elsewhere)

    stage(project, conv)

    assert elsewhere.read_text() == "first"
    assert copied(project, conv) == {"notes.md": "second"}


@posix_only
def test_a_file_over_the_size_limit_is_named_and_its_old_copy_removed(project):
    big = write(project, "big.csv", "x" * 10)
    write(project, "small.md", "ok")
    conv = uuid4()
    stage(project, conv)
    big.write_text("x" * 2000)

    result = stage(project, conv)

    assert copied(project, conv) == {"small.md": "ok"}
    assert not_copied(project, conv) == {"big.csv": NotCopiedReason.TOO_LARGE}
    assert (result.files, result.not_copied) == (1, 1)


@posix_only
def test_files_past_the_file_count_are_named(project):
    for name in ("a.md", "b.md", "c.md"):
        write(project, name)
    conv = uuid4()

    result = stage(project, conv, max_files=2)

    assert len(copied(project, conv)) == 2
    assert list(not_copied(project, conv).values()) == [NotCopiedReason.TOO_MANY_FILES]
    assert (result.files, result.not_copied) == (2, 1)


@posix_only
def test_files_past_the_total_size_are_named(project):
    for name in ("a.md", "b.md", "c.md"):
        write(project, name, "x" * 400)
    conv = uuid4()

    result = stage(project, conv, max_total_bytes=1000)

    assert len(copied(project, conv)) == 2
    assert list(not_copied(project, conv).values()) == [NotCopiedReason.OVER_TOTAL_BYTES]
    assert result.bytes == 800


@posix_only
def test_the_not_copied_list_goes_once_everything_fits(project):
    write(project, "a.md")
    write(project, "b.md")
    conv = uuid4()
    stage(project, conv, max_files=1)
    assert len(not_copied(project, conv)) == 1

    stage(project, conv)

    assert not (copy_of(project, conv) / NOT_COPIED_FILENAME).exists()
    assert len(copied(project, conv)) == 2


@posix_only
def test_past_the_time_budget_the_walk_stops_and_what_it_did_not_reach_goes(project):
    """Copies the walk did not reach were never checked against the project
    this turn, so they go too, and the list says the walk stopped early. The
    next turn with time copies them again."""
    write(project, "old.md", "old")
    conv = uuid4()
    stage(project, conv)
    write(project, "new.md", "new")

    result = stage(project, conv, seconds=0.0)

    assert copied(project, conv) == {}
    assert listing(project, conv) == NotCopiedFiles(listing_truncated=True)
    assert (result.files, result.copied, result.failed) == (0, 0, False)
    stage(project, conv)
    assert copied(project, conv) == {"old.md": "old", "new.md": "new"}
    assert listing(project, conv) is None


@posix_only
def test_the_time_budget_bounds_the_walk(project, monkeypatch):
    """Once the budget is spent the walk stops, so a large project does not
    hold the turn's file store slot for one look at each of its entries."""
    for index in range(300):
        write(project, f"folder-{index % 10}/file-{index}.md")
    looked: list[str] = []
    real_lstat = project_file_view.dir_lstat

    def counted(directory, name):
        looked.append(name)
        return real_lstat(directory, name)

    monkeypatch.setattr(project_file_view, "dir_lstat", counted)

    result = stage(project, uuid4(), seconds=0.0)

    assert len(looked) < 20, len(looked)
    assert result.files == 0 and not result.failed


@posix_only
def test_a_copy_the_budget_cuts_short_is_removed_and_named(project, monkeypatch):
    """The budget is checked between chunks, so one large file can't hold
    the turn's file store slot past it, and a half-written copy is never
    left for the agent to read."""
    write(project, "long.md", "abcdefghijkl")
    conv = uuid4()
    now = [0.0]
    monkeypatch.setattr(files.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(files, "_COPY_CHUNK_BYTES", 4)
    real_write_all = files.write_all

    def slow_write(fd, data):
        real_write_all(fd, data)
        now[0] += 10.0

    monkeypatch.setattr(files, "write_all", slow_write)

    result = stage(project, conv, seconds=5.0)

    assert copied(project, conv) == {}
    assert not_copied(project, conv) == {"long.md": NotCopiedReason.OUT_OF_TIME}
    assert not result.failed


@posix_only
@pytest.mark.parametrize("failure", [errno.EIO, errno.EACCES])
def test_a_copy_that_cant_finish_empties_the_folder(project, monkeypatch, failure):
    """A read the copy can't make is not a missing file: the folder is
    emptied rather than left holding a file the project deleted."""
    write(project, "a.md", "first")
    deleted = write(project, "b.md", "deleted later")
    conv = uuid4()
    stage(project, conv)
    folder_inode = copy_of(project, conv).stat().st_ino
    deleted.unlink()
    write(project, "a.md", "second")
    real_dir_open = files.dir_open

    def refuse_the_source(directory, name, flags, mode=0o777):
        if directory.path == project and name == "a.md":
            raise OSError(failure, os.strerror(failure))
        return real_dir_open(directory, name, flags, mode)

    monkeypatch.setattr(files, "dir_open", refuse_the_source)

    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.FAILED
    assert [p.name for p in copy_of(project, conv).iterdir()] == [NOT_COPIED_FILENAME]
    assert listing(project, conv) == NotCopiedFiles(copy_failed=True)
    assert copy_of(project, conv).stat().st_ino == folder_inode


@posix_only
def test_an_unreadable_record_fails_closed(project, monkeypatch):
    write(project, "a.md")
    conv = uuid4()
    stage(project, conv)
    real_read_bytes = Path.read_bytes

    def denied(self):
        if self.name == f"{conv}.json":
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", denied)

    assert stage(project, conv).outcome is ProjectFilesOutcome.FAILED
    assert [p.name for p in copy_of(project, conv).iterdir()] == [NOT_COPIED_FILENAME]


@posix_only
def test_a_copy_that_cant_empty_its_folder_refuses_the_turn(project, monkeypatch):
    """When the folder can be neither emptied nor set aside, it may still
    hold a copy the project no longer backs, so the outcome tells the caller
    not to run the turn."""
    write(project, "a.md")
    conv = uuid4()
    stage(project, conv)

    def broken(*_args, **_kwargs):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(files, "_prune_folder", broken)
    monkeypatch.setattr(files._CopyFolders, "set_aside", broken)

    assert stage(project, conv).outcome is ProjectFilesOutcome.REFUSED


@posix_only
@not_root
@pytest.mark.parametrize("mode", [0o500, 0o300, 0o000])
@pytest.mark.parametrize("folder", ["conversation", "project-files", "project-files/data", "project-files/pod-made"])
def test_a_folder_the_pod_locked_does_not_block_the_next_copy(project, folder, mode):
    """The pod runs as the server's user, so it can chmod its conversation's
    folder, the copy's folders and folders it made there, so nothing can be
    listed, made or removed in them, or the folder can't even be opened. The
    copy gives each folder its owner's access back without opening it first,
    so the next turn still adds new files and removes deleted ones and the
    pod's own."""
    write(project, "data/old.md", "old")
    conv = uuid4()
    stage(project, conv)
    (project / "data" / "old.md").unlink()
    write(project, "data/new.md", "new")
    write(project, "top.md", "top")
    copy = copy_of(project, conv)
    write(copy, "pod-made/inner/notes.md", "the pod's")
    wedged = {"conversation": copy.parent}.get(folder, copy.parent / folder)
    if folder == "project-files/pod-made":
        (wedged / "inner").chmod(mode)
    wedged.chmod(mode)

    try:
        result = stage(project, conv)
    finally:
        _unlock(copy.parent)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert copied(project, conv) == {"data/new.md": "new", "top.md": "top"}
    assert not (copy / "pod-made").exists()
    assert not (copy.parent / files.REMOVED_DIRNAME).exists()
    if folder != "project-files/pod-made":
        assert stat.S_IMODE(wedged.stat().st_mode) == 0o700


@posix_only
@not_root
def test_a_folder_swapped_while_its_mode_is_set_is_never_opened(tmp_path, monkeypatch):
    """The mode is set before the folder is opened, so the pod can swap
    another folder in between. The open must find the folder whose mode was
    set, or refuse."""
    from cowork.common.paths import pinned_dir

    (tmp_path / "locked").mkdir()
    (tmp_path / "locked").chmod(0o000)
    (tmp_path / "other").mkdir()
    real_chmod = files.dir_chmod_folder

    def pod_swaps_after_the_chmod(directory, name, *, mode):
        info = real_chmod(directory, name, mode=mode)
        os.rename(tmp_path / name, tmp_path / "moved")
        os.rename(tmp_path / "other", tmp_path / name)
        return info

    monkeypatch.setattr(files, "dir_chmod_folder", pod_swaps_after_the_chmod)

    with pinned_dir(tmp_path) as parent, pytest.raises(OSError) as refused:
        files._open_workspace_folder(parent, "locked").close()

    assert refused.value.errno == errno.EAGAIN


def _unlock(root: Path) -> None:
    """Give every folder below ``root`` its owner's access back, so a test
    that failed half-way leaves a tree its cleanup can remove."""
    if not root.exists():
        return
    root.chmod(root.stat().st_mode | stat.S_IRWXU)
    for folder, names, _files in os.walk(root):
        for name in names:
            path = Path(folder, name)
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IRWXU)


@posix_only
@not_root
@pytest.mark.parametrize("change", ["unchanged", "edited", "deleted"])
def test_a_copy_the_pod_made_unreadable_is_copied_again_or_removed(project, change):
    """A copy file the pod set to ``0o000`` can't be opened to be rewritten
    in place: it is replaced with a new copy, or removed when its project
    file is gone, and never fails the turn."""
    source = write(project, "notes.md", "first")
    write(project, "other.md", "other")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv) / "notes.md"
    copy.chmod(0o000)
    if change == "edited":
        source.write_text("second")
    elif change == "deleted":
        source.unlink()

    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    expected = {"other.md": "other"}
    if change != "deleted":
        expected["notes.md"] = "second" if change == "edited" else "first"
        assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert copied(project, conv) == expected


@posix_only
@not_root
def test_a_copy_the_pod_made_unreadable_while_it_was_copied_is_copied_again(project, monkeypatch):
    """The pod can chmod a copy while the server writes it, before the stat
    the record keeps. The server's own descriptor still writes and reads it
    back, and the next turn sees the mode and copies the file again rather
    than keep a copy the agent can't read."""
    source = "abcdefghijklmnopqrstuvwxyz"
    write(project, "notes.md", source)
    conv = uuid4()
    monkeypatch.setattr(files, "_COPY_CHUNK_BYTES", 10)
    real_write_all = files.write_all
    writes: list[int] = []

    def pod_locks_the_copy(fd, data):
        real_write_all(fd, data)
        writes.append(len(data))
        if len(writes) == 1:
            (copy_of(project, conv) / "notes.md").chmod(0o000)

    monkeypatch.setattr(files, "write_all", pod_locks_the_copy)
    first = stage(project, conv)
    monkeypatch.setattr(files, "write_all", real_write_all)
    assert (first.outcome, first.files) == (ProjectFilesOutcome.STAGED, 1)
    assert stat.S_IMODE((copy_of(project, conv) / "notes.md").stat().st_mode) == 0o000

    result = stage(project, conv)

    assert result.copied == 1
    assert copied(project, conv) == {"notes.md": source}


def _nest(root: Path, depth: int) -> None:
    """Make ``depth`` folders, each inside the last, below ``root``, and a
    file in the deepest. Each is made from its parent's descriptor, so the
    path can run past what one system call takes."""
    root.mkdir()
    fd = os.open(root, os.O_RDONLY)
    try:
        for _level in range(depth):
            os.mkdir("d", dir_fd=fd)
            child = os.open("d", os.O_RDONLY, dir_fd=fd)
            os.close(fd)
            fd = child
        os.close(os.open("leaf.txt", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd))
    finally:
        os.close(fd)


@posix_only
@pytest.mark.parametrize("copy_fails", [False, True])
def test_a_tree_deeper_than_the_recursion_limit_never_fails_the_turn(project, monkeypatch, copy_fails):
    """The pod can nest folders in its copy deeper than Python recurses or a
    process holds descriptors. Removing them moves each folder up into the
    removed folder and deletes it a level at a time, so the turn copies, or
    empties the folder when the copy fails, and never refuses."""
    write(project, "notes.md", "kept")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv)
    _nest(copy / "pod-made", sys.getrecursionlimit() + 100)
    # The entry budget, not the time, is under test: a loaded machine must
    # not run out of time on the deep tree.
    monkeypatch.setattr(files, "_REMOVAL_SECONDS", 120.0)
    if copy_fails:
        real_dir_open = files.dir_open

        def refuse_the_source(directory, name, flags, mode=0o777):
            if directory.path == project and name == "notes.md":
                raise OSError(errno.EIO, os.strerror(errno.EIO))
            return real_dir_open(directory, name, flags, mode)

        monkeypatch.setattr(files, "dir_open", refuse_the_source)
        write(project, "notes.md", "edited")

    result = stage(project, conv)

    if copy_fails:
        assert result.outcome is ProjectFilesOutcome.FAILED
        assert [p.name for p in copy.iterdir()] == [NOT_COPIED_FILENAME]
    else:
        assert result.outcome is ProjectFilesOutcome.STAGED
        assert copied(project, conv) == {"notes.md": "kept"}
        # The turn deletes what it moved aside within its budget.
        assert not (copy.parent / files.REMOVED_DIRNAME).exists()
    assert not (copy / "pod-made").exists()


@posix_only
def test_leftovers_past_the_budget_set_the_folder_aside_for_later_turns(project, monkeypatch):
    """More leftovers than one turn may examine set the whole project-files
    folder aside, in the conversation's removed folder, and the copy goes on
    in a new one. Each later turn deletes some of what was set aside, within
    its own budget, until it is gone."""
    write(project, "notes.md", "kept")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv)
    for index in range(40):
        write(copy, f"junk-{index}.txt", "the pod's")
    old_folder = copy.stat().st_ino
    monkeypatch.setattr(files, "_REMOVAL_MAX_ENTRIES", 10)

    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert copied(project, conv) == {"notes.md": "kept"}
    removed = copy.parent / files.REMOVED_DIRNAME
    [aside] = list(removed.iterdir())
    assert aside.stat().st_ino == old_folder
    left = len(list(aside.iterdir()))
    assert left > 0

    stage(project, conv)
    assert 0 < len(list(aside.iterdir())) < left

    monkeypatch.setattr(files, "_REMOVAL_MAX_ENTRIES", 10_000)
    stage(project, conv)
    assert not removed.exists()
    assert copied(project, conv) == {"notes.md": "kept"}


@posix_only
def test_the_agent_never_lists_the_removed_folder(project, monkeypatch):
    """The agent lists only the project-files folder, and what the copy
    removes waits in a hidden folder beside it, never inside it."""
    write(project, "notes.md", "kept")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv)
    write(copy, "pod-made/notes.md", "the pod's")
    for index in range(20):
        write(copy, f"junk-{index}.txt", "the pod's")
    monkeypatch.setattr(files, "_REMOVAL_MAX_ENTRIES", 5)

    stage(project, conv)

    removed = copy.parent / files.REMOVED_DIRNAME
    assert removed.exists() and removed.parent == copy.parent
    assert files.REMOVED_DIRNAME.startswith(".")
    assert sorted(p.name for p in copy.iterdir()) == ["notes.md"]


def _open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


@posix_only
@pytest.mark.parametrize("pod", ["removes", "links"])
def test_a_kept_folder_the_pod_changes_before_the_prune_opens_it_leaks_no_descriptor(
    project, monkeypatch, tmp_path, pod,
):
    """The prune opens each folder on the way to a kept file, from a list it
    made first. The pod can remove such a folder, or put a link at its name,
    after the prune listed it and before the prune opens it. The folders
    already opened on the way are closed either way, so no turn leaves a
    descriptor open in the server, which every organization shares."""
    for index in range(5):
        write(project, f"d{index}/sub/c.txt")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv)
    real_open_copy_path = files._open_copy_path
    changed: list[tuple[str, ...]] = []

    def pod_changes_the_folder(mirror, parts):
        if len(parts) == 2:
            changed.append(parts)
            target = copy.joinpath(*parts)
            shutil.rmtree(target)
            if pod == "links":
                target.symlink_to(tmp_path, target_is_directory=True)
        return real_open_copy_path(mirror, parts)

    monkeypatch.setattr(files, "_open_copy_path", pod_changes_the_folder)
    before = _open_descriptors()

    for _turn in range(3):
        result = stage(project, conv)
        assert result.outcome is not ProjectFilesOutcome.REFUSED

    assert changed
    assert _open_descriptors() == before


@posix_only
def test_a_removed_folder_deleted_under_a_staging_is_made_again(project, monkeypatch):
    """A staging opens the removed folder on its first removal and moves each
    later folder into it. Another staging of the conversation can delete that
    folder in between, once it emptied it, and so can the pod. The next move
    then finds no folder to move into: the staging makes it again and moves
    the folder there, rather than take the folder for gone and leave its
    copy for the agent."""
    write(project, "keep.md", "k")
    write(project, "old/x.md", "o")
    write(project, "docs/y.md", "d")
    conv = uuid4()
    stage(project, conv)
    shutil.rmtree(project / "old")
    shutil.rmtree(project / "docs")
    removed = copy_of(project, conv).parent / files.REMOVED_DIRNAME
    real_rename = files.dir_rename
    moved: list[str] = []

    def another_staging_deletes_the_removed_folder(source, source_name, destination, destination_name):
        real_rename(source, source_name, destination, destination_name)
        moved.append(source_name)
        if len(moved) == 1:
            shutil.rmtree(removed)

    monkeypatch.setattr(files, "dir_rename", another_staging_deletes_the_removed_folder)

    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert sorted(moved) == ["docs", "old"]
    assert copied(project, conv) == {"keep.md": "k"}
    assert not removed.exists()


@posix_only
@not_root
def test_access_the_pod_takes_away_again_does_not_refuse_the_turn(project, monkeypatch):
    """The pod runs as the server's user, so it can take a folder's access
    away again right after the server gave it back. A copy that fails that
    way tries again to empty its folder, opening it afresh, so the pod's
    interference costs the turn its copy, never the turn itself."""
    write(project, "notes.md")
    conv = uuid4()
    stage(project, conv)
    copy = copy_of(project, conv)
    real_lstat = files.dir_lstat
    taken: list[Path] = []

    def pod_takes_access_away(directory, name):
        if directory.path == copy.parent and len(taken) < 2:
            taken.append(directory.path)
            directory.path.chmod(0o000)
        return real_lstat(directory, name)

    monkeypatch.setattr(files, "dir_lstat", pod_takes_access_away)
    try:
        result = stage(project, conv)
    finally:
        _unlock(copy.parent)

    assert len(taken) == 2
    assert result.outcome is ProjectFilesOutcome.FAILED
    assert listing(project, conv) == NotCopiedFiles(copy_failed=True)


@posix_only
def test_a_removed_folder_that_cant_be_checked_never_fails_a_finished_copy(project, monkeypatch):
    """Deleting what the removed folder holds runs after the copy finished,
    and whatever the pod did there is logged and left for a later turn."""
    write(project, "notes.md", "kept")
    conv = uuid4()
    real_lstat = files.dir_lstat

    def refused(directory, name):
        if name == files.REMOVED_DIRNAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), name)
        return real_lstat(directory, name)

    monkeypatch.setattr(files, "dir_lstat", refused)

    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert copied(project, conv) == {"notes.md": "kept"}


@posix_only
def test_each_project_copies_only_its_own_files(org_deployment):
    """A conversation in project Q never gets project P's files, and one
    that moves from P to Q gets Q's alone. Its record then names Q's files
    only, and a record written in Q matches none of P's, so moving back
    copies P's files again."""
    suffix = uuid4().hex[:8]
    p, q = projects() / f"p-{suffix}", projects() / f"q-{suffix}"
    write(p, "chapter-1.md", "The porter kept the keys.")
    write(q, "budget.csv", "a,b")
    in_p, in_q = uuid4(), uuid4()

    stage(p, in_p)
    stage(q, in_q)
    moved = stage(q, in_p)

    assert copied(p, in_p) == {"chapter-1.md": "The porter kept the keys."}
    assert copied(q, in_q) == {"budget.csv": "a,b"}
    assert copied(q, in_p) == {"budget.csv": "a,b"} and moved.copied == 1
    assert not (q / "conversations" / str(in_q) / "project-files" / "chapter-1.md").exists()
    record = files.CopyRecord.model_validate_json(files.copy_record_path(SCOPE, in_p).read_bytes())
    assert sorted(record.files) == ["budget.csv"]
    back = stage(p, in_p)
    assert back.copied == 1, "a record written for Q must not let P skip a copy"


@posix_only
def test_a_missing_project_copies_nothing(org_deployment):
    result = stage(projects() / f"gone-{uuid4().hex[:8]}", uuid4())

    assert (result.files, result.failed) == (0, False)


@posix_only
def test_a_project_of_another_organization_is_refused_and_left_alone(org_deployment):
    """The copy trusts the project folder it is given, so it refuses one that
    is not in the turn's organization's projects store, even under the same
    name, and touches nothing there."""
    name = f"research-{uuid4().hex[:8]}"
    elsewhere = projects(str(uuid4())) / name
    write(elsewhere, "chapter-1.md", "another organization's text")
    conv = uuid4()

    result = stage(elsewhere, conv)

    assert result.outcome is ProjectFilesOutcome.REFUSED
    assert not (elsewhere / "conversations").exists()
    assert not files.copy_record_path(SCOPE, conv).exists()


@posix_only
def test_deleting_a_project_file_removes_its_copies_from_every_conversation(org_deployment):
    client = TestClient(_app())
    created = client.post("/api/v1/projects/", json={"name": f"copies-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    name, project = created.json()["name"], Path(created.json()["path"])
    write(project, "chapter-1.md", "The porter kept the keys.")
    write(project, "data/notes.md", "nested")
    write(project, "other.md", "stays")
    conversations = [uuid4(), uuid4()]
    for conv in conversations:
        stage(project, conv)

    for path in ("chapter-1.md", "data/notes.md"):
        deleted = client.delete(f"/api/v1/projects/{name}/files/{path}", headers=MEMBER)
        assert deleted.status_code == 200, deleted.text

    for conv in conversations:
        assert copied(project, conv) == {"other.md": "stays"}


@posix_only
def test_a_delete_past_its_time_limit_leaves_the_rest_to_each_next_turn(org_deployment, monkeypatch, caplog):
    """The delete bounds how long it spends on the project's conversations;
    a copy it did not reach goes on that conversation's next turn."""
    from cowork.api.v1.endpoints import project_files

    client = TestClient(_app())
    created = client.post("/api/v1/projects/", json={"name": f"slow-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    name, project = created.json()["name"], Path(created.json()["path"])
    write(project, "chapter-1.md", "The porter kept the keys.")
    conv = uuid4()
    stage(project, conv)
    monkeypatch.setattr(project_files, "COPY_PURGE_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING, logger=project_files.logger.name):
        deleted = client.delete(f"/api/v1/projects/{name}/files/chapter-1.md", headers=MEMBER)

    assert deleted.status_code == 200, deleted.text
    assert copied(project, conv) == {"chapter-1.md": "The porter kept the keys."}
    assert any("time limit" in record.getMessage() for record in caplog.records)
    stage(project, conv)
    assert copied(project, conv) == {}


@posix_only
def test_deleting_a_conversation_removes_its_copy_record(org_deployment):
    client = TestClient(_app())
    created = client.post("/api/v1/projects/", json={"name": f"record-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    project = Path(created.json()["path"])
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": created.json()["id"]}, headers=MEMBER,
    )
    assert conversation.status_code == 201, conversation.text
    conv = UUID(conversation.json()["id"])
    write(project, "chapter-1.md")
    stage(project, conv)
    record = files.copy_record_path(SCOPE, conv)
    assert record.is_file()

    deleted = client.delete(f"/api/v1/conversations/{conv}", headers=MEMBER)

    assert deleted.status_code in (200, 204), deleted.text
    assert not record.exists()
    assert not (project / "conversations" / str(conv)).exists()


@posix_only
def test_the_project_listing_skips_the_conversations_copies(org_deployment):
    client = TestClient(_app())
    created = client.post("/api/v1/projects/", json={"name": f"listing-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    name, project = created.json()["name"], Path(created.json()["path"])
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": created.json()["id"]}, headers=MEMBER,
    )
    assert conversation.status_code == 201, conversation.text
    conv = UUID(conversation.json()["id"])
    write(project, "chapter-1.md")
    stage(project, conv)
    write(project / "conversations" / str(conv), "result.csv")
    # What a copy set aside and a later turn has not deleted yet.
    write(project / "conversations" / str(conv) / files.REMOVED_DIRNAME / uuid4().hex, "chapter-1.md")

    listed = client.get(f"/api/v1/projects/{name}/files", headers=MEMBER)

    assert listed.status_code == 200, listed.text
    paths = [entry["path"] for entry in listed.json()["files"] if not entry.get("synthetic")]
    assert sorted(paths) == ["chapter-1.md", f"conversations/{conv}/result.csv"]


def _write_non_utf8(folder: Path, name: bytes, content: bytes = b"x") -> None:
    """Write a file whose name is not valid UTF-8, or skip where the file
    system refuses such a name (APFS on macOS does)."""
    try:
        fd = os.open(os.fsencode(folder) + b"/" + name, os.O_WRONLY | os.O_CREAT, 0o600)
    except OSError as error:
        if error.errno in (errno.EILSEQ, errno.EINVAL):
            pytest.skip("this file system takes UTF-8 names only")
        raise
    try:
        os.write(fd, content)
    finally:
        os.close(fd)


@posix_only
def test_a_file_whose_name_is_not_utf8_is_counted_not_copied(project):
    """The copy's list and record are JSON, which can't hold such a name, so
    the file is left out and counted, and the rest of the copy goes on."""
    from cowork.services.files import project_holds_files

    _write_non_utf8(project, b"report-\xff.txt")
    conv = uuid4()

    assert project_holds_files(project, cloud=True)
    write(project, "notes.md", "kept")
    (project / "data").mkdir()
    _write_non_utf8(project / "data", b"\xfe\xfd.csv")
    result = stage(project, conv)

    assert result.outcome is ProjectFilesOutcome.STAGED
    assert copied(project, conv) == {"notes.md": "kept"}
    assert listing(project, conv) == NotCopiedFiles(unnamed=2)
    assert (result.files, result.not_copied) == (1, 2)
    assert stage(project, conv).copied == 0


@posix_only
def test_a_deleted_files_copies_are_removed_without_listing_the_pods_folders(org_deployment, monkeypatch):
    """The pod can fill its copy's folders with entries, so the delete opens
    the copy's path by its names and never lists a folder inside a
    conversation."""
    from cowork.api.v1.endpoints import project_files

    client = TestClient(_app())
    created = client.post("/api/v1/projects/", json={"name": f"listed-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert created.status_code == 201, created.text
    name, project = created.json()["name"], Path(created.json()["path"])
    write(project, "data/notes.md", "nested")
    conv = uuid4()
    stage(project, conv)
    for index in range(50):
        write(copy_of(project, conv), f"data/junk-{index}.txt")
    listed: list[Path] = []
    real_scandir = project_files.dir_scandir

    def counted(directory):
        listed.append(directory.path)
        return real_scandir(directory)

    monkeypatch.setattr(project_files, "dir_scandir", counted)

    deleted = client.delete(f"/api/v1/projects/{name}/files/data/notes.md", headers=MEMBER)

    assert deleted.status_code == 200, deleted.text
    assert not (copy_of(project, conv) / "data" / "notes.md").exists()
    conversation = project / "conversations" / str(conv)
    assert not [path for path in listed if path == conversation or conversation in path.parents]


def _app():
    from cowork.server import create_app

    return create_app()


async def test_an_in_process_turn_never_copies_the_projects_files(monkeypatch):
    """On the desktop and self-hosted, the agent runs in this process with the
    project's folder as its workspace, so it reads the project's files where
    they are. Only a cloud turn copies them: an in-process turn in a project
    with files never calls stage_project_files and leaves no copy."""
    import asyncio

    import httpx

    import cowork.handlers.responses as responses_mod
    from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
    from cowork.streaming import registry

    from _fakes import PausedHarness

    called: list[str] = []

    def never(*args, **kwargs):
        called.append("stage_project_files")

    async def never_staged(**kwargs):
        called.append("stage_remote_workspace")

    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(files, "stage_project_files", never)
    monkeypatch.setattr(responses_mod, "stage_remote_workspace", never_staged)
    monkeypatch.setattr(responses_mod, "decide_route", decide)
    harness = PausedHarness()
    harness.release.set()
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness)

    try:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/api/v1/projects/", json={"name": f"desk-{uuid4().hex[:8]}"})
            assert created.status_code == 201, created.text
            project = Path(created.json()["path"])
            write(project, "chapter-1.md", "The porter kept the keys.")
            answered = await asyncio.wait_for(
                client.post(
                    "/api/v1/responses/",
                    json={"input": "Who kept the keys?", "stream": True, "project": created.json()["name"]},
                ),
                timeout=10,
            )
    finally:
        registry.reset()

    assert called == []
    assert answered.status_code == 200 and "response.completed" in answered.text, answered.text
    assert harness.started == 1
    assert not list(project.rglob("project-files"))


@posix_only
def test_two_copies_of_one_conversation_at_once_both_finish(project):
    """Two stagings of one conversation can race to make the same folder or
    file. The one that loses takes what the other made rather than failing
    and emptying the copy."""
    for index in range(60):
        write(project, f"folder-{index % 3}/file-{index}.md", str(index))
    expected = {f"folder-{index % 3}/file-{index}.md": str(index) for index in range(60)}

    for _round in range(5):
        conv = uuid4()
        barrier = threading.Barrier(2)
        results: list = []

        def run(conv=conv, barrier=barrier, results=results):
            barrier.wait()
            results.append(stage(project, conv, max_files=100))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert [result.outcome for result in results] == [ProjectFilesOutcome.STAGED] * 2
        assert copied(project, conv) == expected


@posix_only
def test_each_folder_of_the_copy_is_opened_once_per_turn(project, monkeypatch):
    """The walk is depth first, so the copy keeps a folder open while the walk
    is inside it: the copy opens each folder once, not once per file in it.
    The clearing before the copy and the prune after it open each folder on
    the way to a kept file a few times more, never once per file either."""
    for index in range(40):
        write(project, f"a/b/file-{index}.md", str(index))
    conv = uuid4()
    stage(project, conv)
    opened: list[str] = []
    real_open = files.open_pinned_child

    def counted(directory, name, **options):
        opened.append(name)
        return real_open(directory, name, **options)

    monkeypatch.setattr(files, "open_pinned_child", counted)

    result = stage(project, conv)

    assert (result.files, result.copied) == (40, 0)
    assert opened.count("a") <= 6 and opened.count("b") <= 6, opened


@posix_only
def test_the_list_counts_the_files_it_does_not_name(project, monkeypatch):
    monkeypatch.setattr(files, "MAX_NAMED_NOT_COPIED", 2)
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        write(project, name)
    conv = uuid4()

    result = stage(project, conv, max_files=1)

    listed = listing(project, conv)
    assert (len(listed.files), listed.unnamed, listed.listing_truncated) == (2, 2, False)
    assert result.not_copied == 4


@posix_only
def test_the_list_says_when_the_walk_stopped_at_its_entry_bound(project, monkeypatch):
    monkeypatch.setattr(project_file_view, "MAX_LIST_ENTRIES", 3)
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md"):
        write(project, name)
    conv = uuid4()

    result = stage(project, conv)

    assert listing(project, conv) == NotCopiedFiles(listing_truncated=True)
    # The conversations folder the copy lives in is one of the entries seen.
    assert 0 < result.files < 5


@posix_only
def test_the_list_says_when_the_walk_stopped_at_its_depth(project, monkeypatch):
    monkeypatch.setattr(project_file_view, "MAX_DEPTH", 2)
    write(project, "a/near.md", "near")
    write(project, "a/b/deep.md", "deep")
    conv = uuid4()

    stage(project, conv)

    assert copied(project, conv) == {"a/near.md": "near"}
    assert listing(project, conv) == NotCopiedFiles(listing_truncated=True)


def test_the_copy_limits_default_to_the_decided_values(monkeypatch):
    """50 MiB a file, 200 files and 128 MiB a conversation, 10 seconds a
    turn; each can be set from the environment."""
    for name in ("MAX_FILE_BYTES", "MAX_FILES", "MAX_TOTAL_BYTES", "COPY_SECONDS"):
        monkeypatch.delenv(f"COWORK_TURN_PROJECT_FILES_{name}", raising=False)

    assert ProjectFileCopyLimits.from_settings() == ProjectFileCopyLimits(
        max_file_bytes=50 * 1024 * 1024, max_files=200, max_total_bytes=128 * 1024 * 1024, seconds=10.0,
    )
    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_MAX_FILES", "7")
    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_COPY_SECONDS", "2.5")
    assert ProjectFileCopyLimits.from_settings()[1:] == (7, 128 * 1024 * 1024, 2.5)


def test_the_servers_project_folders_are_never_ordinary():
    """The rule keeps its own list of private names; every folder the server
    keeps in a project must be on it, or routing and the copy would treat a
    new one as the user's."""
    from cowork.services.artifact_roots import SERVER_PROJECT_FOLDERS

    assert SERVER_PROJECT_FOLDERS <= project_file_view.PRIVATE_NAMES


@posix_only
def test_a_file_that_shrinks_while_it_is_copied_is_left_for_the_next_turn(project, monkeypatch):
    """A file whose bytes run out before its size was read is not the file
    the walk saw: its partial copy is removed, every descriptor is closed,
    and the next turn copies it whole."""
    write(project, "notes.md", "abcdefghijkl")
    conv = uuid4()
    real_read_bounded = files.read_bounded

    def shrunk(fd, size, chunk_size=1 << 16):
        yield from real_read_bounded(fd, size - 4, chunk_size)

    monkeypatch.setattr(files, "read_bounded", shrunk)
    open_before = len(os.listdir("/dev/fd"))

    result = stage(project, conv)

    assert len(os.listdir("/dev/fd")) == open_before
    assert (result.files, result.failed) == (0, False)
    assert copied(project, conv) == {}
    monkeypatch.setattr(files, "read_bounded", real_read_bounded)
    assert stage(project, conv).files == 1
    assert copied(project, conv) == {"notes.md": "abcdefghijkl"}
