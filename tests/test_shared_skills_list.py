"""The shared-skill list, ``<COWORK_SHARED_DIR>/shared/skills.json`` (cowork.shared_skills).

It follows COWORK_SHARED_DIR. A missing list shares nothing, and so does a
malformed one, with a warning naming the file. Every change takes the list's
flock, reads it after, and replaces the file at once, readable by everyone,
leaving no temporary file; a change that fails or changes nothing writes
nothing, and a malformed list is replaced. A list the filesystem refuses to
read is never replaced: the change answers busy. Two writers never lose each
other's entry, and one that can't take the lock in time answers busy.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import threading
import uuid
from pathlib import Path

import pytest

from cowork import managed, shared_skills
from cowork.common.settings.app_settings import get_app_settings
from cowork.managed import SharedSkill, SharedSkills
from cowork.shared_skills import (
    SKILL_SHARING_BUSY,
    SharedSkillsBusyError,
    change_shared_skills,
    read_shared_skills,
    shared_skills_file,
)
from tests._fakes import shared_skill_list_lock_held

USER = "11111111-1111-4111-8111-111111111111"


def _skill(slug: str, *, partition: str | None = None) -> SharedSkill:
    return SharedSkill(user_id=USER, partition_id=partition or str(uuid.uuid4()), skill=slug)


def _adding(entry: SharedSkill):
    def change(current: SharedSkills) -> SharedSkills:
        return SharedSkills((*current.root, entry))

    return change


@pytest.fixture
def shared_root(monkeypatch, tmp_path):
    root = tmp_path / "shared-root"
    monkeypatch.setenv("COWORK_SHARED_DIR", str(root))
    get_app_settings.cache_clear()
    yield root
    get_app_settings.cache_clear()


def _warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.shared_skills" and record.levelno == logging.WARNING
    ]


def test_the_list_follows_the_shared_root(shared_root):
    assert shared_skills_file() == shared_root / "shared" / "skills.json"


def test_a_missing_list_shares_nothing(shared_root):
    assert read_shared_skills() == SharedSkills(())


@pytest.mark.parametrize("content", ['{"not": "a list"}', "not json", '[{"skill": "../escape"}]'])
def test_a_malformed_list_shares_nothing_and_warns_naming_the_file(shared_root, caplog, owned_logger, content):
    shared_skills_file().parent.mkdir(parents=True)
    shared_skills_file().write_text(content)
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    assert read_shared_skills() == SharedSkills(())

    warnings = _warnings(caplog)
    assert warnings and str(shared_skills_file()) in warnings[0]


def test_a_change_writes_the_list_at_once_readable_by_everyone(shared_root):
    # A writer's umask that would hide the file from every other user.
    previous = os.umask(0o077)
    try:
        entry = _skill("team-report")
        updated = change_shared_skills(change=_adding(entry))
    finally:
        os.umask(previous)

    path = shared_skills_file()
    assert updated == SharedSkills((entry,))
    assert json.loads(path.read_text()) == [entry.model_dump()]
    assert path.stat().st_mode & 0o777 == 0o644
    # Only the list and its lock: no temporary file is left behind.
    assert sorted(child.name for child in path.parent.iterdir()) == ["skills.json", "skills.json.lock"]


def test_a_change_that_fails_or_changes_nothing_writes_nothing(shared_root):
    change_shared_skills(change=_adding(_skill("team-report")))
    path = shared_skills_file()
    written = (path.stat().st_ino, path.read_bytes())

    def refused(current: SharedSkills) -> SharedSkills:
        raise ValueError("refused")

    with pytest.raises(ValueError, match="refused"):
        change_shared_skills(change=refused)
    change_shared_skills(change=lambda current: current)

    assert (path.stat().st_ino, path.read_bytes()) == written
    assert sorted(child.name for child in path.parent.iterdir()) == ["skills.json", "skills.json.lock"]


def test_a_change_replaces_a_malformed_list_starting_from_empty(shared_root, caplog, owned_logger):
    path = shared_skills_file()
    path.parent.mkdir(parents=True)
    path.write_text('{"not": "a list"}')
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    change_shared_skills(change=lambda current: current)

    assert json.loads(path.read_text()) == []
    warnings = _warnings(caplog)
    assert warnings and "replacing" in warnings[0] and str(path) in warnings[0]


def test_a_change_leaves_a_list_the_filesystem_refuses_to_read(shared_root, monkeypatch, caplog, owned_logger):
    """A read error (EMFILE, EIO, EACCES) says nothing about the list's
    content: a change that replaced it as malformed would drop every share."""
    path = shared_skills_file()
    kept = (_skill("bobs-skill"), _skill("carols-skill"))
    SharedSkills(kept).write(path)
    written = path.read_bytes()
    real_open_fd = managed.open_fd
    refused: list[Path] = []

    def refuse_the_list_once(fd_path, flags, mode=0o777, *, dir_fd=None):
        if Path(fd_path) == path and not refused:
            refused.append(Path(fd_path))
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open_fd(fd_path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(managed, "open_fd", refuse_the_list_once)
    owned_logger("cowork.shared_skills", level=logging.WARNING)
    alice = _skill("alices-skill")

    with pytest.raises(SharedSkillsBusyError, match=SKILL_SHARING_BUSY):
        change_shared_skills(change=_adding(alice))

    assert refused == [path]
    assert path.read_bytes() == written
    logged = [
        (record.levelno, record.getMessage()) for record in caplog.records if record.name == "cowork.shared_skills"
    ]
    assert len(logged) == 1
    level, message = logged[0]
    assert level == logging.ERROR and str(path) in message and "replacing" not in message
    # Read again, the list keeps every entry and takes the new one.
    change_shared_skills(change=_adding(alice))
    assert read_shared_skills() == SharedSkills((*kept, alice))


def test_two_writers_at_once_both_land(shared_root):
    entries = [_skill(f"skill-{index}") for index in range(8)]
    start = threading.Barrier(len(entries))
    errors: list[BaseException] = []

    def write(entry: SharedSkill) -> None:
        start.wait()
        try:
            change_shared_skills(change=_adding(entry))
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(entry,)) for entry in entries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert {entry.skill for entry in read_shared_skills().root} == {entry.skill for entry in entries}


def _read_without_hanging(timeout: float = 2.0) -> SharedSkills:
    """read_shared_skills() off the calling thread, so a read that blocks on a
    FIFO with no writer fails the test instead of hanging it forever."""
    result: list[SharedSkills] = []

    def read() -> None:
        result.append(read_shared_skills())

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "read_shared_skills() blocked on a non-regular file"
    return result[0]


def _make_fifo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)


def _make_directory(path: Path) -> None:
    path.mkdir(parents=True)


def _make_symlink(path: Path) -> None:
    """A link to a valid list holding one entry, so only the refusal to follow
    the link keeps the read empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = path.parent / "elsewhere.json"
    SharedSkills((_skill("linked-skill"),)).write(elsewhere)
    path.symlink_to(elsewhere)


@pytest.mark.parametrize("make", [_make_fifo, _make_directory, _make_symlink], ids=["fifo", "directory", "symlink"])
def test_a_non_regular_list_shares_nothing_without_hanging(shared_root, make):
    make(shared_skills_file())

    assert _read_without_hanging() == SharedSkills(())


def test_a_fifo_at_the_path_makes_a_change_answer_busy(shared_root):
    _make_fifo(shared_skills_file())
    raised: list[Exception] = []

    def change() -> None:
        try:
            change_shared_skills(change=_adding(_skill("team-report")))
        except Exception as error:
            raised.append(error)

    thread = threading.Thread(target=change, daemon=True)
    thread.start()
    thread.join(2.0)
    assert not thread.is_alive(), "change_shared_skills() blocked on a FIFO"
    assert len(raised) == 1
    assert isinstance(raised[0], SharedSkillsBusyError)
    assert SKILL_SHARING_BUSY in str(raised[0])


def test_a_list_larger_than_the_cap_shares_nothing(shared_root, caplog, owned_logger):
    from cowork.managed import MAX_REGISTRY_BYTES

    path = shared_skills_file()
    path.parent.mkdir(parents=True)
    # A valid list padded past the cap, so only the size check keeps it out.
    entry = SharedSkills((_skill("team-report"),)).dump_json().encode()
    path.write_bytes(entry[:1] + b" " * (MAX_REGISTRY_BYTES + 1) + entry[1:])
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    assert read_shared_skills() == SharedSkills(())
    assert any(str(path) in message and "larger than" in message for message in _warnings(caplog))


def test_a_failed_write_leaves_no_temporary_file(shared_root, monkeypatch):
    path = shared_skills_file()
    path.parent.mkdir(parents=True)

    def boom(self: SharedSkills) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(SharedSkills, "dump_json", boom)

    with pytest.raises(OSError, match="No space left on device"):
        SharedSkills((_skill("team-report"),)).write(path)

    assert list(path.parent.iterdir()) == []


def test_a_write_reaches_the_disk_before_it_replaces_the_list(shared_root, monkeypatch):
    path = shared_skills_file()
    synced: list[int] = []
    real_replace = os.replace

    def fsync(descriptor: int) -> None:
        synced.append(os.fstat(descriptor).st_size)

    def replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        assert synced, "the list replaced the old one before its bytes were flushed to disk"
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)

    written = SharedSkills((_skill("team-report"),))
    written.write(path)

    assert synced == [len(written.dump_json().encode())]
    assert path.read_bytes() == written.dump_json().encode()


def test_a_held_lock_answers_busy_after_the_timeout(shared_root, monkeypatch):
    monkeypatch.setattr(shared_skills, "SHARED_SKILLS_LOCK_TIMEOUT", 0.2)
    path = shared_skills_file()
    with shared_skill_list_lock_held(), pytest.raises(SharedSkillsBusyError) as busy:
        change_shared_skills(change=_adding(_skill("team-report")))

    # The route's 503 and the operator command show this text, and the README quotes it.
    assert SKILL_SHARING_BUSY == "Sharing is busy right now. Try again in a moment."
    assert str(busy.value) == SKILL_SHARING_BUSY
    assert not path.exists()
    # Released, the next change goes through.
    change_shared_skills(change=_adding(_skill("team-report")))
    assert [entry.skill for entry in read_shared_skills().root] == ["team-report"]
