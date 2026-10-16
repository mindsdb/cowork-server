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

from cowork import shared_skills
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
    read_bytes = Path.read_bytes
    refused: list[Path] = []

    def refuse_the_list_once(self: Path) -> bytes:
        if self == path and not refused:
            refused.append(self)
            raise OSError(errno.EMFILE, "Too many open files")
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", refuse_the_list_once)
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
