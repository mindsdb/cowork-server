"""I5: filesystem consumers under ``<project>/.anton/*`` must never follow a
symlink an untrusted pod could plant, even though these paths are not currently
pod-writable. Regression for the resolve-both-legs / no-O_NOFOLLOW class shared
with the attachment-staging Critical: ``Path.resolve()`` follows a symlinked
base, so the old containment checks passed against an already-escaped path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cowork.common.paths import opened_subdir_nofollow
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import ProjectMemoryStore
from cowork.services.conversations import _sweep_skill_drafts
from cowork.services.scratchpad_sessions import remove_conversation_sessions

_CONV = "c4bf6e0c-0018-4668-bb18-9e19c658318d"


def _victim(tmp_path):
    victim = tmp_path / "other-org"
    (victim / "sub").mkdir(parents=True)
    (victim / "secret.txt").write_text("do not touch")
    (victim / "sub" / "data.txt").write_text("keep me")
    return victim


def _assert_intact(victim):
    assert (victim / "secret.txt").read_text() == "do not touch"
    assert (victim / "sub" / "data.txt").read_text() == "keep me"
    assert {p.name for p in victim.iterdir()} == {"secret.txt", "sub"}


# --- the shared primitive ---------------------------------------------------


def test_opened_subdir_nofollow_refuses_a_symlinked_component(tmp_path):
    base = tmp_path / "project"
    (base / ".anton").mkdir(parents=True)
    (base / ".anton" / "memory").symlink_to(_victim(tmp_path), target_is_directory=True)
    with pytest.raises(OSError):
        with opened_subdir_nofollow(base, ".anton", "memory"):
            pass


def test_opened_subdir_nofollow_creates_real_dirs(tmp_path):
    base = tmp_path / "project"
    base.mkdir()
    with opened_subdir_nofollow(base, ".anton", "memory", create=True) as d:
        assert isinstance(d.fd, int)  # a real O_NOFOLLOW descriptor on POSIX
    assert (base / ".anton" / "memory").is_dir()
    assert not (base / ".anton" / "memory").is_symlink()


# --- ProjectMemoryStore (I5b) ----------------------------------------------


def test_project_memory_does_not_read_or_write_through_symlinked_dir(tmp_path):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    victim = _victim(tmp_path)
    (project / ".anton" / "memory").symlink_to(victim, target_is_directory=True)

    store = ProjectMemoryStore(project)
    assert store.read(MemorySlot.RULES) == ""  # never reads through the link
    with pytest.raises(OSError):
        store.write(MemorySlot.RULES, "planted")  # refuses to write through it
    store.delete(MemorySlot.RULES)  # no-op, must not raise/escape
    _assert_intact(victim)


def test_project_memory_does_not_write_through_symlinked_slot_file(tmp_path):
    project = tmp_path / "project"
    mem = project / ".anton" / "memory"
    mem.mkdir(parents=True)
    victim = tmp_path / "secret.md"
    victim.write_text("original")
    # a symlink squatting the slot file name, pointing at another file
    from cowork.harnesses.memory.registry import SLOT_REGISTRY

    (mem / SLOT_REGISTRY[MemorySlot.RULES].filename).symlink_to(victim)

    store = ProjectMemoryStore(project)
    assert store.read(MemorySlot.RULES) == ""  # O_NOFOLLOW: ELOOP -> ""
    with pytest.raises(OSError):
        store.write(MemorySlot.RULES, "planted")
    assert victim.read_text() == "original"  # never written through the link
    store.delete(MemorySlot.RULES)  # unlinks the link, not the target
    assert victim.read_text() == "original"
    assert not (mem / SLOT_REGISTRY[MemorySlot.RULES].filename).exists()


# --- scratchpad-session retention (I5a) ------------------------------------


def test_remove_conversation_sessions_does_not_delete_through_symlink(tmp_path):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    victim = _victim(tmp_path)
    (project / ".anton" / "scratchpad-sessions").symlink_to(
        victim, target_is_directory=True
    )

    assert remove_conversation_sessions(project, _CONV) is False
    _assert_intact(victim)


# --- skill-draft sweep (I5c) -----------------------------------------------


def test_skill_draft_sweep_does_not_delete_through_symlink(tmp_path):
    project = tmp_path / "project"
    (project / ".anton").mkdir(parents=True)
    victim = _victim(tmp_path)
    (project / ".anton" / "skill_drafts").symlink_to(victim, target_is_directory=True)

    session = SimpleNamespace(
        get=lambda _model, _pid: SimpleNamespace(path=str(project))
    )
    # slug names match the victim's entries; a followed link would rmtree them
    _sweep_skill_drafts(session, "pid", {"sub", "secret.txt"})
    _assert_intact(victim)
