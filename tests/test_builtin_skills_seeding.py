"""Seeding the packaged builtins into a per-org skill store.

A fresh org's store starts empty and there is no org-creation hook, so seeding
runs lazily on first read. The interesting cases are all about the version
marker: when it blocks a re-seed, and when it must NOT block one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.skills import (
    BUILTIN_SKILLS_DIR,
    BUILTIN_SKILLS_MARKER,
    BUILTIN_SKILLS_VERSION,
    SkillService,
    build_turn_skills,
)


def ensure_builtin_skills(scope) -> bool:
    """``SkillService(scope).ensure_builtin_skills()``, worded for the tests
    below: seeding is a scope-level fact, not a fact about one instance."""
    return SkillService(scope).ensure_builtin_skills()

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _org(org: str | None = ORG_A, user: str = "u") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path / "skills"
    get_app_settings.cache_clear()


def _store(org: str = ORG_A) -> Path:
    """Where this org's skills actually live. Asked of the service rather than
    built by hand, so these tests survive the next move of the org layout."""
    return SkillService(_org(org)).root


def _packaged_slugs() -> set[str]:
    return {p.name for p in BUILTIN_SKILLS_DIR.iterdir()
            if p.is_dir() and (p / "SKILL.md").exists()}


def test_a_fresh_org_gets_the_packaged_builtins(skills_root):
    assert ensure_builtin_skills(_org()) is True
    assert {s.name for s in SkillService(_org()).list_skills()} == _packaged_slugs()
    assert _packaged_slugs()  # the set is non-empty, or this proves nothing


def test_seeding_is_per_org(skills_root):
    ensure_builtin_skills(_org(ORG_A))
    assert SkillService(_org(ORG_B)).list_skills() == []
    ensure_builtin_skills(_org(ORG_B))
    assert {s.name for s in SkillService(_org(ORG_B)).list_skills()} == _packaged_slugs()


def test_the_second_call_does_nothing(skills_root):
    ensure_builtin_skills(_org())
    assert ensure_builtin_skills(_org()) is False


def test_a_deleted_builtin_does_not_come_back(skills_root):
    """The marker outlives the skills, so a deliberate delete sticks."""
    ensure_builtin_skills(_org())
    svc = SkillService(_org())
    victim = sorted(_packaged_slugs())[0]
    assert svc.delete_skill(victim) is True

    ensure_builtin_skills(_org())
    assert victim not in {s.name for s in svc.list_skills()}


def test_a_lost_volume_reseeds(skills_root):
    """The marker lives with the skills, not in the DB. Wiping the store must
    look like a fresh org — a DB sentinel would survive and leave it empty."""
    ensure_builtin_skills(_org())
    import shutil

    shutil.rmtree(_store())

    assert ensure_builtin_skills(_org()) is True
    assert {s.name for s in SkillService(_org()).list_skills()} == _packaged_slugs()


def test_a_bumped_version_reseeds(skills_root, monkeypatch):
    ensure_builtin_skills(_org())
    marker = _store() / BUILTIN_SKILLS_MARKER
    assert marker.read_text().strip() == str(BUILTIN_SKILLS_VERSION)

    monkeypatch.setattr("cowork.services.skills.BUILTIN_SKILLS_VERSION", BUILTIN_SKILLS_VERSION + 1)
    assert ensure_builtin_skills(_org()) is True
    assert marker.read_text().strip() == str(BUILTIN_SKILLS_VERSION + 1)


def test_a_corrupt_marker_is_treated_as_unseeded(skills_root):
    _store().mkdir(parents=True)
    (_store() / BUILTIN_SKILLS_MARKER).write_text("not-a-number")
    assert ensure_builtin_skills(_org()) is True


def test_the_marker_is_invisible_to_readers(skills_root):
    """It is a plain file in the store's root, so it must not surface as a skill
    in either read path."""
    ensure_builtin_skills(_org())
    assert BUILTIN_SKILLS_MARKER not in {s.name for s in SkillService(_org()).list_skills()}
    assert BUILTIN_SKILLS_MARKER not in build_turn_skills(_org(), None)


def test_local_mode_is_left_alone(skills_root):
    """Desktop seeds at boot into the unkeyed root; this must not double up."""
    assert ensure_builtin_skills(LOCAL_SCOPE) is False
    assert ensure_builtin_skills(None) is False
    assert not skills_root.exists() or list(skills_root.iterdir()) == []


def test_a_missing_org_still_fails_closed(skills_root):
    """Seeding must not soften the store's fail-closed contract: org mode without
    an org id raises, exactly as the SkillService construction right after it
    would. Only filesystem trouble degrades to a no-op."""
    from cowork.db.scoped import MissingTenantScopeError

    with pytest.raises(MissingTenantScopeError):
        ensure_builtin_skills(_org(None))


def test_a_turn_seeds_on_its_own(skills_root, tmp_path):
    """An org that has only ever chatted, never opened the skills menu, still
    gets the builtins into its turn payload."""
    project = tmp_path / "proj"
    project.mkdir()
    payload = build_turn_skills(_org(), str(project))
    assert _packaged_slugs() & set(payload)


def test_an_unwritable_store_does_not_break_reads(skills_root, monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("cowork.services.skills.SkillService._copy_builtin_skills", _boom)
    assert ensure_builtin_skills(_org()) is False
    assert build_turn_skills(_org(), None) == {}


def test_a_poisoned_slug_does_not_break_the_org(skills_root, tmp_path):
    """The agent writes into its own org's tree on shared storage, so it can
    plant `skills/<builtin-slug>` as a symlink pointing out of the store. That
    one builtin is skipped; letting it propagate would 500 every skills read and
    every turn for the org — a tenant must not be able to do that to itself."""
    escaped = tmp_path / "escaped"
    store = _store()
    store.mkdir(parents=True, exist_ok=True)
    victim = sorted(_packaged_slugs())[0]
    (store / victim).symlink_to(escaped, target_is_directory=True)

    ensure_builtin_skills(_org())

    assert not escaped.exists()                       # no write through the link
    names = {s.name for s in SkillService(_org()).list_skills()}
    assert victim not in names
    assert names == _packaged_slugs() - {victim}      # the rest still seeded
    assert build_turn_skills(_org(), None)            # and turns still work


def test_a_build_without_the_packaged_skills_is_not_marked_seeded(skills_root, monkeypatch):
    """Nothing to seed from is a packaging fault, not a seeded org. Marking it
    done would leave the org empty forever once the image is fixed."""
    import cowork.services.skills as skills_mod

    packaged = skills_mod.BUILTIN_SKILLS_DIR
    monkeypatch.setattr(skills_mod, "BUILTIN_SKILLS_DIR", Path("/nonexistent"))
    _store().mkdir(parents=True, exist_ok=True)     # store exists, just unseeded

    assert ensure_builtin_skills(_org()) is False
    assert not (_store() / BUILTIN_SKILLS_MARKER).exists()

    # A later build that ships them seeds normally. Restored narrowly rather than
    # with monkeypatch.undo(), which would also revert the fixture's env vars.
    monkeypatch.setattr(skills_mod, "BUILTIN_SKILLS_DIR", packaged)
    assert ensure_builtin_skills(_org()) is True
    assert {s.name for s in SkillService(_org()).list_skills()} == _packaged_slugs()


def test_a_symlink_planted_under_dest_does_not_break_the_org(skills_root, tmp_path):
    """Sibling of the poisoned-slug case: the agent can create the slug folder
    with a symlinked SKILL.md between our exists() check and the write. safe_join
    rejects it with ValueError, which must not escape and 500 the org."""
    escaped = tmp_path / "escaped.md"
    store = _store()
    victim = sorted(_packaged_slugs())[0]
    (store / victim).mkdir(parents=True)
    (store / victim / "SKILL.md").symlink_to(escaped)   # dangling: exists() is False

    ensure_builtin_skills(_org())                        # must not raise

    assert not escaped.exists()                          # no write through the link
    names = {s.name for s in SkillService(_org()).list_skills()}
    assert names == _packaged_slugs() - {victim}         # the rest still seeded
    assert build_turn_skills(_org(), None)               # and turns still work


def test_file_mode_survives_seeding(skills_root, tmp_path, monkeypatch):
    """copytree preserved mode; the per-file copy must too, or a future builtin
    shipping an executable helper arrives without +x."""
    import cowork.services.skills as skills_mod

    packaged = tmp_path / "packaged"
    (packaged / "with-script" / "scripts").mkdir(parents=True)
    (packaged / "with-script" / "SKILL.md").write_text("---\nname: with-script\n---\nbody")
    helper = packaged / "with-script" / "scripts" / "run.sh"
    helper.write_text("#!/bin/sh\necho hi\n")
    helper.chmod(0o755)
    monkeypatch.setattr(skills_mod, "BUILTIN_SKILLS_DIR", packaged)

    assert ensure_builtin_skills(_org()) is True
    copied = _store() / "with-script" / "scripts" / "run.sh"
    assert copied.stat().st_mode & 0o111, "executable bit lost in the copy"
