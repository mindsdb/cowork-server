"""Updating a store's copy of a packaged builtin when a new version ships.

A copy is replaced only when it is exactly a version we shipped, in one atomic
replace of its SKILL.md; one someone edited, one holding a link, and one someone
deleted are left as they are.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

import cowork.services.skills as skills_mod
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.skills import (
    BUILTIN_SKILLS_DIR,
    BUILTIN_SKILLS_MARKER,
    BUILTIN_SKILLS_SHIPPED,
    BUILTIN_SKILLS_VERSION,
    CODE_BUILTIN_SKILLS_VERSION,
    CODE_ONLY_BUILTIN_SKILL_NAMES,
    SkillService,
    builtin_fingerprint,
    build_turn_skills,
)

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
SLUG = "pdf"
OLD_BODY = "---\nname: pdf\ndescription: An earlier shipped version.\n---\n\nUse pdftotext.\n"


def _org() -> TenantScope:
    return TenantScope(org_mode=True, org_id=ORG, user_id="u")


@pytest.fixture()
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    get_app_settings.cache_clear()
    yield tmp_path / "skills"
    get_app_settings.cache_clear()


def _store() -> Path:
    return SkillService(_org()).root


def _seed_at_previous_version_with(body: str, *, shipped: bool, monkeypatch) -> Path:
    """An org seeded one version ago whose pdf copy holds ``body``, plus a usage sidecar."""
    SkillService(_org()).ensure_builtin_skills()
    copy = _store() / SLUG
    (copy / "SKILL.md").write_text(body)
    (copy / "stats.json").write_text('{"total_recalls": 3}')
    (_store() / BUILTIN_SKILLS_MARKER).write_text(f"{BUILTIN_SKILLS_VERSION - 1}\n")
    if shipped:
        old = builtin_fingerprint(copy)
        real = skills_mod.shipped_builtin_fingerprints
        monkeypatch.setattr(
            skills_mod, "shipped_builtin_fingerprints",
            lambda slug: real(slug) | {old} if slug == SLUG else real(slug),
        )
    return copy


def test_an_unedited_copy_of_an_older_version_is_updated(skills_root, monkeypatch):
    copy = _seed_at_previous_version_with(OLD_BODY, shipped=True, monkeypatch=monkeypatch)

    assert SkillService(_org()).ensure_builtin_skills() is True

    assert (copy / "SKILL.md").read_bytes() == (BUILTIN_SKILLS_DIR / SLUG / "SKILL.md").read_bytes()
    assert (copy / "stats.json").read_text() == '{"total_recalls": 3}'
    # Nothing left behind that a listing could mistake for a skill.
    assert {s.name for s in SkillService(_org()).list_skills()} == SkillService(_org()).builtin_skill_names()
    assert SLUG in build_turn_skills(_org(), None)


def test_an_edited_copy_is_left_as_it_is(skills_root, monkeypatch):
    edited = OLD_BODY + "\nMy own extra step.\n"
    copy = _seed_at_previous_version_with(edited, shipped=False, monkeypatch=monkeypatch)

    SkillService(_org()).ensure_builtin_skills()

    assert (copy / "SKILL.md").read_text() == edited


def test_a_deleted_builtin_stays_deleted_when_a_new_version_ships(skills_root):
    SkillService(_org()).ensure_builtin_skills()
    shutil.rmtree(_store() / SLUG)
    (_store() / BUILTIN_SKILLS_MARKER).write_text(f"{BUILTIN_SKILLS_VERSION - 1}\n")

    SkillService(_org()).ensure_builtin_skills()

    assert not (_store() / SLUG).exists()
    assert (_store() / BUILTIN_SKILLS_MARKER).read_text().strip() == str(BUILTIN_SKILLS_VERSION)


def test_a_copy_holding_a_link_is_not_updated_or_written_through(skills_root, monkeypatch, tmp_path):
    copy = _seed_at_previous_version_with(OLD_BODY, shipped=False, monkeypatch=monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("not ours")
    (copy / "extra.md").symlink_to(outside)
    # Even if the files read through the link matched a shipped version, a copy
    # holding a link is never treated as one.
    as_if_regular = skills_mod.fingerprint_skill_files(
        {"SKILL.md": OLD_BODY.encode(), "extra.md": b"not ours"})
    monkeypatch.setattr(skills_mod, "shipped_builtin_fingerprints", lambda slug: frozenset({as_if_regular}))

    SkillService(_org()).ensure_builtin_skills()

    assert (copy / "SKILL.md").read_text() == OLD_BODY
    assert outside.read_text() == "not ours"


def test_a_slug_planted_as_a_link_is_not_updated(skills_root, monkeypatch, tmp_path):
    SkillService(_org()).ensure_builtin_skills()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "SKILL.md").write_text(OLD_BODY)
    shutil.rmtree(_store() / SLUG)
    (_store() / SLUG).symlink_to(elsewhere, target_is_directory=True)
    (_store() / BUILTIN_SKILLS_MARKER).write_text(f"{BUILTIN_SKILLS_VERSION - 1}\n")
    old = builtin_fingerprint(elsewhere)
    monkeypatch.setattr(skills_mod, "shipped_builtin_fingerprints", lambda slug: frozenset({old}))

    SkillService(_org()).ensure_builtin_skills()

    assert (elsewhere / "SKILL.md").read_text() == OLD_BODY
    assert os.path.islink(_store() / SLUG)


def test_desktop_updates_an_unedited_copy_and_keeps_a_deleted_one_deleted(skills_root, monkeypatch):
    from cowork.db.session import get_open_session
    from cowork.migrations import BUILTIN_SKILLS_SENTINEL, Setting, seed_builtin_skills
    from cowork.services.settings import SettingService

    store = SkillService(LOCAL_SCOPE)
    store._copy_builtin_skills()
    (store.root / SLUG / "SKILL.md").write_text(OLD_BODY)
    old = builtin_fingerprint(store.root / SLUG)
    real = skills_mod.shipped_builtin_fingerprints
    monkeypatch.setattr(skills_mod, "shipped_builtin_fingerprints",
                        lambda slug: real(slug) | {old} if slug == SLUG else real(slug))
    deleted = "gmail"
    shutil.rmtree(store.root / deleted)

    session = get_open_session()
    try:
        # A raw row, as migrations writes it: the sentinel isn't a user setting.
        row = SettingService(session)._fetch_row(BUILTIN_SKILLS_SENTINEL)
        if row is None:
            row = Setting(key=BUILTIN_SKILLS_SENTINEL, value="")
        row.value = str(BUILTIN_SKILLS_VERSION - 1)
        session.add(row)
        session.commit()
        assert seed_builtin_skills(session) is True
    finally:
        row = SettingService(session)._fetch_row(BUILTIN_SKILLS_SENTINEL)
        if row is not None:
            session.delete(row)
            session.commit()
        session.close()

    assert (store.root / SLUG / "SKILL.md").read_bytes() == (BUILTIN_SKILLS_DIR / SLUG / "SKILL.md").read_bytes()
    assert not (store.root / deleted).exists()


def test_every_packaged_builtin_is_in_the_shipped_list():
    """Run scripts/builtin_skills_shipped.py after changing a builtin, or stores
    holding the version you replaced can never be updated."""
    shipped = json.loads(BUILTIN_SKILLS_SHIPPED.read_text())
    for src in sorted(p for p in BUILTIN_SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()):
        entry = shipped.get(src.name)
        assert entry, f"{src.name} is missing from {BUILTIN_SKILLS_SHIPPED.name}"
        assert builtin_fingerprint(src) in entry["shipped"], f"{src.name}'s current content is not listed"
        # Edited without regenerating: the script, which refuses a change
        # without a version bump, never saw this content.
        assert entry["current"] == builtin_fingerprint(src), (
            f"{src.name} changed since {BUILTIN_SKILLS_SHIPPED.name} was written: bump the version, "
            "then run python scripts/builtin_skills_shipped.py"
        )
        store_version = (
            CODE_BUILTIN_SKILLS_VERSION if src.name in CODE_ONLY_BUILTIN_SKILL_NAMES else BUILTIN_SKILLS_VERSION
        )
        assert 1 <= entry["introduced"] <= entry["version"] <= store_version


def _shipped_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "builtin_skills_shipped.py"
    spec = importlib.util.spec_from_file_location("builtin_skills_shipped", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_new_builtin_is_introduced_in_the_bumped_version():
    script = _shipped_script()
    previous = {"pdf": {"introduced": 1, "version": 3, "current": "a"}}
    # Added and regenerated before the bump: refused, or stores at 3 would
    # take it for a builtin they deleted and never get it.
    with pytest.raises(script.NeedsVersionBump):
        script.entry_for("new-skill", "n", 3, previous, store_last=3)
    # After the bump: introduced in 4, and it stays 4 when regenerated again.
    entry = script.entry_for("new-skill", "n", 4, previous, store_last=3)
    assert entry == {"introduced": 4, "version": 4, "current": "n"}
    again = script.entry_for("new-skill", "n", 5, {**previous, "new-skill": entry}, store_last=4)
    assert again["introduced"] == 4 and again["version"] == 4


def test_a_changed_builtin_needs_a_version_bump():
    script = _shipped_script()
    previous = {"pdf": {"introduced": 1, "version": 3, "current": "old"}}
    with pytest.raises(script.NeedsVersionBump):
        script.entry_for("pdf", "new", 3, previous, store_last=3)
    assert script.entry_for("pdf", "new", 4, previous, store_last=3) == {
        "introduced": 1, "version": 4, "current": "new"}
    # Unchanged content keeps the version it shipped under.
    assert script.entry_for("pdf", "old", 4, previous, store_last=3)["version"] == 3


def test_a_change_is_compared_with_the_stores_last_version_not_the_skills():
    # A bump that changed only docx leaves pdf recorded at 3 while stores are
    # at 4. Editing pdf with the version still at 4 must be refused: stores at
    # marker 4 never re-seed, so the edit would never arrive.
    script = _shipped_script()
    released = {
        "pdf": {"introduced": 1, "version": 3, "current": "pdf-v3"},
        "docx": {"introduced": 1, "version": 4, "current": "docx-v4"},
    }
    with pytest.raises(script.NeedsVersionBump, match="above 4"):
        script.entry_for("pdf", "pdf-edited", 4, released, store_last=4)
    assert script.entry_for("pdf", "pdf-edited", 5, released, store_last=4)["version"] == 5


def test_every_packaged_builtin_is_a_single_skill_file():
    """The update replaces SKILL.md in one atomic step; a builtin with more
    files would never be updated in place, so adding one needs a new design."""
    for src in sorted(p for p in BUILTIN_SKILLS_DIR.iterdir() if p.is_dir()):
        assert [p.name for p in src.rglob("*")] == ["SKILL.md"], src.name


def test_a_locked_file_on_desktop_does_not_stop_boot(skills_root, monkeypatch):
    """A Windows lock or antivirus scan makes the replace fail with an OSError.
    Boot carries on, the copy stays usable, and a later start retries it."""
    from cowork.db.session import get_open_session
    from cowork.migrations import BUILTIN_SKILLS_SENTINEL, Setting, seed_builtin_skills
    from cowork.services.settings import SettingService

    store = SkillService(LOCAL_SCOPE)
    store._copy_builtin_skills()
    (store.root / SLUG / "SKILL.md").write_text(OLD_BODY)
    old = builtin_fingerprint(store.root / SLUG)
    real = skills_mod.shipped_builtin_fingerprints
    monkeypatch.setattr(skills_mod, "shipped_builtin_fingerprints",
                        lambda slug: real(slug) | {old} if slug == SLUG else real(slug))

    def locked(*_args, **_kwargs):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(os, "rename", locked)

    session = get_open_session()
    try:
        row = SettingService(session)._fetch_row(BUILTIN_SKILLS_SENTINEL)
        if row is None:
            row = Setting(key=BUILTIN_SKILLS_SENTINEL, value="")
        row.value = str(BUILTIN_SKILLS_VERSION - 1)
        session.add(row)
        session.commit()
        seed_builtin_skills(session)  # must not raise
    finally:
        row = SettingService(session)._fetch_row(BUILTIN_SKILLS_SENTINEL)
        if row is not None:
            session.delete(row)
            session.commit()
        session.close()

    assert (store.root / SLUG / "SKILL.md").read_text() == OLD_BODY
    assert not [p.name for p in (store.root / SLUG).iterdir() if p.name.endswith(".tmp")]


def test_a_deleted_builtin_does_not_come_back_from_a_leftover_copy(skills_root):
    """A folder an earlier update could leave beside the store is never used to
    put back a builtin the person deleted."""
    SkillService(_org()).ensure_builtin_skills()
    leftover = _store() / ".builtin-staging" / f"old-{SLUG}-0123456789ab"
    leftover.mkdir(parents=True)
    (leftover / "SKILL.md").write_text(OLD_BODY)
    shutil.rmtree(_store() / SLUG)
    (_store() / BUILTIN_SKILLS_MARKER).write_text(f"{BUILTIN_SKILLS_VERSION - 1}\n")

    SkillService(_org()).ensure_builtin_skills()

    assert not (_store() / SLUG).exists()


def test_a_skill_file_planted_as_a_link_is_not_written_through(skills_root, monkeypatch, tmp_path):
    copy = _seed_at_previous_version_with(OLD_BODY, shipped=False, monkeypatch=monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text(OLD_BODY)
    (copy / "SKILL.md").unlink()
    (copy / "SKILL.md").symlink_to(outside)
    # Even if what the link points at is a shipped version.
    shipped = skills_mod.fingerprint_skill_files({"SKILL.md": OLD_BODY.encode()})
    monkeypatch.setattr(skills_mod, "shipped_builtin_fingerprints", lambda slug: frozenset({shipped}))

    SkillService(_org()).ensure_builtin_skills()

    assert outside.read_text() == OLD_BODY
    assert os.path.islink(copy / "SKILL.md")


def test_a_temp_file_left_by_an_interrupted_write_does_not_block_the_update(skills_root, monkeypatch):
    copy = _seed_at_previous_version_with(OLD_BODY, shipped=True, monkeypatch=monkeypatch)
    (copy / ".SKILL.md.0123abcd.tmp").write_text("half")

    SkillService(_org()).ensure_builtin_skills()

    assert (copy / "SKILL.md").read_bytes() == (BUILTIN_SKILLS_DIR / SLUG / "SKILL.md").read_bytes()
