"""The skills payload for remote (pod) turns: build_turn_skills.

Cloud turns can't use symlink distribution (deliberately disabled in org mode —
it resolved from the unkeyed root), so the org's skills travel inside the turn
payload instead. These tests pin the selection rules (enabled + project scoping,
mirroring skill_links), the org keying, and the size bounds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import MissingTenantScopeError, TenantScope
from cowork.services.skills import SkillService, build_turn_skills

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _org(org: str | None, user: str = "u") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def skills_root(tmp_path, monkeypatch):
    """An org whose builtins are already seeded, so the exact-payload assertions
    below stay about the selection rules. Seeding itself is covered by
    tests/test_builtin_skills_seeding.py."""
    from cowork.migrations import BUILTIN_SKILLS_MARKER, BUILTIN_SKILLS_VERSION

    root = tmp_path / "skills"
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(root))
    get_app_settings.cache_clear()
    for org in (ORG_A, ORG_B):
        (root / org).mkdir(parents=True)
        (root / org / BUILTIN_SKILLS_MARKER).write_text(f"{BUILTIN_SKILLS_VERSION}\n")
    yield root
    get_app_settings.cache_clear()


# ── selection ────────────────────────────────────────────────────────────────

def test_turn_skills_payload_is_org_scoped(skills_root):
    SkillService(_org(ORG_A)).create_skill(label="Report", name="report", instructions="steps")

    payload = build_turn_skills(_org(ORG_A))
    assert list(payload) == ["report"]
    assert "steps" in payload["report"]["files"]["SKILL.md"]

    assert build_turn_skills(_org(ORG_B)) == {}   # invisible cross-org


def test_turn_skills_fail_closed_without_org(skills_root):
    with pytest.raises(MissingTenantScopeError):
        build_turn_skills(_org(None))


def test_disabled_skill_is_not_sent(skills_root):
    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Off", name="off", instructions="i", enabled=False)
    svc.create_skill(label="On", name="on", instructions="i")
    assert list(build_turn_skills(_org(ORG_A))) == ["on"]


def test_project_scoping_mirrors_symlink_distribution(skills_root, tmp_path):
    """metadata.projects entries are project FOLDER names (skill_links parity):
    empty means every project, listed means only those."""
    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Everywhere", name="everywhere", instructions="i")
    svc.create_skill(label="Scoped", name="scoped", instructions="i", projects=["proj-a"])

    proj_a = str(tmp_path / "projects" / "proj-a")
    proj_b = str(tmp_path / "projects" / "proj-b")
    assert set(build_turn_skills(_org(ORG_A), proj_a)) == {"everywhere", "scoped"}
    assert set(build_turn_skills(_org(ORG_A), proj_b)) == {"everywhere"}
    # No project context → a project-scoped skill has nowhere to apply.
    assert set(build_turn_skills(_org(ORG_A))) == {"everywhere"}


# ── directory contents ───────────────────────────────────────────────────────

def test_payload_carries_the_whole_directory_as_text(skills_root):
    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Multi", name="multi", instructions="body")
    d = svc.root / "multi"
    (d / "references").mkdir()
    (d / "references" / "recipe.md").write_text("- tip", encoding="utf-8")
    (d / "helper.py").write_text("print(1)\n", encoding="utf-8")
    (d / "stats.json").write_text("{}", encoding="utf-8")          # private sidecar
    (d / ".hidden").write_text("x", encoding="utf-8")              # never authored content
    (d / "image.png").write_bytes(b"\x89PNG\x00\xff\xfe")          # binary → skipped

    files = build_turn_skills(_org(ORG_A))["multi"]["files"]
    assert set(files) == {"SKILL.md", "references/recipe.md", "helper.py"}
    assert files["references/recipe.md"] == "- tip"


def test_oversized_sibling_is_skipped_not_truncated(skills_root, monkeypatch):
    import cowork.services.skills as skills_mod

    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Big", name="big", instructions="body")
    (svc.root / "big" / "huge.md").write_text("x" * 500, encoding="utf-8")
    monkeypatch.setattr(skills_mod, "_TURN_SKILL_FILE_MAX", 400)

    files = build_turn_skills(_org(ORG_A))["big"]["files"]
    # A truncated script/reference that half-applies is worse than an absent one.
    assert "huge.md" not in files and "SKILL.md" in files


def test_oversized_skill_md_drops_the_whole_skill(skills_root, monkeypatch):
    import cowork.services.skills as skills_mod

    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Fat", name="fat", instructions="y" * 500)
    svc.create_skill(label="Fine", name="fine", instructions="ok")
    monkeypatch.setattr(skills_mod, "_TURN_SKILL_FILE_MAX", 400)

    assert list(build_turn_skills(_org(ORG_A))) == ["fine"]


def test_per_file_cap_counts_wire_bytes_not_chars(skills_root, monkeypatch):
    """The pod stdin cap is bytes and the controller uses ensure_ascii JSON
    (CJK → 6 bytes/char), so the per-file cap must count wire bytes: a file
    small in chars but large on the wire still has to drop."""
    import cowork.services.skills as skills_mod

    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="cjk", name="cjk", instructions="日" * 100)  # ~135 chars, ~640 wire bytes
    monkeypatch.setattr(skills_mod, "_TURN_SKILL_FILE_MAX", 300)
    # char count would keep the SKILL.md; wire count drops it, and with no
    # SKILL.md the whole skill drops.
    assert build_turn_skills(_org(ORG_A)) == {}


def test_local_mode_reads_the_shared_root(skills_root):
    SkillService().create_skill(label="Local", name="local", instructions="i")
    assert list(build_turn_skills(None)) == ["local"]


def test_slug_is_the_directory_name_not_the_frontmatter(skills_root):
    """A hand-edited store can drift frontmatter `name` from the dir name. The
    payload must ship the listed directory's own files under its dir name —
    resolving by frontmatter would ship another dir's files, letting an
    impostor dir claiming a disabled skill's name smuggle its content out."""
    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Mine", name="mine", instructions="DISABLED CONTENT",
                     enabled=False)
    svc.create_skill(label="Impostor", name="impostor", instructions="impostor body")
    md = (svc.root / "impostor" / "SKILL.md")
    md.write_text(md.read_text(encoding="utf-8").replace("name: impostor", "name: mine"),
                  encoding="utf-8")

    payload = build_turn_skills(_org(ORG_A))
    assert list(payload) == ["impostor"]
    assert "DISABLED CONTENT" not in payload["impostor"]["files"]["SKILL.md"]


def test_symlinks_cannot_pull_in_another_orgs_files(skills_root):
    """The tenancy wall: a symlinked file or directory planted inside org A's
    skill must never ship org B's content (or anything outside the skill dir)."""
    import os

    victim = SkillService(_org(ORG_B))
    victim.create_skill(label="Secret", name="secret", instructions="ORG B SECRET")

    svc = SkillService(_org(ORG_A))
    svc.create_skill(label="Evil", name="evil", instructions="body")
    d = svc.root / "evil"
    os.symlink(victim.root / "secret" / "SKILL.md", d / "steal.md")   # file link
    os.symlink(victim.root / "secret", d / "linkdir")                 # dir link

    payload = build_turn_skills(_org(ORG_A))
    dumped = str(payload)
    assert set(payload["evil"]["files"]) == {"SKILL.md"}
    assert "ORG B SECRET" not in dumped


def test_symlinked_top_level_skill_dir_is_dropped(skills_root):
    """The dir itself being a symlink into another org's store: is_dir() follows
    it, so the whole entry must be rejected before its contents are walked."""
    import os

    victim = SkillService(_org(ORG_B))
    victim.create_skill(label="Secret", name="secret", instructions="ORG B SECRET")

    svc = SkillService(_org(ORG_A))
    svc.root.mkdir(parents=True, exist_ok=True)
    os.symlink(victim.root / "secret", svc.root / "evil")   # evil -> org B's dir

    payload = build_turn_skills(_org(ORG_A))
    assert payload == {}
    assert "ORG B SECRET" not in str(payload)
