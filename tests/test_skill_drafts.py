"""Tests for the skill-draft turn-end protocol (services.task_objects).

A skill the agent builds via skill-creator must NOT auto-persist: it is staged
under <project>/.anton/skill_drafts/<slug>/ and surfaced as a self-contained
draft payload. A stray write into the live <project>/skills/ dir is relocated
into a draft (the auto-save backstop).
"""
import os
from pathlib import Path

from cowork.services.task_objects import (
    finalize_turn_skill_drafts,
    snapshot_skill_drafts,
    snapshot_stray_skills,
    stage_skill_draft,
)


def _point_store_at(monkeypatch, root: Path) -> None:
    """Redirect the skill store root (`get_app_settings().skill.root_dir`)."""
    import cowork.common.settings.app_settings as app_settings

    skill = type("S", (), {"root_dir": str(root)})()
    settings = type("Cfg", (), {"skill": skill, "tenancy_mode": "local"})()
    monkeypatch.setattr(app_settings, "get_app_settings", lambda: settings)

SKILL_MD = """---
name: Competitive Analysis
description: research competitors and produce a comparison report
---
Step 1. gather competitors
Step 2. compare pricing and UX
"""


def _write_skill(folder: Path, body: str = SKILL_MD) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(body, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    (tmp_path / ".anton" / "skill_drafts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_snapshot_skill_drafts_finds_only_skill_folders(tmp_path: Path):
    drafts = tmp_path / ".anton" / "skill_drafts"
    _write_skill(drafts / "alpha")
    (drafts / "not-a-skill").mkdir(parents=True)  # no SKILL.md → ignored
    assert set(snapshot_skill_drafts(drafts)) == {"alpha"}


def test_new_draft_yields_self_contained_payload(tmp_path: Path):
    project = _project(tmp_path)
    drafts = project / ".anton" / "skill_drafts"
    before = snapshot_skill_drafts(drafts)
    before_strays = snapshot_stray_skills(project / "skills")

    _write_skill(drafts / "competitive-analysis")
    (drafts / "competitive-analysis" / "helper.py").write_text("print(1)\n", encoding="utf-8")

    payloads = finalize_turn_skill_drafts(project, before, before_strays)
    assert len(payloads) == 1
    p = payloads[0]
    assert p["slug"] == "competitive-analysis"
    assert p["label"] == "competitive-analysis"
    assert p["name"] == "competitive-analysis"  # slug (no separate display name set)
    assert p["description"] == "research competitors and produce a comparison report"
    assert "gather competitors" in p["instructions"]
    assert p["skill_md"].startswith("---")  # full file for offline download
    assert {f["name"] for f in p["files"]} == {"helper.py"}


def test_preexisting_draft_not_re_emitted(tmp_path: Path):
    project = _project(tmp_path)
    drafts = project / ".anton" / "skill_drafts"
    _write_skill(drafts / "competitive-analysis")
    before = snapshot_skill_drafts(drafts)  # already contains the draft
    before_strays = snapshot_stray_skills(project / "skills")
    assert finalize_turn_skill_drafts(project, before, before_strays) == []


def test_stage_skill_draft_seeds_from_saved_skill(tmp_path: Path, monkeypatch):
    store = tmp_path / "store"
    _write_skill(store / "my-skill")
    (store / "my-skill" / "helper.py").write_text("print(1)\n", encoding="utf-8")
    _point_store_at(monkeypatch, store)

    drafts = tmp_path / "drafts"
    result = stage_skill_draft(drafts, "my-skill")
    assert result["slug"] == "my-skill"
    seeded = drafts / "my-skill"
    assert (seeded / "SKILL.md").read_text(encoding="utf-8").startswith("---")  # editing starts from saved
    assert (seeded / "helper.py").exists()  # siblings seeded too


def test_stage_skill_draft_fresh_when_no_saved_skill(tmp_path: Path, monkeypatch):
    _point_store_at(monkeypatch, tmp_path / "empty-store")

    result = stage_skill_draft(tmp_path / "drafts", "brand-new")
    assert result["slug"] == "brand-new"
    assert (tmp_path / "drafts" / "brand-new").is_dir()
    assert not (tmp_path / "drafts" / "brand-new" / "SKILL.md").exists()  # nothing to seed


def test_stage_skill_draft_does_not_clobber_in_progress_draft(tmp_path: Path, monkeypatch):
    store = tmp_path / "store"
    _write_skill(store / "dup", body="STORE VERSION\n")
    _point_store_at(monkeypatch, store)

    drafts = tmp_path / "drafts"
    (drafts / "dup").mkdir(parents=True)
    (drafts / "dup" / "SKILL.md").write_text("DRAFT IN PROGRESS\n", encoding="utf-8")

    stage_skill_draft(drafts, "dup")
    # An in-progress draft wins — the store copy must not overwrite it.
    assert (drafts / "dup" / "SKILL.md").read_text(encoding="utf-8") == "DRAFT IN PROGRESS\n"


def test_stage_skill_draft_rejects_empty_name(tmp_path: Path):
    assert "error" in stage_skill_draft(tmp_path, "")
    assert "error" in stage_skill_draft(tmp_path, "   ")


def _point_org_store_at(monkeypatch, root: Path) -> None:
    import cowork.common.settings.app_settings as app_settings
    import cowork.db.scoped as scoped_mod

    skill = type("S", (), {"root_dir": str(root)})()
    storage = type("St", (), {"shared_root": str(root)})()
    settings = type("Cfg", (), {"skill": skill, "storage": storage, "tenancy_mode": "org"})()
    monkeypatch.setattr(app_settings, "get_app_settings", lambda: settings)
    # scoped.py binds get_app_settings at import; patch its binding too so the
    # org-first path resolution reads the same stub.
    monkeypatch.setattr(scoped_mod, "get_app_settings", lambda: settings)


def test_seed_in_org_mode_without_scope_fails_closed(tmp_path: Path, monkeypatch):
    """The one path where agent input indexes a shared root: with no tenant
    scope bound, org mode must seed nothing rather than read the unkeyed root."""
    store = tmp_path / "store"
    _write_skill(store / "my-skill")
    _point_org_store_at(monkeypatch, store)

    result = stage_skill_draft(tmp_path / "drafts", "my-skill")
    assert result["slug"] == "my-skill"
    assert not (tmp_path / "drafts" / "my-skill" / "SKILL.md").exists()


def test_seed_in_org_mode_reads_the_orgs_own_root(tmp_path: Path, monkeypatch):
    from cowork.common.settings.user_settings import use_settings_scope
    from cowork.db.scoped import TenantScope

    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    store = tmp_path / "store"
    # org-first layout: <shared>/<org>/skills/<slug> (shared_root == store here)
    _write_skill(store / org / "skills" / "my-skill")
    _write_skill(store / "my-skill", body="UNKEYED — must not be read\n")
    _point_org_store_at(monkeypatch, store)

    with use_settings_scope(TenantScope(org_mode=True, org_id=org, user_id="u")):
        stage_skill_draft(tmp_path / "drafts", "my-skill")

    seeded = (tmp_path / "drafts" / "my-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "UNKEYED" not in seeded and seeded.startswith("---")


def test_seed_does_not_follow_symlinks_out_of_the_store(tmp_path: Path, monkeypatch):
    """The store is shared across the org; a symlinked skill dir or child must
    not let a draft seed copy in another org's (or arbitrary) files."""
    import os

    store = tmp_path / "store"
    _write_skill(store / "real")
    (store / "real" / "helper.py").write_text("real\n", encoding="utf-8")
    secret = tmp_path / "elsewhere" / "SECRET.md"
    secret.parent.mkdir(parents=True)
    secret.write_text("ANOTHER ORG SECRET\n", encoding="utf-8")
    os.symlink(secret, store / "real" / "stolen.md")     # symlinked child
    _point_store_at(monkeypatch, store)

    drafts = tmp_path / "drafts"
    stage_skill_draft(drafts, "real")
    assert (drafts / "real" / "helper.py").exists()      # real file seeded
    assert not (drafts / "real" / "stolen.md").exists()  # symlink not followed

    # And a symlinked skill DIR seeds nothing at all.
    os.symlink(tmp_path / "elsewhere", store / "linked")
    stage_skill_draft(drafts, "linked")
    assert list((drafts / "linked").iterdir()) == []


def test_refined_draft_is_re_emitted_and_persists(tmp_path: Path):
    project = _project(tmp_path)
    drafts = project / ".anton" / "skill_drafts"
    _write_skill(drafts / "competitive-analysis")
    before = snapshot_skill_drafts(drafts)  # captures the pre-refine content hash
    before_strays = snapshot_stray_skills(project / "skills")

    # Agent refines the SAME draft in place (rewrites SKILL.md).
    _write_skill(drafts / "competitive-analysis", body=SKILL_MD + "Step 3. refine the report\n")

    payloads = finalize_turn_skill_drafts(project, before, before_strays)
    assert len(payloads) == 1
    assert payloads[0]["slug"] == "competitive-analysis"
    assert "refine the report" in payloads[0]["instructions"]  # latest content
    assert (drafts / "competitive-analysis").is_dir()  # durable across the turn


def test_stray_auto_saved_skill_is_relocated_into_a_draft(tmp_path: Path):
    project = _project(tmp_path)
    drafts = project / ".anton" / "skill_drafts"
    before = snapshot_skill_drafts(drafts)
    before_strays = snapshot_stray_skills(project / "skills")

    # Agent wrote a real skill folder straight into the live skills dir.
    _write_skill(project / "skills" / "competitive-analysis")

    payloads = finalize_turn_skill_drafts(project, before, before_strays)
    assert len(payloads) == 1 and payloads[0]["slug"] == "competitive-analysis"
    # The stray was MOVED out of the live skills dir ...
    assert not (project / "skills" / "competitive-analysis").exists()
    # ... and staged as a draft that PERSISTS (swept only on Save/Dismiss).
    assert (drafts / "competitive-analysis").is_dir()


def test_symlinked_skill_is_not_a_stray(tmp_path: Path):
    project = _project(tmp_path)
    canonical = tmp_path / "canonical" / "enabled-skill"
    _write_skill(canonical)
    # An enabled skill is a SYMLINK into the canonical store — legitimate, never moved.
    os.symlink(canonical, project / "skills" / "enabled-skill")

    before = snapshot_skill_drafts(project / ".anton" / "skill_drafts")
    before_strays = snapshot_stray_skills(project / "skills")
    assert before_strays == set()  # symlink is not a stray
    assert finalize_turn_skill_drafts(project, before, before_strays) == []
    assert (project / "skills" / "enabled-skill").is_symlink()  # untouched
