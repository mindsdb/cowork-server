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
)

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
    assert snapshot_skill_drafts(drafts) == {"alpha"}


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


def test_stray_auto_saved_skill_is_relocated_into_a_draft(tmp_path: Path):
    project = _project(tmp_path)
    drafts = project / ".anton" / "skill_drafts"
    before = snapshot_skill_drafts(drafts)
    before_strays = snapshot_stray_skills(project / "skills")

    # Agent wrote a real skill folder straight into the live skills dir.
    _write_skill(project / "skills" / "competitive-analysis")

    payloads = finalize_turn_skill_drafts(project, before, before_strays)
    assert len(payloads) == 1 and payloads[0]["slug"] == "competitive-analysis"
    # The stray was MOVED out of the live skills dir (not persisted) ...
    assert not (project / "skills" / "competitive-analysis").exists()
    # ... and now lives as a draft.
    assert (drafts / "competitive-analysis" / "SKILL.md").exists()


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
