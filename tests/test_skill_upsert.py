"""Tests for SkillService.save_skill (insert-or-update by slug).

The in-chat draft Save re-saves a refined skill under the same slug; it must
overwrite the stored version (scope included) rather than 409, while the manual
"Add" path (create_skill) still rejects a duplicate.
"""
from pathlib import Path

import pytest
from cowork.services.skills import SkillService


@pytest.fixture
def svc(tmp_path: Path):
    s = SkillService()
    s.root = tmp_path
    yield s


def test_save_skill_creates_when_absent(svc: SkillService):
    skill = svc.save_skill(label="Demo Skill", instructions="v1", description="first")
    assert skill.name == "demo-skill"
    assert (svc.root / "demo-skill" / "SKILL.md").exists()


def test_save_skill_overwrites_content_and_scope(svc: SkillService):
    svc.save_skill(label="Demo Skill", instructions="v1", description="first", projects=["proj-a"])
    updated = svc.save_skill(
        label="Demo Skill", instructions="v2 refined", description="second", projects=["proj-b"],
    )
    assert updated.instructions == "v2 refined"
    assert updated.description == "second"
    assert updated.projects == ["proj-b"]  # scope overwritten, not merged


def test_create_skill_still_rejects_duplicate(svc: SkillService):
    svc.create_skill(label="Demo Skill", instructions="v1")
    with pytest.raises(ValueError):
        svc.create_skill(label="Demo Skill", instructions="v2")
