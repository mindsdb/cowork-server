"""Tests for SkillService.import_skill (upload of a SKILL.md file)."""
from pathlib import Path

import pytest

from cowork.services.skills import SkillService

VALID = """---
name: My Test Skill
description: does a thing
---
Step 1. do the thing
"""


@pytest.fixture
def svc(tmp_path: Path) -> SkillService:
    s = SkillService()
    s.root = tmp_path
    return s


def test_import_valid(svc: SkillService):
    skill = svc.import_skill(VALID)
    assert skill.name == "my-test-skill"  # normalized slug
    assert skill.description == "does a thing"
    assert skill.created_at is not None
    assert (svc.root / "my-test-skill" / "SKILL.md").exists()


def test_import_unparseable(svc: SkillService):
    with pytest.raises(ValueError):
        svc.import_skill("no frontmatter here")


def test_import_duplicate(svc: SkillService):
    svc.import_skill(VALID)
    with pytest.raises(FileExistsError):
        svc.import_skill(VALID)
