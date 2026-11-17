"""The packaged skill that sets up a task to keep an artifact up to date.

Its text is the product, so these pin what it depends on: it parses as a
skill, and every tool it tells the agent to call still exists by that name.
"""
from __future__ import annotations

from anton.core.tools.skill_format import parse_skill_dir
from anton.core.tools.tool_defs import LIST_ARTIFACTS_TOOL
from anton.tools import PUBLISH_TOOL

from cowork.harnesses.anton_harness.schedule_tools import MANAGE_TOOL_NAME
from cowork.services.skills import BUILTIN_SKILL_SLUGS, BUILTIN_SKILLS_DIR

SLUG = "scheduled-artifact-refresh"
SKILL_DIR = BUILTIN_SKILLS_DIR / SLUG


def test_it_is_a_packaged_builtin_named_after_its_folder():
    skill = parse_skill_dir(SKILL_DIR)

    assert SLUG in BUILTIN_SKILL_SLUGS
    assert skill.name == SLUG
    assert "keep" in skill.description.lower() and "artifact" in skill.description.lower()


def test_every_tool_it_names_exists_under_that_name():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")

    for name in (MANAGE_TOOL_NAME, LIST_ARTIFACTS_TOOL.name, PUBLISH_TOOL.name):
        assert f"`{name}`" in text
