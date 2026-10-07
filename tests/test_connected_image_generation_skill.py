"""The image generation host skill must stay wired to the connector flow.

Its "not connected" path recalls `connect-datasource` (the skill that unlocks
the deferred connection tools) and looks up the `replicate` spec by id. Both
are plain strings in the skill body, so a rename on either side would leave
the agent with no tools and no form, and it would fall back to wandering.
"""

from __future__ import annotations

from pathlib import Path

from anton.core.memory.skills import SkillStore

import cowork.harnesses.anton_harness.harness as harness_mod
from cowork.harnesses.anton_harness.tools import build_cowork_lookup_connector_tool
from cowork.services.connectors.specs._registry import registry

_HOST_SKILLS_ROOT = Path(harness_mod.__file__).parent / "skills"
_LABEL = "connected-image-generation"


def _store(tmp_path: Path) -> SkillStore:
    return SkillStore(root=tmp_path / "user", extra_roots=[_HOST_SKILLS_ROOT])


def test_skill_ships_as_a_host_skill(tmp_path):
    skill = _store(tmp_path).load(_LABEL)
    assert skill is not None
    assert skill.label == _LABEL
    assert skill.provenance == "host"


def test_missing_connection_path_unlocks_the_connector_tools(tmp_path):
    skill = _store(tmp_path).load(_LABEL)
    unlock = build_cowork_lookup_connector_tool().unlock_skill
    assert f"recall the `{unlock}` skill" in skill.declarative_md
    assert _store(tmp_path).load(unlock) is not None


def test_looked_up_connector_id_exists(tmp_path):
    skill = _store(tmp_path).load(_LABEL)
    assert 'lookup_connector(id="replicate")' in skill.declarative_md
    assert registry.get_connector("replicate") is not None


def test_token_var_matches_the_spec_field(tmp_path):
    skill = _store(tmp_path).load(_LABEL)
    (method,) = registry.get_connector("replicate").form.methods
    (field,) = method.fields
    assert f"__{field.name.upper()}" in skill.declarative_md


def test_user_skill_with_a_generic_image_name_does_not_shadow_it(tmp_path):
    user_skill = tmp_path / "user" / "generate-image"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text(
        "---\nname: generate-image\ndescription: a draft image skill\n---\nbody\n",
        encoding="utf-8",
    )
    store = _store(tmp_path)

    skill = store.load(_LABEL)
    assert skill is not None
    assert skill.provenance == "host"
    assert {s.label for s in store.list_all()} >= {"generate-image", _LABEL}
