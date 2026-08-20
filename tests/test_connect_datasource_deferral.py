"""ENG-764: the deferred connection tools must stay wired to their skill.

`lookup_connector` / `request_credentials` / `label_connection` are hidden up
front and unlocked only when the model recalls the `connect-datasource` skill.
That wiring is a plain string match between each tool's `unlock_skill` and the
shipped skill's label — a rename of the skill folder (or a typo in one builder)
would silently regress the tools to always-eager with no error, since anton's
allowlist-mismatch check only fires when a `tool_allowlist` is configured.
These tests pin that agreement.
"""

from __future__ import annotations

from pathlib import Path

from anton.core.memory.skills import SkillStore

import cowork.harnesses.anton_harness.harness as harness_mod
from anton.tools import CONNECT_DATASOURCE_TOOL
from cowork.harnesses.anton_harness.tools import (
    build_cowork_label_connection_tool,
    build_cowork_lookup_connector_tool,
    build_cowork_publish_tool,
    build_cowork_request_credentials_tool,
)

# NOTE: these tests require an anton pin that includes the ENG-764 deferred-tool
# mechanism (anton#318: ToolDef.unlock_skill / SkillStore.extra_roots). On an
# older pin they fail (not skip) — that red is the intended signal that the pin
# must be bumped before this branch is merged.

# The host skills root exactly as the harness wires it into `skills_extra_roots`.
_HOST_SKILLS_ROOT = Path(harness_mod.__file__).parent / "skills"

_DEFERRED_BUILDERS = (
    build_cowork_lookup_connector_tool,
    build_cowork_request_credentials_tool,
    build_cowork_label_connection_tool,
)


def test_deferred_tools_share_one_unlock_skill():
    labels = {b().unlock_skill for b in _DEFERRED_BUILDERS}
    assert labels == {"connect-datasource"}, labels


def test_unlock_skill_matches_a_shipped_skill(tmp_path):
    """Each builder's `unlock_skill` must resolve to a real shipped skill whose
    canonical label is identical — catches a folder rename or a builder typo."""
    store = SkillStore(root=tmp_path / "user", extra_roots=[_HOST_SKILLS_ROOT])
    for builder in _DEFERRED_BUILDERS:
        label = builder().unlock_skill
        skill = store.load(label)
        assert skill is not None, f"no shipped skill for unlock_skill={label!r}"
        assert skill.label == label, (builder.__name__, label, skill.label)


def test_reactive_tools_stay_eager():
    """Guard the deliberate exceptions: these must never gain an `unlock_skill`."""
    assert build_cowork_publish_tool().unlock_skill is None
    assert CONNECT_DATASOURCE_TOOL.unlock_skill is None
