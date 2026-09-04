"""hermes-agent must stay optional, or the published wheel cannot be installed.

hermes-agent hard-pins ``openai==2.24.0`` at every version it has published,
while anton-agent requires ``openai>=3.0``. A wheel whose metadata requires
both is unsatisfiable, so every install of it fails during resolution.

Build-level for the same reason as the image version assertions: there is no
runtime symptom. The tree imports and tests fine either way, because
``[tool.uv.sources]`` resolves both agents from git and anton's ``main`` has not
taken the openai 3 requirement yet. The failure only surfaces when a resolver is
handed the wheel's own metadata, which in CI is the publish job, after merge.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text())


def test_hermes_agent_is_not_a_base_dependency():
    deps = _PYPROJECT["project"]["dependencies"]
    assert [d for d in deps if d.startswith("hermes-agent")] == []


def test_hermes_agent_is_reachable_through_its_extra():
    extra = _PYPROJECT["project"]["optional-dependencies"]["hermes"]
    assert [d for d in extra if d.startswith("hermes-agent")] != []
