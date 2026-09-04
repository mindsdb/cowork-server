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

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text())
_UNIT_WORKFLOW = (_ROOT / ".github/workflows/tests-unit.yml").read_text()


def _names(requirements: list[str]) -> set[str]:
    """PEP 503 normalized distribution names, version specifiers dropped.

    Comparing raw strings would let the regression back in under a different
    spelling: pip and uv treat ``hermes_agent`` and ``Hermes-Agent`` as the same
    distribution, so a base dependency written either way is just as
    unsatisfiable while a literal match on ``hermes-agent`` sees nothing.
    """
    return {
        re.split(r"[<>=!~;\[\s]", req, maxsplit=1)[0].strip().lower().replace("_", "-")
        for req in requirements
    }


def _stripped_lines(text: str) -> list[str]:
    """Config lines with comments truncated, so prose cannot satisfy a match.

    Same hazard as tests/test_hosted_image_version.py documents at length: the
    comment explaining the flag below contains the flag itself, which would
    satisfy a raw substring check on its own. Kept local rather than shared,
    since two call sites is not yet a reason to extract.
    """
    return [line.split("#", 1)[0].strip() for line in text.splitlines()]


def test_hermes_agent_is_not_a_base_dependency() -> None:
    """A base install must not require hermes-agent, whatever it is spelled."""
    assert "hermes-agent" not in _names(_PYPROJECT["project"]["dependencies"])


def test_hermes_agent_is_reachable_through_its_extra() -> None:
    """Emptying or renaming the extra would silently strand the harness.

    ``uv sync --extra hermes`` would then install nothing and the two registry
    tests in test_hermes_artifact_tools.py would skip green.
    """
    extra = _PYPROJECT["project"]["optional-dependencies"]["hermes"]
    assert "hermes-agent" in _names(extra)


def test_the_unit_job_installs_the_hermes_extra() -> None:
    """Dropping the extra from CI puts those two registry tests back to skipping.

    That is the coverage loss this file's sibling assertions exist to catch, and
    it has already happened once: guarding the tests made the job green by
    removing them from it. Exact line rather than a substring, so reordering or
    removing the flag fails loudly instead of matching something adjacent.
    """
    assert "run: uv sync --group dev --extra hermes" in _stripped_lines(_UNIT_WORKFLOW)
