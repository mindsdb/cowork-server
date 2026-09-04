"""The hermes harness must register only where hermes-agent is installed.

Registering it without the package produces a harness that is offered, accepted
by the settings validator, and then fails partway through a turn on
``from run_agent import AIAgent``. The gate in hermes_harness/__init__.py keeps
it out of the registry instead.

In a subprocess, because registration is a global side effect of import that
this process has already taken: the extra is installed in CI, and
test_channel_context, test_app_settings, test_hermes_turn_failure and
test_trace_tags_forwarding all import the harness module directly. A child with
find_spec patched is the only way to observe the decision the gate makes.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec

import pytest

# Report three facts, so absence of hermes cannot be confused with a registry
# that failed to populate at all.
_PROBE = """
import importlib.util
_real = importlib.util.find_spec
{patch}
import cowork.harnesses
from cowork.harnesses.base import registered_harness_ids
from cowork.harnesses.memory.adapter import get_memory_adapter
ids = registered_harness_ids()
print("anton" in ids, "hermes" in ids, get_memory_adapter("hermes") is not None)
"""

_HIDE_RUN_AGENT = (
    'importlib.util.find_spec = '
    'lambda n, p=None: None if n == "run_agent" else _real(n, p)'
)


def _probe(patch: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", _PROBE.format(patch=patch)],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_hermes_stays_unregistered_when_the_package_is_missing() -> None:
    """anton still registers, so a green result cannot mean an empty registry.

    The memory adapter must survive too: it needs no hermes import, and
    harnesses/memory/migration.py resolves it to unlink legacy hermes files.
    """
    assert _probe(_HIDE_RUN_AGENT) == "True False True"


def test_hermes_registers_when_the_package_is_present() -> None:
    """The other direction, or the gate could simply never let anything through."""
    if find_spec("run_agent") is None:
        pytest.skip("hermes-agent (optional extra) not installed")
    assert _probe("pass") == "True True True"
