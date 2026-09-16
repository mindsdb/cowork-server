"""Whether a missing prerequisite is an absence or a defect.

A post-deploy test that cannot find its target or its identity has nothing to
assert, so it skips. That is the right answer on a laptop and the wrong one in
the release pipeline, where every prerequisite is supposed to be present and a
skip reads exactly like a pass. Nine skipped tests and nine passed tests print
the same green.

`COWORK_REQUIRE_INTEGRATION` says which of the two the suite is in.
`.github/workflows/tests-integration.yml` sets it to `true` for the namespaces
that must have every prerequisite, and to `false` for best-effort PR runs.
"""

from __future__ import annotations

import os
from typing import NoReturn

import pytest


def missing_prerequisite(reason: str) -> NoReturn:
    """Skip, unless the environment promised this prerequisite is present."""
    if os.environ.get("COWORK_REQUIRE_INTEGRATION") == "true":
        pytest.fail(
            f"{reason}. COWORK_REQUIRE_INTEGRATION is set, so this environment is "
            f"supposed to have it and the absence is a defect rather than a skip."
        )
    pytest.skip(reason)
