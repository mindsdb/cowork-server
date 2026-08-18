"""The filesystem root for all cowork state.

Every piece of cowork state — the SQLite database, uploaded files, projects,
skills, memory, streams, the connector vault, the master key, the ``.env`` —
lives under a single data root. Pointing that root elsewhere isolates an
entire install: preview/stable desktop builds set ``COWORK_HOME`` to
``~/.cowork-<kind>`` so their state never collides with a user's production
``~/.cowork`` (ENG-324). Production leaves it unset and gets ``~/.cowork``.

Every default path in the codebase MUST derive from :func:`cowork_home` (via
the settings classes or directly) — a path that hardcodes ``~/.cowork`` would
silently leak across builds and defeat the isolation.
"""

import os
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cowork"


def cowork_home() -> Path:
    """Root directory for all cowork state (default ``~/.cowork``).

    Overridable via the ``COWORK_HOME`` env var, which desktop preview/stable
    builds set to isolate their data. Read from the environment on each call so
    tests can monkeypatch it; the desktop app sets it before the server process
    starts, so it is stable for the lifetime of a real run.
    """
    raw = os.environ.get("COWORK_HOME")
    return Path(raw).expanduser() if raw else _DEFAULT_HOME


def safe_join(base: Path | str, *parts: str) -> Path:
    """Join user-controlled *parts* onto *base*, guaranteeing containment.

    Two checks, both required:

    1. Lexical. Normalizes the result and rejects anything landing outside
       *base* (a ``..`` segment, an absolute component resetting the join, a
       name carrying a separator). Comparison is on whole path components, so
       *base* and a sibling like ``<base>-other`` are correctly unrelated.

    2. Symbolic. Resolves symlinks and re-checks. On shared storage the agent
       writes into its own org's tree but cowork-server reads every org's, so a
       link like ``<org_A>/x -> ../../<org_B>`` is a cross-tenant read if we
       only check the string. ``os.path.realpath`` on a path that does not exist
       yet resolves the existing prefix and leaves the rest literal, so callers
       that join before creating still work.
    """
    base_norm = os.path.normpath(str(base))
    target = os.path.normpath(os.path.join(base_norm, *parts))
    if os.path.commonpath([base_norm, target]) != base_norm:
        raise ValueError(f"path {target!r} escapes base directory {base_norm!r}")

    base_real = os.path.realpath(base_norm)
    target_real = os.path.realpath(target)
    if os.path.commonpath([base_real, target_real]) != base_real:
        raise ValueError(f"path {target!r} resolves outside base directory {base_norm!r}")

    return Path(target)
