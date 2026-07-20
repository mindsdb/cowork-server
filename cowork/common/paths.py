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


def is_default_home() -> bool:
    """True when the effective home is the default ``~/.cowork``.

    Use this — not ``"COWORK_HOME" in os.environ`` — to decide whether behavior
    scoped to the *production* install applies (e.g. the legacy ``~/.anton/.env``
    fallback). The desktop prod build sets ``COWORK_HOME`` explicitly to
    ``~/.cowork``, so "is the var set?" would misclassify prod as an isolated
    build; "is the effective home the default?" is the correct invariant and is
    robust regardless of who sets the var (desktop, cloud, tests).
    """
    return cowork_home() == _DEFAULT_HOME
