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

The ``~/.cowork`` fallback is **fail-closed for source checkouts** (ENG-1541):
the desktop app sets ``COWORK_HOME`` before spawning the server, so any run
that reaches this module with ``COWORK_HOME`` unset *and* is running from a
source tree bypassed the app (a bare ``uv run cowork-server``, the
``cowork-dev-setup`` entrypoint, ``npm run dev:web``, a pyenv-shim binary).
Silently defaulting such a run to production has migrated the prod DB twice, so
we refuse it. Only an installed wheel — where the desktop prod build legitimately
leaves ``COWORK_HOME`` unset — keeps the ``~/.cowork`` default.
"""

import os
import tomllib
from functools import cache
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".cowork"


@cache
def _running_from_source() -> bool:
    """True when imported from a cowork-server source checkout, not a wheel.

    An installed wheel lives under ``site-packages`` with no project metadata
    beside it; a dev checkout has the repo's ``pyproject.toml`` (``name =
    "cowork-server"``) at its root. Cached because the answer is fixed for the
    process and reading the file on every ``cowork_home()`` call would be waste.
    """
    # paths.py → common → cowork → repo root
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return data.get("project", {}).get("name") == "cowork-server"


def cowork_home() -> Path:
    """Root directory for all cowork state (default ``~/.cowork``).

    Overridable via the ``COWORK_HOME`` env var, which desktop preview/stable
    builds set to isolate their data. Read from the environment on each call so
    tests can monkeypatch it; the desktop app sets it before the server process
    starts, so it is stable for the lifetime of a real run.

    Fails closed when ``COWORK_HOME`` is unset in a source checkout — see the
    module docstring — rather than silently returning the production home.
    """
    raw = os.environ.get("COWORK_HOME")
    if raw:
        return Path(raw).expanduser()
    if _running_from_source():
        raise RuntimeError(
            "COWORK_HOME is unset while cowork-server runs from a source "
            "checkout; refusing to default to the production home (~/.cowork). "
            "Set COWORK_HOME to an isolated dev home, e.g. "
            "COWORK_HOME=~/.cowork-dev, or COWORK_HOME=~/.cowork to "
            "deliberately target production."
        )
    return _DEFAULT_HOME


def safe_join(base: Path | str, *parts: str) -> Path:
    """Join user-controlled *parts* onto *base*, guaranteeing containment.

    Normalizes the result and rejects (``ValueError``) anything that lands
    outside *base* — a ``..`` segment, an absolute component that resets the
    join, or a name carrying a path separator. Comparison is on whole path
    components (``os.path.commonpath``), not a string prefix, so ``base`` and a
    sibling like ``<base>-other`` are correctly treated as unrelated.
    """
    base_norm = os.path.normpath(str(base))
    target = os.path.normpath(os.path.join(base_norm, *parts))
    if os.path.commonpath([base_norm, target]) != base_norm:
        raise ValueError(f"path {target!r} escapes base directory {base_norm!r}")
    return Path(target)
