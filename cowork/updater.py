"""Self-update mechanism for cowork-server.

On startup, checks PyPI for a newer version of cowork-server and, if one
is found, upgrades via ``uv tool install --upgrade`` and re-execs so the
new code loads cleanly.  The parent Electron process sees this as a
slightly slower cold start (well within the 45-second health probe
timeout).

All errors are swallowed so a failed update never prevents the server
from booting on its current version.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

_PACKAGE_NAME = "cowork-server"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{_PACKAGE_NAME}/json"
_PYPI_TIMEOUT = 5  # seconds
_LOOP_GUARD_VAR = "_COWORK_SERVER_UPDATED"
_DISABLE_VAR = "COWORK_SERVER_DISABLE_AUTOUPDATE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv() -> str | None:
    """Locate the ``uv`` binary.

    ``shutil.which`` first (PATH), then standard install locations --
    when the Electron app launches the Python server, PATH is whatever
    Electron inherited from launchctl, which usually omits
    ``~/.local/bin`` where uv installs itself.
    """
    found = shutil.which("uv")
    if found:
        return found
    home = Path.home()
    for cand in (
        home / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
        home / ".cargo" / "bin" / "uv",
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _current_version() -> str:
    """Return the currently installed version of cowork-server."""
    from importlib.metadata import version
    return version(_PACKAGE_NAME)


def _parse_version_tuple(v: str) -> tuple[int, ...]:
    """Convert a PEP-440 version string to a tuple of ints for comparison.

    Handles simple versions like ``0.1.2``.  Pre-release suffixes
    (``a1``, ``rc2``, etc.) are stripped so comparison is approximate,
    but good enough for "is there a newer release?" checks.
    """
    import re
    # Strip pre-release / post-release suffixes for a rough comparison.
    clean = re.split(r"[^0-9.]", v)[0].rstrip(".")
    return tuple(int(p) for p in clean.split(".") if p)


def _latest_pypi_version() -> str | None:
    """Fetch the latest version string from PyPI.  Returns ``None`` on
    any error (network, timeout, unexpected JSON shape)."""
    req = Request(_PYPI_JSON_URL, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=_PYPI_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except (URLError, OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_self_update() -> None:
    """Check PyPI for a newer cowork-server and upgrade + re-exec if found.

    This function is designed to be called very early in ``main()``
    before the FastAPI app or settings are loaded.  It is completely
    safe to call -- any failure is logged as a warning and the server
    proceeds on its current version.
    """
    try:
        _do_update_check()
    except Exception:
        logger.warning("cowork-server self-update check failed", exc_info=True)


def _do_update_check() -> None:
    # Loop guard -- if we already updated and re-exec'd, don't check again.
    if os.environ.get(_LOOP_GUARD_VAR) == "1":
        return

    # User opt-out.
    disable = os.environ.get(_DISABLE_VAR, "").lower()
    if disable in ("1", "true"):
        logger.debug("Auto-update disabled via %s", _DISABLE_VAR)
        return

    current = _current_version()
    latest = _latest_pypi_version()
    if latest is None:
        logger.debug("Could not fetch latest version from PyPI; skipping update")
        return

    current_t = _parse_version_tuple(current)
    latest_t = _parse_version_tuple(latest)

    if latest_t <= current_t:
        logger.debug(
            "cowork-server is up to date (current=%s, latest=%s)", current, latest
        )
        return

    logger.info(
        "New cowork-server version available: %s -> %s; upgrading...",
        current,
        latest,
    )

    uv = _find_uv()
    if uv is None:
        logger.warning("Cannot self-update: uv binary not found")
        return

    result = subprocess.run(
        [uv, "tool", "install", "--upgrade", _PACKAGE_NAME],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        logger.warning(
            "uv tool install --upgrade failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return

    logger.info("Upgrade complete; re-execing into new version...")
    os.environ[_LOOP_GUARD_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
