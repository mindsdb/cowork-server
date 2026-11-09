"""The per-spawn secret the desktop app hands the server it starts.

The app sets ``COWORK_APP_SECRET_STDIN=1`` and writes one line,
``{"app_secret": "<hex>"}``, to the server's stdin, then closes it. The secret
lives only in this module: never in ``os.environ`` (the local Scratchpad
inherits the server's environment), never on disk, never in a log. Without it
the server runs in legacy mode, where nothing checks the
``X-Cowork-App-Secret`` header.
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import sys
import threading
from typing import TextIO

STDIN_FLAG = "COWORK_APP_SECRET_STDIN"
HEADER = "X-Cowork-App-Secret"
READ_TIMEOUT_SECONDS = 10.0

_SECRET_SHAPE = re.compile(r"[0-9a-f]{32,128}")
_MAX_LINE = 4096

_log = logging.getLogger(__name__)
_secret: str | None = None


def get_app_secret() -> str | None:
    return _secret


def matches(given: str | None) -> bool:
    """Whether ``given`` is the held secret. False when none is held."""
    if _secret is None or not given:
        return False
    return hmac.compare_digest(given.encode(), _secret.encode())


def read_from_stdin(stream: TextIO | None = None, *, timeout: float = READ_TIMEOUT_SECONDS) -> bool:
    """Read the app's one line from ``stream`` (stdin by default) and hold the
    secret it carries. Returns whether a secret is now held.

    The read runs on a daemon thread so a parent that never writes the line
    costs at most ``timeout`` at startup instead of hanging the server.
    """
    global _secret
    stream = sys.stdin if stream is None else stream
    line: list[str] = []

    def read() -> None:
        try:
            line.append(stream.readline(_MAX_LINE))
        except (OSError, ValueError):
            line.append("")

    reader = threading.Thread(target=read, name="app-secret-stdin", daemon=True)
    reader.start()
    reader.join(timeout)
    if not line:
        _log.warning("%s is set but no app secret arrived on stdin; running without it", STDIN_FLAG)
        return False
    try:
        value = json.loads(line[0]).get("app_secret")
    except (ValueError, AttributeError):
        value = None
    if not isinstance(value, str) or not _SECRET_SHAPE.fullmatch(value):
        _log.warning("%s is set but stdin carried no valid app secret; running without it", STDIN_FLAG)
        return False
    _secret = value
    return True
