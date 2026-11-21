"""The per-launch secrets the desktop app hands the server it starts.

The app sets ``COWORK_APP_SECRET_STDIN`` in the spawn environment and writes
one JSON line to the server's stdin, then closes it. The flag's value names
what the line carries:

- ``1`` (``SECRET_PROTOCOL``): ``{"app_secret": "<hex>"}``. A line that is
  missing, late or malformed leaves the server in legacy mode, where nothing
  checks the ``X-Cowork-App-Secret`` header. Org tenancy skips the read.
- ``2`` (``HANDOFF_PROTOCOL``): the launch hand-off,
  ``{"app_secret": "<hex>", "auth_token": "<hex>", "window_proof": "<hex>"}``
  (``LaunchHandoff``). The server reads it whatever its settings say about
  tenancy, takes its bearer token from it alone (cowork.server.create_app),
  and refuses to start when the line is missing, late, malformed or
  incomplete (``read_handoff_from_stdin``). The window proof is what the
  app's own window sends as ``X-Cowork-Window-Proof``
  (``window_proof_matches``).

No value, as in a server started from a terminal, reads nothing.

These secrets live only in this module: never in ``os.environ`` (the local
Scratchpad inherits the server's environment), never on disk, never in a
log.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import sys
import threading
from typing import Annotated, TextIO

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

# The names and numbers below are mirrored by the desktop app's
# src/main/app-secret.ts (cowork repository); change both together.
STDIN_FLAG = "COWORK_APP_SECRET_STDIN"
HEADER = "X-Cowork-App-Secret"
CHALLENGE_HEADER = "X-Cowork-App-Challenge"
#: The header the app's own window sends the launch hand-off's window proof in.
WINDOW_PROOF_HEADER = "X-Cowork-Window-Proof"
#: STDIN_FLAG's value, and /health's app_secret_protocol, for the app secret alone.
SECRET_PROTOCOL = 1
#: STDIN_FLAG's value, and /health's app_secret_protocol, for the launch hand-off.
HANDOFF_PROTOCOL = 2

READ_TIMEOUT_SECONDS = 10.0

_SECRET_SHAPE = re.compile(r"[0-9a-f]{32,128}")
_CHALLENGE_SHAPE = re.compile(r"[0-9a-fA-F]{32,128}")
_MAX_LINE = 4096

_log = logging.getLogger(__name__)
_secret: str | None = None


def _secret_shape(value: str) -> str:
    if not _SECRET_SHAPE.fullmatch(value):
        raise ValueError("expected 32 to 128 lowercase hex characters")
    return value


_HandoffSecret = Annotated[str, AfterValidator(_secret_shape), Field(repr=False)]


class LaunchHandoff(BaseModel):
    """The launch hand-off's line. Keys this server does not know are
    ignored, so a newer app can add keys for a newer server."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, hide_input_in_errors=True)

    app_secret: _HandoffSecret
    auth_token: _HandoffSecret
    window_proof: _HandoffSecret


class LaunchHandoffError(Exception):
    """The launch hand-off did not arrive whole. The message is one line
    that names the problem and never a value."""


_handoff: LaunchHandoff | None = None


def get_app_secret() -> str | None:
    return _secret


def held_handoff() -> LaunchHandoff | None:
    """The launch hand-off this server holds, or None outside a value-2 launch."""
    return _handoff


def held_protocol() -> int | None:
    """Which hand-off this server holds: HANDOFF_PROTOCOL, SECRET_PROTOCOL
    for the app secret alone, or None."""
    if _handoff is not None:
        return HANDOFF_PROTOCOL
    if _secret is not None:
        return SECRET_PROTOCOL
    return None


def matches(given: str | None) -> bool:
    """Whether ``given`` is the held secret. False when none is held."""
    if _secret is None or not given:
        return False
    return hmac.compare_digest(given.encode(), _secret.encode())


def window_proof_matches(*, given: str | None) -> bool:
    """Whether ``given`` is the held launch hand-off's window proof, compared
    in constant time. False without a hand-off or a value."""
    if _handoff is None or not given:
        return False
    return hmac.compare_digest(given.encode(), _handoff.window_proof.encode())


def proof(challenge: str | None) -> str | None:
    """HMAC-SHA256 of the app's ``challenge`` keyed by the held secret, in
    lowercase hex: lets the app tell the server it handed the secret to from
    a stand-in on its port. None without a secret or a well-formed challenge."""
    if _secret is None or challenge is None or not _CHALLENGE_SHAPE.fullmatch(challenge):
        return None
    return hmac.new(_secret.encode(), challenge.encode(), hashlib.sha256).hexdigest()


def _read_line(*, stream: TextIO | None, timeout: float) -> str | None:
    """One line from ``stream`` (stdin by default), or None when none arrived
    within ``timeout``.

    The read runs on a daemon thread so a parent that never writes the line
    costs at most ``timeout`` at startup instead of hanging the server.
    """
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
    return line[0] if line else None


def read_from_stdin(stream: TextIO | None = None, *, timeout: float = READ_TIMEOUT_SECONDS) -> bool:
    """Read the app's ``SECRET_PROTOCOL`` line from ``stream`` (stdin by
    default) and hold the secret it carries. Returns whether a secret is now
    held; on any failure the server runs in legacy mode."""
    global _secret
    line = _read_line(stream=stream, timeout=timeout)
    if line is None:
        _log.warning("%s is set but no app secret arrived on stdin; running without it", STDIN_FLAG)
        return False
    try:
        value = json.loads(line).get("app_secret")
    except (ValueError, AttributeError):
        value = None
    if not isinstance(value, str) or not _SECRET_SHAPE.fullmatch(value):
        _log.warning("%s is set but stdin carried no valid app secret; running without it", STDIN_FLAG)
        return False
    _secret = value
    return True


def read_handoff_from_stdin(
    *, stream: TextIO | None = None, timeout: float = READ_TIMEOUT_SECONDS
) -> LaunchHandoff:
    """Read the app's ``HANDOFF_PROTOCOL`` line from ``stream`` (stdin by
    default) and hold the hand-off it carries, its app secret included.

    Raises LaunchHandoffError, and holds nothing, when the line is missing,
    late, malformed or incomplete: the caller stops the start, and nothing
    falls back to a token from ``.env``.
    """
    global _secret, _handoff
    line = _read_line(stream=stream, timeout=timeout)
    if line is None:
        raise LaunchHandoffError(f"no launch hand-off arrived on stdin within {timeout:g} s")
    if not line.strip():
        raise LaunchHandoffError("the launch hand-off on stdin is empty")
    try:
        handoff = LaunchHandoff.model_validate_json(line)
    except ValidationError as error:
        raise LaunchHandoffError(_describe(error=error)) from None
    _handoff = handoff
    _secret = handoff.app_secret
    return handoff


def _describe(*, error: ValidationError) -> str:
    """One line naming the first problem in a hand-off line, without its value."""
    first = error.errors(include_input=False)[0]
    if first["type"] == "json_invalid":
        return "the launch hand-off on stdin is not valid JSON"
    if first["type"] in ("model_type", "model_attributes_type"):
        return "the launch hand-off on stdin is not a JSON object"
    key = ".".join(str(part) for part in first["loc"])
    if first["type"] == "missing":
        return f"the launch hand-off on stdin has no {key}"
    return f"the launch hand-off on stdin has an invalid {key}"
