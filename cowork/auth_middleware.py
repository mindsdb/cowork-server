"""Optional bearer-token auth middleware for the Cowork server.

When COWORK_REQUIRE_AUTH=true the server validates every request (except
OPTIONS preflight and the /health endpoint) against a shared secret token
stored in ~/.cowork/.env as COWORK_AUTH_TOKEN.  If no token is set, one is
auto-generated at startup and written back to that file so the desktop app
can read it.

The feature is off by default — existing installs see no behaviour change
unless they explicitly set COWORK_REQUIRE_AUTH=true.
"""

from __future__ import annotations

import logging
import re
import secrets
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that are always accessible without a token (health probe + CORS preflight).
_EXEMPT_PATHS = frozenset({"/api/v1/health", "/api/v1/health/"})


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry the correct bearer token.

    Registered in create_app() only when COWORK_REQUIRE_AUTH=true.
    """

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        # Always let CORS preflight through — the browser sends OPTIONS before
        # the real request and never includes an Authorization header.
        if request.method == "OPTIONS":
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != self._token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        return await call_next(request)


def ensure_auth_token(env_path: Path) -> str:
    """Return the COWORK_AUTH_TOKEN from env_path, generating one if absent.

    If a new token is generated it is appended to env_path so both the server
    and the desktop app can read it from disk on the next cold start.
    """
    token = _read_token(env_path)
    if token:
        return token

    token = secrets.token_urlsafe(32)
    _write_token(env_path, token)
    logger.info("auth: generated COWORK_AUTH_TOKEN and wrote to %s", env_path)
    return token


# ── private helpers ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"^COWORK_AUTH_TOKEN\s*=\s*(.+)$", re.MULTILINE)


def _read_token(env_path: Path) -> str:
    """Extract COWORK_AUTH_TOKEN from env_path; return '' if not found."""
    if not env_path.exists():
        return ""
    text = env_path.read_text(encoding="utf-8")
    m = _TOKEN_RE.search(text)
    return m.group(1).strip().strip('"').strip("'") if m else ""


def _write_token(env_path: Path, token: str) -> None:
    """Append COWORK_AUTH_TOKEN=<token> to env_path (creating the file if needed)."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    if _TOKEN_RE.search(existing):
        # Replace the existing line instead of duplicating it.
        new_text = _TOKEN_RE.sub(f"COWORK_AUTH_TOKEN={token}", existing)
    else:
        sep = "\n" if existing and not existing.endswith("\n") else ""
        new_text = existing + sep + f"COWORK_AUTH_TOKEN={token}\n"

    env_path.write_text(new_text, encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
