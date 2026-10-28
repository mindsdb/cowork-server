"""Bearer-token auth middleware for the Cowork server.

When COWORK_REQUIRE_AUTH is true the server validates every request (except
OPTIONS preflight and the /health endpoint) against a shared secret token
stored in ~/.cowork/.env as COWORK_AUTH_TOKEN.  If no token is set, one is
auto-generated at startup and written back to that file so the desktop app
can read it. With COWORK_TRUST_IDENTITY_HEADERS on, the token comes from the
environment alone, and create_app removes any copy from that file
(remove_auth_token).

Defaults on in local/desktop tenancy (see AppSettings.require_auth) and stays
off in org mode, where it isn't supported at all — see create_app()'s
RuntimeError for why.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Collection
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from cowork.common.paths import open_fd

logger = logging.getLogger(__name__)

# Paths that are always accessible without a token (health probe + CORS preflight).
_EXEMPT_PATHS = frozenset({
    "/api/v1/health", "/api/v1/health/",
    "/api/v1/health/live",
})


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry the correct bearer token.

    Registered in create_app() only when COWORK_REQUIRE_AUTH=true.
    """

    def __init__(self, app: ASGIApp, token: str, exempt_paths: Collection[str] = ()) -> None:
        super().__init__(app)
        self._token = token
        # Routes reachable without a bearer token in addition to /health — e.g.
        # external channel webhooks, which carry their own signature. create_app
        # passes a set it keeps filling while mounting routes; it finishes
        # before the first request, so a reference is enough.
        self._exempt_paths = exempt_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # Let a CORS preflight through: the browser sends OPTIONS before the
        # real request and never includes an Authorization header. Only a
        # genuine one, carrying Origin and Access-Control-Request-Method
        # together, as TrustedHeaderMiddleware requires: an OPTIONS without
        # them is an ordinary request, and passing it here would let it name
        # a person in the identity headers with no token.
        if (
            request.method == "OPTIONS"
            and request.headers.get("origin")
            and request.headers.get("access-control-request-method")
        ):
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS or request.url.path in self._exempt_paths:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        # Constant-time compare so response timing can't leak the token.
        if not hmac.compare_digest(token, self._token):
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


def sync_auth_token(env_path: Path, token: str) -> None:
    """Mirror the effective token into env_path when it isn't already there.

    The server may resolve COWORK_AUTH_TOKEN from a real environment variable
    or ~/.anton/.env, but the desktop app only reads ~/.cowork/.env — write the
    resolved value there so both sides validate against the same secret.
    """
    if token and _read_token(env_path) != token:
        _write_token(env_path, token)


def remove_auth_token(env_path: Path) -> bool:
    """Remove every COWORK_AUTH_TOKEN line from env_path, keeping the other
    lines as they are. Returns whether one was there.

    For a deployment whose token must not sit in cowork-home's .env, where an
    earlier start mirrored it (sync_auth_token). The kept lines go to a
    temporary file beside it, with the file's mode, flushed to disk and moved
    over it: a kill midway never leaves a torn .env, and a link or second
    name at env_path is replaced rather than written through.
    """
    if not env_path.exists():
        return False
    existing = env_path.read_text(encoding="utf-8")
    kept = _TOKEN_LINE_RE.sub("", existing)
    if kept == existing:
        return False
    mode = stat.S_IMODE(env_path.stat().st_mode)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=env_path.parent, prefix=".env-", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(kept)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, env_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    logger.info("auth: removed COWORK_AUTH_TOKEN from %s", env_path)
    return True


# ── private helpers ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"^COWORK_AUTH_TOKEN\s*=\s*(.+)$", re.MULTILINE)
# The whole line, as python-dotenv reads it: optional indent and `export`.
_TOKEN_LINE_RE = re.compile(r"^[ \t]*(?:export[ \t]+)?COWORK_AUTH_TOKEN[ \t]*=.*(?:\r?\n|\Z)", re.MULTILINE)


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

    # Open with O_CREAT and mode 0o600 so a freshly created file is private
    # from the start — never world-readable in the gap between write and chmod.
    # For a pre-existing file the mode arg is ignored, so still chmod to tighten
    # any looser permissions.
    fd = open_fd(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
