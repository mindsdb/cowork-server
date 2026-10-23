"""Signed draft-preview tokens.

One wire format for draft and published-artifact tokens (the published one
lives in mindshub_services ``content_token.py``)::

    v1.<payload>.<sig>

``payload`` is base64url without padding of the claims as
``json.dumps(claims, separators=(",", ":"), sort_keys=True)``; ``sig`` is
base64url without padding of HMAC-SHA256(secret, the ASCII string
``v1.<payload>``). Keys are a ``{kid: secret}`` map: the active kid signs, and
verification picks the key named by the token's own ``kid`` claim, so two keys
can be live during a rotation.

Only cowork-server issues and checks these. The Cloudflare worker in front of
the draft host never holds the key.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time

DRAFT_PURPOSE = "draft"
DRAFT_TTL_SECONDS = 1800

_VERSION = "v1"
_MAX_TOKEN_LENGTH = 4096
_IDENTITY_CLAIMS = ("sub", "org", "proj", "art")


class DraftTokenError(ValueError):
    """A token that must be treated as absent. ``reason`` is safe to log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(secret: str, signing_input: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256)
    return _b64encode(digest.digest())


def issue_draft_token(
    *,
    env: str,
    sub: str,
    org: str,
    proj: str,
    art: str,
    keys: dict[str, str],
    kid: str,
    now: int | None = None,
) -> str:
    """Sign a token for one user's view of one draft, valid for 30 minutes."""
    secret = keys.get(kid) if kid else None
    if not secret:
        raise DraftTokenError("kid")
    issued_at = int(time.time()) if now is None else int(now)
    claims = {
        "p": DRAFT_PURPOSE,
        "env": env,
        "sub": sub,
        "org": org,
        "proj": proj,
        "art": art,
        "exp": issued_at + DRAFT_TTL_SECONDS,
        "kid": kid,
    }
    payload = _b64encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{_VERSION}.{payload}"
    return f"{signing_input}.{_signature(secret, signing_input)}"


def verify_draft_token(
    token: str, *, env: str, keys: dict[str, str], now: int | None = None
) -> dict:
    """Return the claims of a valid draft token for ``env``, else raise.

    Checks, in order: shape, the key named by ``kid``, the signature, the
    purpose, the environment, the expiry and the identity claims.
    """
    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
        raise DraftTokenError("malformed")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _VERSION or not parts[1] or not parts[2]:
        raise DraftTokenError("malformed")
    try:
        claims = json.loads(_b64decode(parts[1]))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise DraftTokenError("malformed") from exc
    if not isinstance(claims, dict):
        raise DraftTokenError("malformed")
    kid = claims.get("kid")
    secret = keys.get(kid) if isinstance(kid, str) else None
    if not secret:
        raise DraftTokenError("kid")
    expected = _signature(secret, f"{parts[0]}.{parts[1]}")
    if not hmac.compare_digest(expected.encode("ascii"), parts[2].encode("utf-8")):
        raise DraftTokenError("signature")
    if claims.get("p") != DRAFT_PURPOSE:
        raise DraftTokenError("purpose")
    if claims.get("env") != env:
        raise DraftTokenError("env")
    exp = claims.get("exp")
    current = int(time.time()) if now is None else int(now)
    if isinstance(exp, bool) or not isinstance(exp, int) or exp <= current:
        raise DraftTokenError("expired")
    for name in _IDENTITY_CLAIMS:
        value = claims.get(name)
        if not isinstance(value, str) or not value:
            raise DraftTokenError("malformed")
    return claims
