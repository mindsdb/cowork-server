from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from jwt import PyJWKClient

from cowork.common.settings.app_settings import get_app_settings


@dataclass(frozen=True)
class RequestPrincipal:
    subject: str
    email: str | None
    name: str | None
    issuer: str
    claims: dict[str, Any]


class AuthenticationError(PermissionError):
    pass


def principal_from_authorization_header(authorization: str | None) -> RequestPrincipal | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Invalid authorization header")
    return principal_from_bearer_token(token.strip())


def principal_from_bearer_token(
    token: str,
    *,
    allowed_issuers: list[str] | tuple[str, ...] | None = None,
) -> RequestPrincipal:
    allowed = _allowed_issuers() if allowed_issuers is None else [item.rstrip("/") for item in allowed_issuers if item]
    if not allowed:
        raise AuthenticationError("Bearer authentication is not configured")
    try:
        unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid bearer token") from exc
    issuer = str(unverified.get("iss") or "").rstrip("/")
    if issuer not in allowed:
        raise AuthenticationError("Untrusted token issuer")
    try:
        signing_key = _jwks_client_for_issuer(issuer).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            issuer=issuer,
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid bearer token") from exc
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise AuthenticationError("Bearer token has no subject")
    return RequestPrincipal(
        subject=subject,
        email=_clean_email(claims.get("email") or claims.get("preferred_username")),
        name=_clean_name(claims.get("name") or claims.get("preferred_username")),
        issuer=issuer,
        claims=dict(claims),
    )


def _allowed_issuers() -> list[str]:
    raw = get_app_settings().auth_issuers or ""
    return [issuer.strip().rstrip("/") for issuer in raw.split(",") if issuer.strip()]


@lru_cache(maxsize=8)
def _jwks_client_for_issuer(issuer: str) -> PyJWKClient:
    return PyJWKClient(f"{issuer.rstrip('/')}/protocol/openid-connect/certs")


def _clean_email(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    return clean if "@" in clean else None


def _clean_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None
