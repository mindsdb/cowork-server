from __future__ import annotations

import pytest
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from cowork.services import request_identity
from cowork.services.request_identity import AuthenticationError, principal_from_bearer_token


def _token(*, issuer: str, key, email: str = " Ada@Example.COM ", subject: str = "user-1"):
    return jwt.encode(
        {
            "iss": issuer,
            "sub": subject,
            "email": email,
            "name": "Ada Lovelace",
        },
        key,
        algorithm="RS256",
        headers={"kid": "test"},
    )


def test_principal_from_bearer_token_verifies_trusted_issuer(monkeypatch: pytest.MonkeyPatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = "https://auth.example.com/auth/realms/mindsdb"

    class FakeSigningKey:
        key = private_key.public_key()

    class FakeJwksClient:
        def get_signing_key_from_jwt(self, token):
            return FakeSigningKey()

    monkeypatch.setattr(request_identity, "_jwks_client_for_issuer", lambda value: FakeJwksClient())

    principal = principal_from_bearer_token(
        _token(issuer=issuer, key=private_key),
        allowed_issuers=[issuer],
    )

    assert principal.subject == "user-1"
    assert principal.email == "ada@example.com"
    assert principal.name == "Ada Lovelace"
    assert principal.issuer == issuer


def test_principal_from_bearer_token_rejects_untrusted_issuer(monkeypatch: pytest.MonkeyPatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with pytest.raises(AuthenticationError):
        principal_from_bearer_token(
            _token(issuer="https://evil.example.com/auth/realms/mindsdb", key=private_key),
            allowed_issuers=["https://auth.example.com/auth/realms/mindsdb"],
        )
