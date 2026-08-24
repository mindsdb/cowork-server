from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from anton.core.datasources.data_vault import LocalDataVault
from cowork.api.v1.endpoints.connectors.connections import validate_and_save_developer_connection
from cowork.db.scoped import LOCAL_SCOPE
from cowork.schemas.connectors import DirectSaveRequest
from cowork.services.connectors.developer_validation import (
    DeveloperCredentialError,
    ValidatedDeveloperIdentity,
    validate_developer_connection,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_github_token_is_verified_and_resolves_account_identity() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.github.com/user"
        assert request.headers["Authorization"] == "Bearer github_pat_secret"
        return httpx.Response(200, json={"login": "ian-mindsdb", "email": None})

    with _client(handle) as client:
        result = validate_developer_connection(
            "github",
            "fine-grained-pat",
            {"access_token": "github_pat_secret", "base_url": "https://github.com"},
            client=client,
        )

    assert result.account_email == "ian-mindsdb"


def test_linear_key_is_verified_and_resolves_account_identity() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.linear.app/graphql"
        assert request.headers["Authorization"] == "lin_api_secret"
        assert b"CodeConnectorViewer" in request.content
        return httpx.Response(200, json={"data": {"viewer": {"id": "u1", "name": "Ian", "email": "ian@mindsdb.com"}}})

    with _client(handle) as client:
        result = validate_developer_connection(
            "linear",
            "personal-api-key",
            {"api_key": "lin_api_secret"},
            client=client,
        )

    assert result.account_email == "ian@mindsdb.com"


@pytest.mark.parametrize(
    ("provider", "method", "values"),
    [
        ("github", "fine-grained-pat", {"access_token": ""}),
        ("linear", "personal-api-key", {"api_key": ""}),
        ("github", "personal-api-key", {"access_token": "token"}),
    ],
)
def test_missing_or_unsupported_credentials_fail_before_network(provider, method, values) -> None:
    with _client(lambda _request: pytest.fail("network called")) as client:
        with pytest.raises(DeveloperCredentialError):
            validate_developer_connection(provider, method, values, client=client)


def test_validate_and_save_augments_the_vault_record_with_verified_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
        lambda: type("S", (), {"vault_dir": str(tmp_path / "vault")})(),
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.connectors.connections.validate_developer_connection",
        lambda *_args, **_kwargs: ValidatedDeveloperIdentity(account_email="ian-mindsdb"),
    )

    result = validate_and_save_developer_connection(
        DirectSaveRequest(
            connector_id="github",
            method="fine-grained-pat",
            values={"access_token": "github_pat_secret", "base_url": "https://github.com"},
        ),
        LOCAL_SCOPE,
    )

    assert result["ok"] is True
    assert result["name"] == "ian-mindsdb"


def test_validated_reconnect_replaces_the_named_record_without_creating_a_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
        lambda: type("S", (), {"vault_dir": str(vault_path)})(),
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.connectors.connections.validate_developer_connection",
        lambda *_args, **_kwargs: ValidatedDeveloperIdentity(account_email="ian-mindsdb"),
    )
    vault = LocalDataVault(vault_path)
    vault.save(
        "github",
        "ian-mindsdb",
        {
            "_connector_id": "github",
            "_method": "browser_oauth_builtin",
            "access_token": "expired",
            "account_email": "ian-mindsdb",
            "status": "needs_reconnect",
        },
        secure_keys=["access_token"],
    )

    result = validate_and_save_developer_connection(
        DirectSaveRequest(
            connector_id="github",
            method="fine-grained-pat",
            name="ian-mindsdb",
            replace_existing=True,
            values={"access_token": "github_pat_new", "base_url": "https://github.com"},
        ),
        LOCAL_SCOPE,
    )

    assert result["name"] == "ian-mindsdb"
    assert [item["name"] for item in vault.list_connections()] == ["ian-mindsdb"]
    record = vault.read_record("github", "ian-mindsdb")
    assert record is not None
    assert record["fields"]["access_token"] == "github_pat_new"
    assert "status" not in record["fields"]
