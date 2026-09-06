from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.api.v1.endpoints.connectors.connections import (
    validate_and_save_developer_connection,
)
from cowork.db.scoped import LOCAL_SCOPE
from cowork.schemas.connectors import DirectSaveRequest
from cowork.services.connectors.developer_validation import (
    DeveloperCredentialError,
    PinnedHostTransport,
    ValidatedDeveloperIdentity,
    validate_developer_connection,
)


def _refusing_network() -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: pytest.fail("network called"))


def test_github_token_is_verified_and_resolves_account_identity() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.github.com/user"
        assert request.headers["Authorization"] == "Bearer github_pat_secret"
        return httpx.Response(200, json={"login": "ian-mindsdb", "email": None})

    result = validate_developer_connection(
        "github",
        "fine-grained-pat",
        {"access_token": "github_pat_secret", "base_url": "https://github.com"},
        transport=httpx.MockTransport(handle),
    )

    assert result.account_email == "ian-mindsdb"


def test_github_enterprise_probe_connects_to_the_validated_address() -> None:
    def resolve(_host, _port, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://93.184.216.34/api/v3/user"
        assert request.headers["Host"] == "github.enterprise.example"
        assert request.extensions["sni_hostname"] == "github.enterprise.example"
        assert request.headers["Authorization"] == "Bearer github_pat_secret"
        return httpx.Response(200, json={"login": "ian-mindsdb"})

    result = validate_developer_connection(
        "github",
        "fine-grained-pat",
        {"access_token": "github_pat_secret", "base_url": "https://github.enterprise.example"},
        transport=httpx.MockTransport(handle),
        resolver=resolve,
    )

    assert result.account_email == "ian-mindsdb"


def test_pinned_enterprise_hosts_with_one_address_receive_separate_connection_pools() -> None:
    created: list[str] = []
    seen: list[tuple[str, str, str]] = []

    def factory() -> httpx.BaseTransport:
        pool = f"pool-{len(created) + 1}"
        created.append(pool)

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append((pool, request.headers["Host"], request.headers["Authorization"]))
            return httpx.Response(200, json={"pool": pool})

        return httpx.MockTransport(handle)

    def resolve(_host, _port, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    transport = PinnedHostTransport(_refusing_network(), resolve, pinned_transport_factory=factory)
    transport.pin("one.enterprise.example")
    transport.pin("two.enterprise.example")
    with transport.client(trust_env=False) as client:
        one = client.get("https://one.enterprise.example/api/v3/user", headers={"Authorization": "Bearer one"})
        two = client.get("https://two.enterprise.example/api/v3/user", headers={"Authorization": "Bearer two"})

    assert one.json() == {"pool": "pool-1"}
    assert two.json() == {"pool": "pool-2"}
    assert seen == [
        ("pool-1", "one.enterprise.example", "Bearer one"),
        ("pool-2", "two.enterprise.example", "Bearer two"),
    ]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://github.enterprise.example",
        "https://127.0.0.1:8443",
        "https://169.254.169.254",
        "https://10.0.0.5",
    ],
)
def test_github_enterprise_rejects_insecure_or_private_base_urls(base_url: str) -> None:
    with pytest.raises(DeveloperCredentialError):
        validate_developer_connection(
            "github",
            "fine-grained-pat",
            {"access_token": "github_pat_secret", "base_url": base_url},
            transport=_refusing_network(),
        )


def test_github_enterprise_resolves_every_address_before_sending_credentials() -> None:
    def resolve(_host, _port, **_kwargs):
        return [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]

    with pytest.raises(DeveloperCredentialError, match="publicly routable"):
        validate_developer_connection(
            "github",
            "fine-grained-pat",
            {"access_token": "github_pat_secret", "base_url": "https://github.enterprise.example"},
            transport=_refusing_network(),
            resolver=resolve,
        )


def test_linear_key_is_verified_and_resolves_account_identity() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.linear.app/graphql"
        assert request.headers["Authorization"] == "lin_api_secret"
        assert b"CodeConnectorViewer" in request.content
        return httpx.Response(200, json={"data": {"viewer": {"id": "u1", "name": "Ian", "email": "ian@mindsdb.com"}}})

    result = validate_developer_connection(
        "linear",
        "personal-api-key",
        {"api_key": "lin_api_secret"},
        transport=httpx.MockTransport(handle),
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
    with pytest.raises(DeveloperCredentialError):
        validate_developer_connection(provider, method, values, transport=_refusing_network())


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
