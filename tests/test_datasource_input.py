"""Datasource input parsing and normalization before the relay to auth.

The rules mirror auth's canonicalize. What matters here is that a rejection
never carries the value that caused it: these payloads are passwords, DSNs
and CA material.
"""
from __future__ import annotations

import pytest

from cowork.schemas.connectors import (
    DatasourceConnectionResponse,
    DatasourceCreateRequest,
    DatasourceEditRequest,
)
from cowork.services.connectors.datasources import (
    InvalidDatasourceInput,
    normalize_datasource_input,
)

PASSWORD = "s3ntinel-p4ssw0rd-do-not-echo"
DSN = f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?sslmode=verify-full"

CA_PEM = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJAJ5sentinelCAblockX\n"
    "-----END CERTIFICATE-----"
)


def _structured(**overrides):
    body = {
        "connector_id": "postgres",
        "method": "host-port",
        "name": "prod reporting",
        "host": "db.example.com",
        "port": 5432,
        "database": "appdb",
        "username": "dbuser",
        "password": PASSWORD,
    }
    body.update(overrides)
    return DatasourceCreateRequest(**body)


def test_structured_input_becomes_the_canonical_auth_payload():
    payload = normalize_datasource_input(_structured())

    assert payload == {
        "connector_id": "postgres",
        "method": "host-port",
        "name": "prod reporting",
        "host": "db.example.com",
        "port": 5432,
        "database": "appdb",
        "username": "dbuser",
        "password": PASSWORD,
        "tls": {"mode": "system", "ca_pem": None},
    }


def test_dsn_input_is_parsed_into_fields_and_the_dsn_is_dropped():
    payload = normalize_datasource_input(
        DatasourceCreateRequest(
            connector_id="postgres",
            method="host-port",
            name="prod reporting",
            input_mode="dsn",
            dsn=DSN,
        )
    )

    assert payload["host"] == "db.example.com"
    assert payload["port"] == 5432
    assert payload["database"] == "appdb"
    assert payload["username"] == "dbuser"
    assert payload["password"] == PASSWORD
    # The raw DSN must not survive into what gets relayed.
    assert "dsn" not in payload
    assert "input_mode" not in payload
    assert DSN not in str(payload)


def test_custom_ca_is_carried_through_and_system_mode_rejects_a_ca():
    payload = normalize_datasource_input(
        _structured(tls={"mode": "custom_ca", "ca_pem": CA_PEM})
    )
    assert payload["tls"] == {"mode": "custom_ca", "ca_pem": CA_PEM}

    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(tls={"mode": "system", "ca_pem": CA_PEM}))


@pytest.mark.parametrize(
    "dsn",
    [
        f"postgres://dbuser:{PASSWORD}@db1.example.com,db2.example.com:5432/appdb?sslmode=verify-full",
        f"postgres://dbuser:{PASSWORD}@/var/run/postgresql/appdb?sslmode=verify-full",
        f"postgres://dbuser:{PASSWORD}@[2001:db8::1]:5432/appdb?sslmode=verify-full",
        f"postgres://dbuser@db.example.com:5432/appdb?sslmode=verify-full",
        f"postgres://dbuser:{PASSWORD}@db.example.com:5432/?sslmode=verify-full",
        f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?sslmode=require",
        f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?sslmode=verify-full&sslkey=/tmp/k",
        f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?options=-c%20search_path%3Dx",
        f"mysql://dbuser:{PASSWORD}@db.example.com:3306/appdb?local_infile=1",
        f"redis://dbuser:{PASSWORD}@db.example.com:5432/appdb",
    ],
)
def test_rejected_dsns_never_echo_the_dsn_or_the_password(dsn):
    connector_id = "mysql" if dsn.startswith("mysql") else "postgres"
    method = "host-password" if connector_id == "mysql" else "host-port"
    model = DatasourceCreateRequest(
        connector_id=connector_id,
        method=method,
        name="prod reporting",
        input_mode="dsn",
        dsn=dsn,
    )

    with pytest.raises(InvalidDatasourceInput) as caught:
        normalize_datasource_input(model)

    message = caught.value.detail["message"]
    assert caught.value.detail["code"] == "invalid_connection"
    assert PASSWORD not in message
    assert dsn not in message


def test_dsn_mode_refuses_structured_fields_alongside_the_dsn():
    model = DatasourceCreateRequest(
        connector_id="postgres",
        method="host-port",
        name="prod reporting",
        input_mode="dsn",
        dsn=DSN,
        host="other.example.com",
    )

    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(model)


def test_structured_mode_refuses_a_dsn():
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(input_mode="structured", dsn=DSN))


def test_a_non_default_port_is_refused_before_relay():
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(port=6543))


def test_unsupported_connector_and_method_are_refused():
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(connector_id="oracle"))
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(method="connection-string"))


def test_an_oversized_ca_is_refused_without_echoing_it():
    oversized = "-----BEGIN CERTIFICATE-----\n" + ("A" * (64 * 1024)) + "\n-----END CERTIFICATE-----"

    with pytest.raises(InvalidDatasourceInput) as caught:
        normalize_datasource_input(_structured(tls={"mode": "custom_ca", "ca_pem": oversized}))

    assert oversized not in caught.value.detail["message"]


def test_an_unknown_key_is_rejected_by_the_schema():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        DatasourceCreateRequest(
            connector_id="postgres",
            method="host-port",
            name="prod reporting",
            host="db.example.com",
            database="appdb",
            username="dbuser",
            password=PASSWORD,
            sslkey="/tmp/key",
        )


def test_edit_requires_a_positive_expected_version():
    import pydantic

    edit = DatasourceEditRequest(
        connector_id="postgres",
        method="host-port",
        name="prod reporting",
        host="db.example.com",
        port=5432,
        database="appdb",
        username="dbuser",
        password=PASSWORD,
        expected_version=3,
    )
    assert edit.expected_version == 3
    assert normalize_datasource_input(edit)["host"] == "db.example.com"

    with pytest.raises(pydantic.ValidationError):
        DatasourceEditRequest(
            connector_id="postgres",
            method="host-port",
            name="prod reporting",
            host="db.example.com",
            database="appdb",
            username="dbuser",
            password=PASSWORD,
            expected_version=0,
        )


def test_the_response_model_drops_an_auth_field_this_server_does_not_know():
    response = DatasourceConnectionResponse.model_validate(
        {
            "id": 7,
            "connector_id": "postgres",
            "method": "host-port",
            "name": "prod reporting",
            "status": "pending",
            "credential_version": 1,
            "host_masked": "db***om",
            "port": 5432,
            "database": "appdb",
            "username": "dbuser",
            "tls_mode": "system",
            "validation_error": None,
            "created_at": "2026-09-17T10:00:00Z",
            "updated_at": "2026-09-17T10:00:00Z",
            "validation_attempt_id": "should-not-survive",
            "password": PASSWORD,
        }
    )

    dumped = response.model_dump()
    assert "validation_attempt_id" not in dumped
    assert "password" not in dumped
    assert PASSWORD not in str(dumped)
