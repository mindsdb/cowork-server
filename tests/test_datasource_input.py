"""Datasource input parsing and normalization before the relay to auth.

These rules are a subset of auth's canonicalize in places (the TLS block is
framing-checked here, parsed as X.509 there) and stricter in others (auth
accepts 127.0.0.1 as a host; this refuses it). What matters throughout is
that a rejection never carries the value that caused it: these payloads are
passwords, DSNs and CA material.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

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
        "schema": None,
        "username": "dbuser",
        "password": PASSWORD,
        "tls": {"mode": "prefer", "ca_pem": None},
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


def test_the_schema_a_connection_names_is_relayed():
    """A PostgreSQL database holds many schemas and the customer's tables are
    rarely in the one the role's search path resolves to."""
    payload = normalize_datasource_input(_structured(schema="sales_ops"))

    assert payload["schema"] == "sales_ops"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_blank_schema_is_no_schema_rather_than_a_bad_one(value):
    """A field the user left blank must not be refused for its length."""
    assert normalize_datasource_input(_structured(schema=value))["schema"] is None


def test_a_schema_is_refused_on_an_engine_that_has_none():
    """MySQL's database is its schema; two names for one thing would leave a
    reader unable to say which won."""
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(
            _structured(connector_id="mysql", method="host-password", schema="sales_ops")
        )


def test_a_schema_longer_than_an_identifier_is_refused():
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(schema="s" * 64))


def test_a_connection_string_cannot_carry_a_schema():
    """The DSN grammar accepts no schema, so one beside it is a field the
    parsed connection would silently disagree with."""
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(
            DatasourceCreateRequest(
                connector_id="postgres",
                method="host-port",
                name="prod reporting",
                input_mode="dsn",
                dsn=DSN,
                schema="sales_ops",
            )
        )


def test_custom_ca_is_carried_through_and_system_mode_rejects_a_ca():
    payload = normalize_datasource_input(
        _structured(tls={"mode": "custom_ca", "ca_pem": CA_PEM})
    )
    assert payload["tls"] == {"mode": "custom_ca", "ca_pem": CA_PEM}

    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(tls={"mode": "system", "ca_pem": CA_PEM}))


def test_a_connection_with_no_trust_block_prefers_encryption():
    """The form stops asking, so this is what nearly every connection carries:
    encryption where the server offers it, and no check on who answered."""
    payload = normalize_datasource_input(_structured(tls=None))

    assert payload["tls"] == {"mode": "prefer", "ca_pem": None}


@pytest.mark.parametrize("mode", ["encrypted", "disabled", "prefer"])
def test_a_mode_that_does_not_verify_is_relayed_as_chosen(mode):
    """A self-hosted server often has an unverifiable certificate or none, and
    the owner says so on the connection rather than being turned away."""
    payload = normalize_datasource_input(_structured(tls={"mode": mode, "ca_pem": None}))

    assert payload["tls"] == {"mode": mode, "ca_pem": None}


@pytest.mark.parametrize("mode", ["encrypted", "disabled", "prefer"])
def test_a_bundle_on_a_mode_that_never_reads_one_is_refused(mode):
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(tls={"mode": mode, "ca_pem": CA_PEM}))


@pytest.mark.parametrize("mode", ["verify-full", "require", "allow", "off"])
def test_a_drivers_own_spelling_is_not_a_mode(mode):
    with pytest.raises(ValidationError):
        _structured(tls={"mode": mode, "ca_pem": None})


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


@pytest.mark.parametrize("port", [6543, 6432, 25060])
def test_a_managed_pooler_port_is_accepted(port):
    """Auth stores any port in 1..65535, and these are real managed endpoints.

    Whether a release can execute against them is the capability policy's
    call, not something to refuse at the storage boundary.
    """
    assert normalize_datasource_input(_structured(port=port))["port"] == port


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_a_port_outside_the_valid_range_is_refused_before_relay(port):
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(port=port))


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "169.254.169.254",
        "localhost",
        # The resolver accepts these spellings too: inet_aton reads them as
        # 169.254.169.254, 169.254.169.254, 127.0.0.1 and 127.0.0.1.
        "2852039166",
        "0xa9fea9fe",
        "127.1",
        "127.000.000.001",
        "0.0.0.0",
        "224.0.0.1",
        "255.255.255.255",
    ],
)
def test_hosts_only_the_pod_can_reach_are_refused(host):
    """The stored host is dialed from a cluster pod later, so the metadata
    service and neighbouring services must not be reachable through it, in
    any spelling that the resolver would accept."""
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(host=host))


def test_an_address_literal_is_stored_in_its_canonical_form():
    """Whatever is relayed has to be what was checked, not another spelling."""
    assert normalize_datasource_input(_structured(host="010.000.000.005"))["host"] == "8.0.0.5"


@pytest.mark.parametrize("host", ["db.example.com", "1password.example.com", "xn--r8jz45g.xn--zckzah"])
def test_ordinary_hostnames_are_untouched(host):
    assert normalize_datasource_input(_structured(host=host))["host"] == host


@pytest.mark.parametrize(
    "connector_id,method,expected", [("postgres", "host-port", 5432), ("mysql", "host-password", 3306)]
)
def test_an_omitted_port_falls_back_to_the_connector_default(connector_id, method, expected):
    payload = normalize_datasource_input(
        _structured(connector_id=connector_id, method=method, port=None)
    )
    assert payload["port"] == expected


def test_a_vpc_private_address_is_still_accepted():
    assert normalize_datasource_input(_structured(host="10.0.0.5"))["host"] == "10.0.0.5"


def test_a_name_that_is_only_whitespace_is_refused_before_relay():
    """It passes min_length=1 but strips to empty, and auth answers a blank
    name with a serializer error that carries no code to relay."""
    with pytest.raises(InvalidDatasourceInput):
        normalize_datasource_input(_structured(name="   "))


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
            "schema": "sales_ops",
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
    # `schema` on the wire in both directions: the field is renamed only
    # because the name shadows an attribute of BaseModel.
    assert response.db_schema == "sales_ops"
    assert response.model_dump(by_alias=True)["schema"] == "sales_ops"
