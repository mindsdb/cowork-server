"""Cloud datasource submissions in org mode are relayed, never stored locally.

A cloud submission carries a database password. It must reach auth's
encrypted store and nothing else: no staging store, no probe, no local vault,
on the success path and on every refusal. Each org case runs with those paths
booby-trapped, so a fallback that quietly saves the credential fails the test
rather than passing it. A connector with no cloud datasource form takes the
ordinary submission flow in org mode too, and never reaches the relay.

The desktop cases at the bottom pin the other half: local mode still stages
and probes exactly as before.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app
from tests._connector_test_request import with_test_request

PATH = "/api/v1/connectors/submissions/"
BEARER = "Bearer sentinel-caller-credential"
PASSWORD = "s3ntinel-p4ssw0rd-do-not-echo"
CA_PEM = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBsentinelcertificatebodythatisnotrealx509material\n"
    "-----END CERTIFICATE-----"
)
ENABLED = '{"manifest_version": 1, "enabled": ["postgres:host-port"]}'

MEMBER_HEADERS = {
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}
AUTH_HEADERS = {**MEMBER_HEADERS, "Authorization": BEARER}

#: A model-written form that passes check_handcrafted_form.
HANDCRAFTED_FORM = with_test_request({
    "form_id": "fm_bbbbbbbbbb",
    "engine": "handcrafted",
    "title": "Connect handcrafted",
    "fields": [{"name": "token", "label": "Token", "type": "password", "secret": True, "required": True}],
})

#: Shaped like auth's DatasourceConnectionResponseSerializer.
CONNECTION = {
    "id": 7,
    "connector_id": "postgres",
    "method": "host-port",
    "name": "prod reporting",
    "status": "pending",
    "credential_version": 1,
    "revision": 6,
    "host_masked": "db***om",
    "port": 5432,
    "database": "appdb",
    "username": "dbuser",
    "tls_mode": "system",
    "validation_error": None,
    "created_at": "2026-09-17T10:00:00Z",
    "updated_at": "2026-09-17T10:00:00Z",
}


def submission(**overrides) -> dict:
    body = {
        "connector_id": "postgres",
        "method": "host-port",
        "name": "prod reporting",
        "values": {
            "host": "db.example.com",
            "port": "5432",
            "database": "appdb",
            "username": "dbuser",
            "password": PASSWORD,
        },
        "skipped": [],
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def forbid_local_paths(monkeypatch):
    """Booby-trap every local credential path a cloud submission must not reach."""
    from anton.core.datasources import data_vault
    from cowork.handlers import probe as probe_module
    from cowork.services.connectors import connections as connections_module
    from cowork.services.connectors import persist as persist_module
    from cowork.services.connectors.submissions import store

    def forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"a cloud submission reached the local path {name}")

        return _raise

    monkeypatch.setattr(store, "stage", forbidden("store.stage"))
    monkeypatch.setattr(persist_module, "vault_for_scope", forbidden("persist.vault_for_scope"))
    monkeypatch.setattr(probe_module, "vault_for_scope", forbidden("probe.vault_for_scope"))
    monkeypatch.setattr(probe_module, "CredentialProbe", forbidden("CredentialProbe"))
    monkeypatch.setattr(data_vault.LocalDataVault, "__init__", forbidden("LocalDataVault"))
    monkeypatch.setattr(connections_module.ConnectionsService, "__init__", forbidden("ConnectionsService"))

    before = set(store._store)
    yield
    assert set(store._store) == before, "a cloud submission left something in the staging store"


@pytest.fixture()
def relay(monkeypatch):
    """Record what is relayed to auth and script auth's answer."""
    from cowork.services.connectors.oauth import auth_proxy

    recorded: list[httpx.Request] = []
    scripted: dict = {"status": 201, "body": CONNECTION, "error": None}

    def handle(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if scripted["error"]:
            raise scripted["error"]
        return httpx.Response(scripted["status"], json=scripted["body"])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        auth_proxy.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    return recorded, scripted


def _org_client(monkeypatch, capabilities: str) -> TestClient:
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", capabilities)
    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def org_client(monkeypatch, adapter_verified_datasources, forbid_local_paths) -> TestClient:
    """Org mode with PostgreSQL enabled and the adapters reported as verified."""
    return _org_client(monkeypatch, ENABLED)


@pytest.fixture()
def default_org_client(monkeypatch, adapter_verified_datasources, forbid_local_paths) -> TestClient:
    """Org mode as a deployment gets it before anyone enables anything."""
    return _org_client(monkeypatch, "")


@pytest.fixture()
def unverified_org_client(monkeypatch, adapter_unverified_datasources, forbid_local_paths) -> TestClient:
    """Org mode with the pair enabled but the adapters reported as unverified."""
    return _org_client(monkeypatch, ENABLED)


@pytest.fixture()
def local_client(monkeypatch) -> TestClient:
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    return TestClient(create_app())


def test_the_route_stays_declared_for_org_members():
    """A whole-route desktop refusal would make every case below unreachable.

    The org branch is the refusal now, per method rather than per route, so a
    change that declares this router desktop-only again would turn a relayed
    submission into a 403 without failing anything else.
    """
    from cowork.api.v1.permissions import AuthenticatedInOrgMode
    from cowork.api.v1.route_walker import declared_permissions
    from cowork.server import create_app

    app = create_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == PATH)

    assert declared_permissions(route) == [AuthenticatedInOrgMode]


def test_an_enabled_cloud_submission_is_relayed_and_never_staged(org_client, relay, caplog):
    recorded, _ = relay

    with caplog.at_level(logging.DEBUG):
        res = org_client.post(PATH, json=submission(), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["cache-control"] == "no-store"
    assert len(recorded) == 1
    sent = recorded[0]
    assert str(sent.url) == "https://auth.example.com/v1/datasources"
    assert sent.headers["authorization"] == BEARER
    body = json.loads(sent.content)
    assert body["connector_id"] == "postgres"
    assert body["method"] == "host-port"
    assert body["name"] == "prod reporting"
    assert body["host"] == "db.example.com"
    assert body["port"] == 5432
    assert body["database"] == "appdb"
    # The schema is optional and this submission names none.
    assert body["schema"] is None
    assert body["username"] == "dbuser"
    assert body["password"] == PASSWORD
    # The form asks nothing about certificates, so the relay sends no choice
    # and the server applies its own.
    assert body["tls"] == {"mode": "prefer", "ca_pem": None}
    # Owner and org are auth's to derive from the bearer.
    assert "user_id" not in body and "organization_id" not in body
    # What the submitter sees back is auth's masked metadata, not their input.
    assert PASSWORD not in res.text
    assert PASSWORD not in caplog.text
    assert "prod reporting" in res.text
    assert "response.completed" in res.text


def test_the_schema_a_submitter_names_reaches_auth(org_client, relay):
    """The one optional field on the PostgreSQL form. Without it the agent
    reads whatever the role's search path resolves to, which is rarely where
    the customer's tables are."""
    recorded, _ = relay
    values = submission()["values"] | {"schema": "sales_ops"}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["schema"] == "sales_ops"


def test_asking_for_a_verified_certificate_relays_that_choice(org_client, relay):
    """The form's one trust question. Unchecked it is absent and the server
    applies its own default, which is the case just above."""
    recorded, _ = relay
    values = submission()["values"] | {"tls_verify": True}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["tls"] == {"mode": "system", "ca_pem": None}


def test_a_checked_box_counts_even_when_it_arrives_as_a_string(org_client, relay):
    """The spec's boolean default reaches the form as JSON, so a checked box
    can submit `"true"`. Reading only the literal stored the connection
    unverified while the form showed the box checked."""
    recorded, _ = relay
    values = submission()["values"] | {"tls_verify": "true"}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["tls"] == {"mode": "system", "ca_pem": None}


def test_an_unchecked_box_is_not_a_trust_choice(org_client, relay):
    recorded, _ = relay
    values = submission()["values"] | {"tls_verify": False}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["tls"] == {"mode": "prefer", "ca_pem": None}


def test_a_field_the_cloud_form_no_longer_has_is_refused(org_client, relay):
    """Certificate trust left the form, so a submission naming it is a client
    sending something this connector does not collect."""
    recorded, _ = relay
    values = submission()["values"] | {"tls_mode": "custom_ca", "ca_pem": CA_PEM}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_connection"
    assert recorded == []


@pytest.fixture()
def mysql_org_client(monkeypatch, adapter_verified_datasources, forbid_local_paths) -> TestClient:
    """Org mode with MySQL enabled and the adapters reported as verified."""
    return _org_client(monkeypatch, '{"manifest_version": 1, "enabled": ["mysql:host-password"]}')


def mysql_submission(**trust) -> dict:
    """A MySQL cloud submission with the given trust fields added to its values."""
    values = {
        "host": "db.example.com",
        "port": "3306",
        "database": "appdb",
        "username": "dbuser",
        "password": PASSWORD,
        **trust,
    }
    return submission(connector_id="mysql", method="host-password", values=values)


@pytest.mark.parametrize(
    ("trust", "stored"),
    [
        pytest.param({"tls_mode": "prefer"}, {"mode": "prefer", "ca_pem": None}, id="prefer"),
        pytest.param({"tls_mode": "system"}, {"mode": "system", "ca_pem": None}, id="system"),
        pytest.param(
            {"tls_mode": "custom_ca", "ca_pem": CA_PEM}, {"mode": "custom_ca", "ca_pem": CA_PEM}, id="custom_ca"
        ),
    ],
)
def test_a_mysql_trust_choice_is_relayed_as_chosen(mysql_org_client, relay, trust, stored):
    """MySQL's form chooses a mode, and a server signed by its own authority
    gets in by pasting that authority, since the gateway sends a MySQL 8
    password only to a server it verified."""
    recorded, _ = relay

    res = mysql_org_client.post(PATH, json=mysql_submission(**trust), headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["tls"] == stored


@pytest.mark.parametrize("mode", ["prefer", "system"])
def test_a_ca_left_in_the_box_under_a_mode_that_reads_none_is_refused(mysql_org_client, relay, mode):
    """The app sends the box only for `custom_ca`; this path forwards it, and a
    row that names one trust while carrying another bundle is refused."""
    recorded, _ = relay

    res = mysql_org_client.post(PATH, json=mysql_submission(tls_mode=mode, ca_pem=CA_PEM), headers=AUTH_HEADERS)

    assert res.status_code == 400
    assert f"{mode} TLS cannot include a CA certificate" in res.text
    assert PASSWORD not in res.text
    assert recorded == []


def test_mysql_no_longer_takes_the_certificate_checkbox(mysql_org_client, relay):
    """The checkbox left MySQL's form; PostgreSQL keeps it (tested above).

    The same submission without the checkbox is accepted first, so the
    refusal can only be the checkbox's.
    """
    recorded, _ = relay
    accepted = mysql_org_client.post(PATH, json=mysql_submission(tls_mode="prefer"), headers=AUTH_HEADERS)
    assert accepted.status_code == 200
    recorded.clear()

    res = mysql_org_client.post(
        PATH, json=mysql_submission(tls_mode="prefer", tls_verify=True), headers=AUTH_HEADERS
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_connection"
    assert "accepts only its cloud connection fields" in res.text
    assert recorded == []


def test_the_default_deployment_refuses_the_method_without_calling_auth(default_org_client, relay):
    recorded, _ = relay

    res = default_org_client.post(PATH, json=submission(), headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert PASSWORD not in res.text
    assert recorded == [], "a method this deployment has not enabled must not reach auth"


def test_a_method_the_specs_call_unverified_is_refused_however_it_is_configured(unverified_org_client, relay):
    recorded, _ = relay

    res = unverified_org_client.post(PATH, json=submission(), headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert recorded == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(submission(method="connection-string"), id="connection-string"),
        pytest.param(submission(method=None), id="no-method"),
    ],
)
def test_a_datasource_submission_that_is_not_an_enabled_cloud_method_is_refused(org_client, relay, body):
    recorded, _ = relay

    res = org_client.post(PATH, json=body, headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert PASSWORD not in res.text
    assert recorded == []


@pytest.mark.parametrize(
    ("connector_id", "method"),
    [("aws_rds_postgres", "iam-auth"), ("snowflake", "user-password"), ("bigquery", None)],
)
def test_a_built_in_database_without_a_cloud_form_is_refused_not_saved(org_client, relay, connector_id, method):
    recorded, _ = relay
    body = submission(connector_id=connector_id, method=method)

    res = org_client.post(PATH, json=body, headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert PASSWORD not in res.text
    assert recorded == []


def test_a_managed_service_submission_is_relayed_as_its_engine(managed_postgres_service, org_client, relay):
    recorded, _ = relay
    connector_id, method = managed_postgres_service

    res = org_client.post(PATH, json=submission(connector_id=connector_id, method=method), headers=AUTH_HEADERS)

    assert res.status_code == 200
    body = json.loads(recorded[0].content)
    assert (body["connector_id"], body["method"]) == ("postgres", "host-port")
    assert PASSWORD not in res.text
    assert "managed_pg" in res.text


def test_the_real_rds_spec_is_relayed_as_postgres(org_client, relay):
    recorded, _ = relay

    res = org_client.post(
        PATH, json=submission(connector_id="aws_rds_postgres", method="connection-string"), headers=AUTH_HEADERS
    )

    assert res.status_code == 200
    body = json.loads(recorded[0].content)
    assert (body["connector_id"], body["method"]) == ("postgres", "host-port")
    assert PASSWORD not in res.text


@pytest.mark.parametrize(
    ("top_level", "form_spec"),
    [
        ({}, {"engine": "postgres"}),
        ({}, {"_connector_id": "postgres"}),
        ({"form_id": "postgres-connector"}, {}),
    ],
    ids=["engine", "connector-id", "form-id"],
)
def test_a_handcrafted_form_naming_postgres_still_goes_to_auth(org_client, relay, top_level, form_spec):
    recorded, _ = relay
    body = {**submission(), **top_level, "form_spec": {"form_id": "x", **form_spec}}
    body.pop("connector_id")

    res = org_client.post(PATH, json=body, headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert json.loads(recorded[0].content)["connector_id"] == "postgres"


@pytest.fixture()
def org_client_with_local_flow(monkeypatch, adapter_verified_datasources) -> TestClient:
    """Org mode with PostgreSQL enabled and the ordinary submission flow left intact."""
    return _org_client(monkeypatch, ENABLED)


@pytest.fixture()
def staged_in_org(monkeypatch):
    """What reaches the submission store, and a probe that must never run."""
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    staged: list[dict] = []
    monkeypatch.setattr(store, "stage", lambda **kwargs: staged.append(kwargs) or "sub_x")

    class NoProbe:
        def __init__(self, *, scope):
            raise AssertionError("an org-mode custom form must not reach the probe")

    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", NoProbe)
    return staged


GRANT_FORM = {
    "form_id": "fm_cccccccccc", "engine": "handcrafted", "title": "Connect handcrafted",
    "methods": [{
        "id": "oauth", "label": "Sign in", "submit_action": "oauth_launch",
        "oauth": {"auth_url": "https://example.com/auth", "token_url": "https://example.com/token"},
    }],
}


@pytest.mark.parametrize(("form", "values", "method"), [
    (HANDCRAFTED_FORM, {"token": PASSWORD}, None),
    (GRANT_FORM, {"access_token": PASSWORD}, "oauth"),
], ids=["handcrafted", "handcrafted oauth grant"])
def test_an_org_custom_form_is_refused_before_it_is_staged(org_client_with_local_flow, relay, staged_in_org, form, values, method):
    """Nothing tests a custom connector in an organization yet, so it is
    refused before its values are checked or kept; a grant is not saved first."""
    recorded, _ = relay

    res = org_client_with_local_flow.post(
        PATH,
        json={"form_id": form["form_id"], "name": "handcrafted", "form_spec": form, "values": values,
              "skipped": [], "method": method},
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 403
    assert "Custom connectors cannot be connected in an organization yet" in res.json()["detail"]
    assert PASSWORD not in res.text
    assert staged_in_org == [] and recorded == []


def test_an_org_stored_custom_connector_is_refused_before_it_is_staged(org_client_with_local_flow, relay, staged_in_org, monkeypatch):
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.schemas.connectors import ConnectorSpecResponse

    stored = ConnectorSpecResponse(
        id="handcrafted", label="Handcrafted", description="", category="other",
        form={k: v for k, v in HANDCRAFTED_FORM.items() if k != "engine"} | {"form_id": "handcrafted-connector"},
    )

    class Catalog:
        def __init__(self, session):
            pass

        def get_connector(self, connector_id):
            return stored if connector_id == "handcrafted" else None

    monkeypatch.setattr(submissions_endpoint, "ConnectorCatalog", Catalog)
    recorded, _ = relay

    res = org_client_with_local_flow.post(
        PATH,
        json={"form_id": "handcrafted-connector", "connector_id": "handcrafted", "name": "handcrafted",
              "form_spec": {"_connector_id": "handcrafted", "form_id": "handcrafted-connector"},
              "values": {"token": PASSWORD}, "skipped": []},
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 403
    assert staged_in_org == [] and recorded == []


def test_an_org_built_in_outside_the_database_category_takes_the_submission_flow(
    org_client_with_local_flow, relay, monkeypatch
):
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    recorded, _ = relay
    staged: dict = {}
    real_stage = store.stage

    def record_stage(**kwargs):
        staged.update(kwargs)
        return real_stage(**kwargs)

    class FakeProbeHandler:
        def __init__(self, *, scope):
            pass

        async def run(self, *args):
            yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(store, "stage", record_stage)
    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", FakeProbeHandler)
    values = {"api_key": PASSWORD, "application_key": "app-key", "region": "us1"}

    res = org_client_with_local_flow.post(
        PATH,
        json=submission(connector_id="datadog", method="api-key", values=values),
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 200
    assert staged["connector_id"] == "datadog"
    assert recorded == []


def test_an_org_connector_unknown_to_the_registry_is_not_relayed(org_client_with_local_flow, relay):
    recorded, _ = relay

    res = org_client_with_local_flow.post(PATH, json=submission(connector_id="not_a_connector"), headers=AUTH_HEADERS)

    assert res.status_code == 404
    assert PASSWORD not in res.text
    assert recorded == []


def test_a_field_the_cloud_form_does_not_have_is_refused_before_the_relay(org_client, relay):
    recorded, _ = relay
    values = submission()["values"] | {"sslkey": "/etc/keys/client.key"}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_connection"
    assert PASSWORD not in res.text
    assert recorded == []


def test_a_missing_password_is_named_and_nothing_is_relayed(org_client, relay):
    recorded, _ = relay
    values = {k: v for k, v in submission()["values"].items() if k != "password"}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["code"] == "invalid_connection"
    assert "password" in detail["message"]
    assert "username" not in detail["message"]
    assert recorded == []


def test_a_value_of_the_wrong_type_is_refused_without_echoing_it(org_client, relay, caplog):
    recorded, _ = relay
    values = submission()["values"] | {"port": {"nested": PASSWORD}}

    with caplog.at_level(logging.DEBUG):
        res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_connection"
    assert PASSWORD not in res.text
    assert PASSWORD not in caplog.text
    assert recorded == []


@pytest.mark.parametrize("status_code", [400, 409, 429, 503])
def test_auths_refusal_is_relayed_with_its_own_status(org_client, relay, status_code):
    _, scripted = relay
    scripted["status"] = status_code
    scripted["body"] = {"code": "invalid_connection", "detail": "The database refused the connection."}

    res = org_client.post(PATH, json=submission(), headers=AUTH_HEADERS)

    assert res.status_code == status_code
    assert PASSWORD not in res.text


def test_an_unreachable_auth_is_a_gateway_error_not_a_local_save(org_client, relay):
    _, scripted = relay
    scripted["error"] = httpx.ConnectError("auth is down")

    res = org_client.post(PATH, json=submission(), headers=AUTH_HEADERS)

    assert res.status_code == 502
    assert PASSWORD not in res.text


def test_a_malformed_body_answers_without_echoing_the_submission(org_client, relay):
    recorded, _ = relay

    res = org_client.post(PATH, json={"connector_id": "postgres", "values": PASSWORD}, headers=AUTH_HEADERS)

    assert res.status_code == 422
    assert PASSWORD not in res.text
    assert recorded == []


def test_an_org_request_without_identity_headers_never_reaches_auth(org_client, relay):
    recorded, _ = relay

    res = org_client.post(PATH, json=submission())

    assert res.status_code == 401
    assert recorded == []


def test_the_conversation_turn_is_written_through_the_request_scope(org_client, relay, monkeypatch):
    """The row belongs to auth's tenant; the turn belongs to this request's."""
    from cowork.db.scoped import ScopedSession
    from cowork.handlers import datasource_relay

    saved: dict = {}

    class FakeConversationService:
        def __init__(self, session):
            saved["session"] = session

        def get_conversation(self, conversation_id):
            saved["resolved"] = conversation_id
            return type("Conversation", (), {"id": conversation_id})()

        def save_assistant_turn(self, conversation_id, text, events):
            saved["text"] = text
            saved["events"] = events

    monkeypatch.setattr(datasource_relay, "ConversationService", FakeConversationService)

    res = org_client.post(
        PATH,
        json=submission(conversation_id="33333333-3333-4333-8333-333333333333"),
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 200
    assert isinstance(saved["session"], ScopedSession)
    assert str(saved["session"].scope.org_id) == MEMBER_HEADERS["X-Organization-Id"]
    assert PASSWORD not in saved["text"]
    assert PASSWORD not in json.dumps(saved["events"])
    assert "prod reporting" in saved["text"]


def test_a_desktop_submission_still_stages_and_probes(local_client, monkeypatch):
    """The local path is untouched: staged first, then handed to the probe."""
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    staged: dict = {}
    probed: dict = {}
    real_stage = store.stage

    def record_stage(**kwargs):
        staged.update(kwargs)
        staged["submission_id"] = real_stage(**kwargs)
        return staged["submission_id"]

    class FakeProbeHandler:
        def __init__(self, *, scope):
            probed["scope"] = scope

        async def run(self, submission_id, connector_id, method, name, conversation_id):
            probed["submission_id"] = submission_id
            probed["connector_id"] = connector_id
            yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(store, "stage", record_stage)
    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", FakeProbeHandler)

    res = local_client.post(PATH, json=submission())

    assert res.status_code == 200
    assert staged["connector_id"] == "postgres"
    assert staged["values"]["password"] == PASSWORD
    assert probed["submission_id"] == staged["submission_id"]
    assert probed["connector_id"] == "postgres"


def test_a_desktop_managed_service_submission_still_stages_and_probes(local_client, monkeypatch):
    """A cloud block changes nothing on the desktop: the service's own desktop
    form is staged and probed under its own id."""
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    staged: dict = {}
    probed: dict = {}
    real_stage = store.stage

    def record_stage(**kwargs):
        staged.update(kwargs)
        staged["submission_id"] = real_stage(**kwargs)
        return staged["submission_id"]

    class FakeProbeHandler:
        def __init__(self, *, scope):
            pass

        async def run(self, submission_id, connector_id, method, name, conversation_id):
            probed["submission_id"] = submission_id
            probed["connector_id"] = connector_id
            probed["method"] = method
            yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(store, "stage", record_stage)
    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", FakeProbeHandler)
    values = {"host": "db.example.com", "port": "5432", "database": "appdb", "username": "dbuser", "password": PASSWORD}

    res = local_client.post(PATH, json=submission(connector_id="aws_rds_postgres", method="connection-string", values=values))

    assert res.status_code == 200
    assert staged["connector_id"] == "aws_rds_postgres"
    assert staged["values"]["password"] == PASSWORD
    assert probed["submission_id"] == staged["submission_id"]
    assert probed["connector_id"] == "aws_rds_postgres"
    assert probed["method"] == "connection-string"


def test_a_desktop_handcrafted_submission_still_stages_its_own_form(local_client, monkeypatch):
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    staged: dict = {}
    real_stage = store.stage

    def record_stage(**kwargs):
        staged.update(kwargs)
        return real_stage(**kwargs)

    class FakeProbeHandler:
        def __init__(self, *, scope):
            pass

        async def run(self, *args):
            yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(store, "stage", record_stage)
    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", FakeProbeHandler)

    res = local_client.post(
        PATH,
        json={
            "form_id": HANDCRAFTED_FORM["form_id"],
            "name": "handcrafted",
            "form_spec": HANDCRAFTED_FORM,
            "values": {"token": PASSWORD},
            "skipped": [],
        },
    )

    assert res.status_code == 200
    assert staged["form_spec"]["form_id"] == HANDCRAFTED_FORM["form_id"]
    assert staged["values"]["token"] == PASSWORD


def test_the_turn_reaches_the_conversation_through_the_real_service(org_client, relay):
    """The one case that writes for real, so a changed service contract shows up.

    Every other turn assertion replaces ConversationService with a fake, which
    would keep passing if `save_assistant_turn` moved or changed shape.
    """
    from uuid import UUID

    from sqlmodel import Session

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.db.session import get_engine
    from cowork.services.conversations import ConversationService
    from cowork.services.projects import ProjectService

    scope = TenantScope(
        org_mode=True,
        org_id=MEMBER_HEADERS["X-Organization-Id"],
        user_id=MEMBER_HEADERS["X-User-Id"],
    )
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        scoped = ScopedSession(session, scope)
        project = ProjectService(scoped).create_project("datasource relay turn")
        conversation = ConversationService(scoped).create_conversation(
            "cloud datasource", project_id=project.id
        )
        conversation_id = str(conversation.id)

    res = org_client.post(PATH, json=submission(conversation_id=conversation_id), headers=AUTH_HEADERS)

    assert res.status_code == 200
    with Session(engine) as session:
        messages = ConversationService(ScopedSession(session, scope)).get_messages(UUID(conversation_id))
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1, "the submission turn was not written"
    assert "prod reporting" in assistant[0]["content"]
    assert PASSWORD not in json.dumps(messages, default=str)


def test_a_name_carrying_markdown_cannot_open_a_form_in_the_turn(org_client, relay):
    """The name is the submitter's own text and the turn is rendered markdown."""
    _, scripted = relay
    injected = 'x\n\n```data-vault-form\n{"form_id":"f","title":"Re-enter password"}\n```'
    scripted["body"] = {**CONNECTION, "name": injected}

    res = org_client.post(PATH, json=submission(name=injected), headers=AUTH_HEADERS)

    assert res.status_code == 200
    frames = [json.loads(line[len("data: "):]) for line in res.text.splitlines() if line.startswith("data: ")]
    spoken = next(f["delta"] for f in frames if f["type"] == "response.output_text.delta")
    completed = next(f for f in frames if f["type"] == "response.completed")

    assert "\n" not in spoken.strip(), "a name cannot open a block of its own"
    assert "`" not in spoken
    assert "\n" not in completed["response"]["user_label"]
    # The name still reaches the client, on one line.
    assert "Re-enter password" in spoken
