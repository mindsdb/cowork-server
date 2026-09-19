"""Connector form submissions in org mode are relayed, never stored locally.

A cloud submission carries a database password. It must reach auth's
encrypted store and nothing else: no staging store, no probe, no local vault,
on the success path and on every refusal. Each org case runs with those paths
booby-trapped, so a fallback that quietly saves the credential fails the test
rather than passing it.

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

#: Shaped like auth's DatasourceConnectionResponseSerializer.
CONNECTION = {
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
            "tls_mode": "system",
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
def unverified_org_client(monkeypatch, forbid_local_paths) -> TestClient:
    """Org mode with the pair enabled but the specs as the release ships them."""
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
    assert body["username"] == "dbuser"
    assert body["password"] == PASSWORD
    assert body["tls"] == {"mode": "system", "ca_pem": None}
    # Owner and org are auth's to derive from the bearer.
    assert "user_id" not in body and "organization_id" not in body
    # What the submitter sees back is auth's masked metadata, not their input.
    assert PASSWORD not in res.text
    assert PASSWORD not in caplog.text
    assert "prod reporting" in res.text
    assert "response.completed" in res.text


def test_a_custom_ca_travels_in_the_tls_block(org_client, relay):
    recorded, _ = relay
    values = submission()["values"] | {"tls_mode": "custom_ca", "ca_pem": CA_PEM}

    res = org_client.post(PATH, json=submission(values=values), headers=AUTH_HEADERS)

    assert res.status_code == 200
    body = json.loads(recorded[0].content)
    assert body["tls"] == {"mode": "custom_ca", "ca_pem": CA_PEM}


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
        pytest.param(submission(connector_id="not-a-connector"), id="unknown-connector"),
        pytest.param(submission(method=None), id="no-method"),
        pytest.param(
            {
                "form_id": "handcrafted-connector",
                "name": "handcrafted",
                "form_spec": {"form_id": "handcrafted-connector", "fields": [{"name": "token"}]},
                "values": {"token": PASSWORD},
                "skipped": [],
            },
            id="handcrafted-form",
        ),
    ],
)
def test_everything_that_is_not_an_enabled_cloud_method_is_refused(org_client, relay, body):
    recorded, _ = relay

    res = org_client.post(PATH, json=body, headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
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
        def __init__(self, session):
            probed["session"] = session

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


def test_a_desktop_handcrafted_submission_still_stages_its_own_form(local_client, monkeypatch):
    from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
    from cowork.services.connectors.submissions import store

    staged: dict = {}
    real_stage = store.stage

    def record_stage(**kwargs):
        staged.update(kwargs)
        return real_stage(**kwargs)

    class FakeProbeHandler:
        def __init__(self, session):
            pass

        async def run(self, *args):
            yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(store, "stage", record_stage)
    monkeypatch.setattr(submissions_endpoint, "ProbeHandler", FakeProbeHandler)

    res = local_client.post(
        PATH,
        json={
            "form_id": "handcrafted-connector",
            "name": "handcrafted",
            "form_spec": {"form_id": "handcrafted-connector", "fields": [{"name": "token", "required": True}]},
            "values": {"token": PASSWORD},
            "skipped": [],
        },
    )

    assert res.status_code == 200
    assert staged["form_spec"]["form_id"] == "handcrafted-connector"
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
