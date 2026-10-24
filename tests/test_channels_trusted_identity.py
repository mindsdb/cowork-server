"""Channels on a local deployment that trusts identity headers.

A channel's config is the deployment's one global row, and every channel
message runs through it, so no person may change it: the routes that
configure a channel answer 403 to every person, whatever roles the proxy
forwards. The operator view (a request with no person, under
COWORK_IDENTITY_ENFORCE=audit) still sets it, and so does
``python -m cowork.ops.channels``. A channel's live adapter and its ingress
are keyed by no org, never a person's partition id, so a channel's turn binds
the scope with no person (tests/test_shared_connections.py has what that turn
reads).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import ORG_MANAGE_ROLE, personal_partition_id

TOKEN = "channels-trusted-identity-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_B = "22222222-2222-4222-8222-222222222222"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
B = {**BEARER, "X-User-Id": SUB_B, "X-Organization-Id": DEPLOYMENT}
REQUIRES_THE_OPERATOR = {"detail": "requires the operator"}

CONFIGURE_ROUTES = (
    ("put", "/api/v1/channels/telegram/config", {"values": {"bot_token": "persons-bot"}}),
    ("delete", "/api/v1/channels/telegram/config", None),
    ("post", "/api/v1/channels/telegram/reload", None),
    ("post", "/api/v1/channels/telegram/setup", None),
    ("post", "/api/v1/channels/telegram/teardown", None),
    ("put", "/api/v1/channels/agent", {"harness": "anton"}),
)


@pytest.fixture
def client(personal_deployment, monkeypatch, tmp_path):
    """A trusted stack create_app accepts, which serves a request with no
    identity as the operator view."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app())
    # The suite shares one database, so start with no Telegram config.
    client.delete("/api/v1/channels/telegram/config", headers=BEARER)
    yield client
    client.delete("/api/v1/channels/telegram/config", headers=BEARER)
    get_app_settings.cache_clear()


def _bot_token_is_set(client: TestClient) -> bool:
    config = client.get("/api/v1/channels/telegram/config", headers=B)
    assert config.status_code == 200
    return config.json()["fields"]["bot_token"]["is_set"]


@pytest.mark.parametrize("headers", [B, {**B, "X-User-Roles": ORG_MANAGE_ROLE}])
def test_every_route_that_configures_a_channel_refuses_a_person(client, headers):
    for method, path, body in CONFIGURE_ROUTES:
        refused = client.request(method, path, headers=headers, json=body)
        assert (refused.status_code, refused.json()) == (403, REQUIRES_THE_OPERATOR), path
    assert not _bot_token_is_set(client)


def test_the_operator_view_sets_a_channel(client):
    saved = client.put(
        "/api/v1/channels/telegram/config",
        headers=BEARER,
        json={"values": {"bot_token": "operators-bot"}},
    )
    assert saved.status_code == 200
    assert _bot_token_is_set(client)
    assert client.delete("/api/v1/channels/telegram/config", headers=BEARER).status_code == 204
    assert not _bot_token_is_set(client)


def test_a_persons_partition_is_never_a_channels_org(personal_deployment):
    from cowork.db.scoped import scope_for_identity
    from cowork.services.channels import installation_org_id

    person = scope_for_identity(org_id=personal_partition_id(DEPLOYMENT, SUB_B), user_id=SUB_B)
    assert person.personal
    assert installation_org_id(person) is None
