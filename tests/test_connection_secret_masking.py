"""A secret saved under a hidden `_` field name.

A connection may keep a secret under a `_`-prefixed name, which every vault
leaves out of the agent's DS_* environment. The connections API still masks
it, on the local vault and for a person reading a connection shared with
them, and request entry still registers its value for redaction, while the
agent's environment goes on without it.
"""
from __future__ import annotations

import json

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import _reset_registered_ds_vars, begin_ds_turn_scope, scrub_credentials, scrub_output
from fastapi.testclient import TestClient
from sqlalchemy import delete

import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import LayeredDataVault
from cowork.services.connectors.persist import own_vault_for_scope, vault_for_scope
from cowork.services.connectors.vault_secrets import register_vault_secrets

SECRET = "sentinel-secret-1"
GENIE_FIELDS = {
    "_connector_id": "databricks",
    "_method": "genie-mcp",
    "workspace_url": "https://workspace.example.com",
    "client_id": "genie-client",
    "_client_secret": SECRET,
}
GENIE_SECURE_KEYS = ["_client_secret"]

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
ALICE = TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)
BOB = TenantScope(personal=True, org_id=PARTITION_B, user_id=SUB_B, personal_resources=ALL_PERSONAL_RESOURCES)
TOKEN = "secret-masking-test-token"


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "home" / "memory"))
    get_app_settings.cache_clear()
    yield tmp_path
    with get_open_session() as db:
        db.exec(delete(Conversation).where(Conversation.org_id.in_([PARTITION_A, PARTITION_B])))
        db.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def scrub_scope(request):
    """This test's own DS_* scrub scope, dropped after it."""
    request.addfinalizer(_reset_registered_ds_vars)
    begin_ds_turn_scope()


def _client() -> TestClient:
    return TestClient(
        cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )


def _assert_secret_masked(body: dict) -> None:
    assert SECRET not in json.dumps(body)
    assert body["fields"]["_client_secret"] == VAULT_KEEP_SENTINEL
    assert "_client_secret" in body["secureKeys"]
    assert body["method"] == "genie-mcp"
    assert body["connector_id"] == "databricks"
    assert body["fields"]["workspace_url"] == "https://workspace.example.com"
    assert body["fields"]["client_id"] == "genie-client"


def test_the_local_vault_masks_a_hidden_secret(stores):
    own_vault_for_scope(LOCAL_SCOPE).save("databricks", "genie", dict(GENIE_FIELDS), secure_keys=GENIE_SECURE_KEYS)

    response = _client().get("/api/v1/connectors/connections/databricks/genie")

    assert response.status_code == 200
    _assert_secret_masked(response.json())


def test_a_reader_of_a_shared_connection_gets_the_hidden_secret_masked(
    monkeypatch, personal_deployment, managed_dir, stores
):
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    own_vault_for_scope(ALICE).save("databricks", "genie", dict(GENIE_FIELDS), secure_keys=GENIE_SECURE_KEYS)
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": PARTITION_A, "engine": "databricks", "name": "genie"},
    ]))
    assert isinstance(vault_for_scope(BOB), LayeredDataVault)

    response = _client().get(
        "/api/v1/connectors/connections/databricks/genie",
        headers={"Authorization": f"Bearer {TOKEN}", "X-User-Id": SUB_B, "X-Organization-Id": DEPLOYMENT},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["shared"] is True
    _assert_secret_masked(body)


async def test_request_entry_registers_a_hidden_secret_without_adding_it_to_the_environment(
    stores, scrub_scope
):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    vault.save("databricks", "genie", dict(GENIE_FIELDS), secure_keys=GENIE_SECURE_KEYS)

    await register_vault_secrets(LOCAL_SCOPE)

    assert SECRET not in scrub_credentials(f"the secret is {SECRET}")
    assert SECRET not in scrub_output(f"the secret is {SECRET}")
    env = vault_for_scope(LOCAL_SCOPE).env_for("databricks", "genie")
    assert SECRET not in env.values()
    assert not any("CLIENT_SECRET" in key for key in env)


async def test_hidden_fields_not_marked_secure_stay_readable(stores, scrub_scope):
    """Bookkeeping such as `_method` is not a secret, so text naming it is
    left as it was, which keeps connect input and published code intact."""
    own_vault_for_scope(LOCAL_SCOPE).save(
        "databricks", "genie", {**GENIE_FIELDS, "_label": "Analytics workspace"}, secure_keys=GENIE_SECURE_KEYS
    )

    await register_vault_secrets(LOCAL_SCOPE)

    text = "method genie-mcp for Analytics workspace"
    assert scrub_credentials(text) == text
