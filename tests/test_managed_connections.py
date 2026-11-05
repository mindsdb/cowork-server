"""The managed folder's connections.

Every vault read layers them over the reader's own (LayeredDataVault): the
connections API lists and reads them with ``managed`` true, ``shared`` false
and their secrets masked, a same-named entry of the person's own stays
hidden, and every mutating route refuses one with 403 and the sentence every
connection the caller can't change answers, a delete before its OAuth
revoke. A turn reads them with the person's own, and a connection
the agent saves in the turn lands in the person's own vault and outlives the
turn. With no managed folder, every vault is the scope's own, as before.
"""
from __future__ import annotations

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import _reset_registered_ds_vars, begin_ds_turn_scope, scrub_credentials
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import delete

import cowork.server as cowork_server
from cowork.api.v1.endpoints.connectors import connections as connections_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.schemas.connectors import DirectSaveRequest, PatchPickedFilesBody
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.layered_vault import (
    SHARED_CONNECTION_REFUSAL,
    LayeredDataVault,
    SharedConnectionError,
)
from cowork.services.connectors.persist import own_vault_for_scope, vault_for_scope
from cowork.services.connectors.vault_secrets import register_vault_secrets
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
ALICE = TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)

MANAGED_SECRET = "operatorSecret123"
OWN_SECRET = "aliceSecret456"


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
        db.exec(delete(Conversation).where(Conversation.org_id == PARTITION_A))
        db.exec(delete(Project).where(Project.org_id == PARTITION_A))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def warehouse(managed_dir, stores):
    """The operator's managed postgres connection, `warehouse`."""
    get_app_settings.cache_clear()
    LocalDataVault(managed_dir.vault_dir).save(
        "postgres",
        "warehouse",
        {"host": "db.example", "user": "svc", "password": MANAGED_SECRET},
        secure_keys=["password"],
    )
    return managed_dir


def _save_own(scope, name: str, *, secret: str = OWN_SECRET) -> None:
    own_vault_for_scope(scope).save(
        "postgres", name, {"host": "mine.example", "password": secret}, secure_keys=["password"]
    )


def _refused(call) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert refused.value.status_code == 403
    assert refused.value.detail == SHARED_CONNECTION_REFUSAL


async def _refused_async(call) -> None:
    with pytest.raises(HTTPException) as refused:
        await call()
    assert refused.value.status_code == 403
    assert refused.value.detail == SHARED_CONNECTION_REFUSAL


# ── listing and reading ─────────────────────────────────────────────────────


async def test_managed_connections_are_listed_and_read_as_managed(warehouse):
    _save_own(ALICE, "sales")
    _save_own(ALICE, "warehouse", secret="hidden-own-secret")

    listed = await connections_ep.list_connections(ALICE, None)
    by_name = {(c.engine, c.name): c for c in listed}
    assert set(by_name) == {("postgres", "warehouse"), ("postgres", "sales")}
    assert by_name[("postgres", "warehouse")].managed is True
    assert by_name[("postgres", "sales")].managed is False
    assert not any(c.shared for c in listed)

    detail = await connections_ep.get_connection("postgres", "warehouse", ALICE, None)
    assert detail.managed is True
    assert detail.shared is False
    assert detail.fields["host"] == "db.example"
    assert MANAGED_SECRET not in detail.fields.values()
    assert "hidden-own-secret" not in detail.fields.values()
    assert detail.secure_keys == ["password"]
    own = await connections_ep.get_connection("postgres", "sales", ALICE, None)
    assert own.managed is False


async def test_without_a_managed_folder_every_vault_is_the_scopes_own(stores):
    _save_own(ALICE, "sales")
    assert not isinstance(vault_for_scope(ALICE), LayeredDataVault)
    listed = await connections_ep.list_connections(ALICE, None)
    assert [(c.name, c.managed) for c in listed] == [("sales", False)]


# ── every change ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("own_copy", [False, True], ids=["no-own-copy", "hidden-own-copy"])
async def test_every_change_to_a_managed_connection_is_refused(warehouse, monkeypatch, own_copy):
    if own_copy:
        _save_own(ALICE, "warehouse", secret="hidden-own-secret")
    revoked = []
    monkeypatch.setattr(
        connections_ep.oauth_service, "revoke", lambda *args, **kwargs: revoked.append(args)
    )

    await _refused_async(lambda: connections_ep.delete_connection("postgres", "warehouse", ALICE, None))
    assert revoked == []
    _refused(
        lambda: connections_ep.patch_connection_token(
            "postgres", "warehouse", connections_ep.PatchTokenBody(status="needs_reconnect"), ALICE
        )
    )
    await _refused_async(
        lambda: connections_ep.patch_picked_files(
            "postgres", "warehouse", PatchPickedFilesBody(files=[]), ALICE, None
        )
    )
    _refused(lambda: connections_ep.delete_picked_file("postgres", "warehouse", "file-1", "general", ALICE))
    await _refused_async(
        lambda: connections_ep.patch_access_mode(
            "postgres", "warehouse", connections_ep.PatchAccessModeBody(access_mode="write"), ALICE, None
        )
    )
    _refused(
        lambda: connections_ep.save_connection_direct(
            DirectSaveRequest(connector_id="postgres", name="warehouse", values={"host": "x"}), ALICE
        )
    )

    managed = LocalDataVault(warehouse.vault_dir).read_record("postgres", "warehouse")
    assert managed["fields"]["password"] == MANAGED_SECRET
    own = own_vault_for_scope(ALICE).read_record("postgres", "warehouse")
    if own_copy:
        assert own["fields"]["password"] == "hidden-own-secret"
    else:
        assert own is None


async def test_the_persons_own_connection_is_deleted_and_revoked_as_before(warehouse, monkeypatch):
    _save_own(ALICE, "sales")
    revoked = []
    monkeypatch.setattr(
        connections_ep.oauth_service, "revoke", lambda engine, name, *args, **kwargs: revoked.append((engine, name))
    )

    await connections_ep.delete_connection("postgres", "sales", ALICE, None)

    assert revoked == [("postgres", "sales")]
    assert own_vault_for_scope(ALICE).read_record("postgres", "sales") is None


def test_the_vault_refuses_a_managed_save_or_delete(warehouse):
    vault = vault_for_scope(ALICE)
    with pytest.raises(SharedConnectionError):
        vault.save("postgres", "warehouse", {"host": "x"})
    with pytest.raises(SharedConnectionError):
        vault.delete("postgres", "warehouse")
    with pytest.raises(SharedConnectionError):
        ConnectionsService(ALICE).patch_token("postgres", "warehouse", {"status": "x"})


def test_a_save_whose_derived_name_lands_on_a_managed_connection_answers_403(monkeypatch, stores, managed_dir):
    """The /save route without a name: the derived slug is the managed one, for
    the same account, so the vault refuses the save and create_app's handler
    answers it."""
    get_app_settings.cache_clear()
    values = {"email": "ops@example.com", "app_password": "managed-app-password"}
    LocalDataVault(managed_dir.vault_dir).save(
        "gmail", "ops-example-com", {**values, "_connector_id": "gmail", "_method": "app-password"}
    )
    client = TestClient(
        cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )

    response = client.post(
        "/api/v1/connectors/connections/save",
        json={"connector_id": "gmail", "method": "app-password", "name": "", "values": values},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": SHARED_CONNECTION_REFUSAL}
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []


# ── secrets and turns ───────────────────────────────────────────────────────


async def test_request_entry_registers_the_managed_secret(warehouse, personal_deployment, request):
    request.addfinalizer(_reset_registered_ds_vars)
    begin_ds_turn_scope()
    await register_vault_secrets(ALICE)
    assert MANAGED_SECRET not in scrub_credentials(f"the password is {MANAGED_SECRET}")


async def test_a_turn_reads_managed_connections_and_keeps_what_it_saves(
    monkeypatch, warehouse, stores, personal_deployment
):
    _save_own(ALICE, "sales")

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=ALICE) as turn:
        vault = turn.config.data_vault
        assert {(c["engine"], c["name"]) for c in vault.list_connections()} == {
            ("postgres", "warehouse"),
            ("postgres", "sales"),
        }
        assert vault.load("postgres", "warehouse")["password"] == MANAGED_SECRET
        # What anton's connect_new_datasource does with a connection the
        # person pastes into the chat.
        vault.save("mysql", "pasted", {"host": "pasted.example", "password": "p"}, secure_keys=["password"])
        with pytest.raises(SharedConnectionError):
            vault.save("postgres", "warehouse", {"host": "x"})

    assert own_vault_for_scope(ALICE).read_record("mysql", "pasted")["fields"]["host"] == "pasted.example"
    assert LocalDataVault(warehouse.vault_dir).read_record("mysql", "pasted") is None


async def test_a_turn_that_disables_connections_keeps_each_records_secret_keys(
    monkeypatch, warehouse, stores, personal_deployment
):
    _save_own(ALICE, "sales")

    async with built_turn(
        monkeypatch=monkeypatch,
        tmp_path=stores,
        scope=ALICE,
        disabled_connections=[{"engine": "postgres", "name": "sales"}],
    ) as turn:
        vault = turn.config.data_vault
        assert [(c["engine"], c["name"]) for c in vault.list_connections()] == [("postgres", "warehouse")]
        assert vault.read_record("postgres", "warehouse")["secure_keys"] == ["password"]

    async with built_turn(
        monkeypatch=monkeypatch,
        tmp_path=stores,
        scope=ALICE,
        disabled_connections=[{"engine": "postgres", "name": "warehouse"}],
    ) as turn:
        assert [(c["engine"], c["name"]) for c in turn.config.data_vault.list_connections()] == [
            ("postgres", "sales")
        ]
