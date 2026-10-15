"""Connections that record their creator, and sharing one with everyone.

On a deployment that trusts identity headers, every save records who created
the connection in ``_created_by``: a create stamps the person's sub, an
update keeps the stored value whatever the request or a later saver says,
and the operator view stamps nobody. The stamp never reaches the agent's
environment, its connection listing or the API's fields.

With connections personal, the operator shares one person's connection by
listing it in the managed folder's ``shared/connections.json``. Everyone else
reads it live from its creator's vault, marked ``managed`` (they can't change
it) and ``shared``, and every change they try answers 403 with the shared
sentence, a delete before its OAuth revoke. Its creator sees it shared and
changes, reconnects and deletes it as before, and everyone sees each change.

With connections shared, everyone reads the one base vault, and only an
entry's creator changes it; an entry with no creator only the operator view.
The desktop and hosted org tenancy keep every route as before.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import build_datasource_context
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import delete

import cowork.server as cowork_server
from cowork.api.v1.endpoints.connectors import connections as connections_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import use_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness.tools import _cowork_label_connection
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.schemas.connectors import DirectSaveRequest, PatchPickedFilesBody
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.layered_vault import (
    CREATED_BY_FIELD,
    SHARED_CONNECTION_REFUSAL,
    LayeredDataVault,
    SharedConnectionError,
)
from cowork.services.connectors.persist import (
    own_vault_for_scope,
    persist_connection,
    set_connection_label,
    turn_vault_for_scope,
    vault_for_scope,
)
from tests._fakes import FakeRequest
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
TOKEN = "shared-connections-test-token"
SHARED_RESOURCES = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS})

ERP = {"host": "erp.example", "user": "svc", "password": "erpSecret123"}


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)


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
def trusted(personal_deployment, stores):
    """Every resource personal, no managed folder."""
    get_app_settings.cache_clear()
    return stores


@pytest.fixture
def sharing(personal_deployment, managed_dir, stores):
    """Every resource personal, with a managed folder whose registry lists
    Alice's postgres connection `erp`, which she saved herself."""
    get_app_settings.cache_clear()
    persist_connection("postgres", None, "erp", dict(ERP), scope=ALICE)
    _write_registry(managed_dir, [_entry(sub=SUB_A, partition=PARTITION_A, name="erp")])
    return managed_dir


@pytest.fixture
def all_shared(personal_deployment, stores, monkeypatch):
    """Connections left off COWORK_PERSONAL_RESOURCES: one base vault."""
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(SHARED_RESOURCES)))
    get_app_settings.cache_clear()
    return stores


def _entry(*, sub: str, partition: str, name: str, engine: str = "postgres") -> dict[str, str]:
    return {"user_id": sub, "partition_id": partition, "engine": engine, "name": name}


def _write_registry(managed_dir, entries: list[dict[str, str]]) -> None:
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_connections_file.write_text(json.dumps(entries))


def _stored(scope: TenantScope, name: str = "erp", engine: str = "postgres") -> dict:
    return own_vault_for_scope(scope).read_record(engine, name)["fields"]


def _names(vault) -> set[tuple[str, str]]:
    return {(c["engine"], c["name"]) for c in vault.list_connections()}


def _no_revoke(monkeypatch) -> list:
    revoked: list = []
    monkeypatch.setattr(
        connections_ep.oauth_service, "revoke", lambda *args, **kwargs: revoked.append(args)
    )
    return revoked


def _refused(call) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (403, SHARED_CONNECTION_REFUSAL)


async def _refused_async(call) -> None:
    with pytest.raises(HTTPException) as refused:
        await call()
    assert (refused.value.status_code, refused.value.detail) == (403, SHARED_CONNECTION_REFUSAL)


async def _every_change_is_refused(scope: TenantScope, name: str = "erp") -> None:
    """Every mutating connections route, and the agent's label tool, for
    ``scope`` on postgres/``name``."""
    await _refused_async(lambda: connections_ep.delete_connection("postgres", name, scope, None))
    _refused(
        lambda: connections_ep.patch_connection_token(
            "postgres", name, connections_ep.PatchTokenBody(access_token="stolen"), scope
        )
    )
    await _refused_async(
        lambda: connections_ep.patch_picked_files(
            "postgres", name, PatchPickedFilesBody(files=[{"id": "f1", "name": "f"}]), scope, None
        )
    )
    _refused(lambda: connections_ep.delete_picked_file("postgres", name, "f1", "general", scope))
    await _refused_async(
        lambda: connections_ep.patch_access_mode(
            "postgres", name, connections_ep.PatchAccessModeBody(access_mode="write"), scope, None
        )
    )
    _refused(
        lambda: connections_ep.save_connection_direct(
            DirectSaveRequest(connector_id="postgres", name=name, values={"host": "x"}), scope
        )
    )
    with use_settings_scope(scope):
        labeled = await _cowork_label_connection(
            None, {"engine": "postgres", "name": name, "label": "Mine now"}
        )
    assert labeled == f"label_connection: {SHARED_CONNECTION_REFUSAL}"
    vault = vault_for_scope(scope)
    with pytest.raises(SharedConnectionError):
        vault.save("postgres", name, {"host": "x"})
    with pytest.raises(SharedConnectionError):
        vault.delete("postgres", name)


# ── the creator stamp ───────────────────────────────────────────────────────


def test_a_create_stamps_the_person_and_never_the_requests_value(trusted):
    persist_connection("postgres", None, "erp", {**ERP, CREATED_BY_FIELD: SUB_B}, scope=ALICE)
    assert _stored(ALICE)[CREATED_BY_FIELD] == SUB_A


def test_every_update_keeps_the_stored_creator(trusted):
    persist_connection("postgres", None, "erp", dict(ERP), scope=ALICE)
    service = ConnectionsService(ALICE)

    # A reconnect whose request names someone else.
    persist_connection(
        "postgres", None, "erp", {**ERP, CREATED_BY_FIELD: SUB_B}, replace_existing=True, scope=ALICE
    )
    assert _stored(ALICE)[CREATED_BY_FIELD] == SUB_A
    # The token patch, which writes whatever keys it is given.
    assert service.patch_token("postgres", "erp", {"access_token": "t2", CREATED_BY_FIELD: SUB_B})
    assert _stored(ALICE)[CREATED_BY_FIELD] == SUB_A
    assert _stored(ALICE)["access_token"] == "t2"
    connections_ep.patch_connection_token(
        "postgres", "erp", connections_ep.PatchTokenBody(access_token="t3"), ALICE
    )
    assert service.merge_picked_files("postgres", "erp", [{"id": "f1"}]) is not None
    assert set_connection_label("postgres", "erp", "Finance", scope=ALICE) == "Finance"
    stored = _stored(ALICE)
    assert (stored[CREATED_BY_FIELD], stored["access_token"], stored["_user_label"]) == (
        SUB_A,
        "t3",
        "Finance",
    )


def test_an_update_keeps_an_entry_with_no_creator_unstamped(trusted):
    """Saved before creators were recorded: an update keeps the stored
    value, which is none, rather than taking the saver's."""
    own_vault_for_scope(ALICE).save("postgres", "legacy", dict(ERP))

    assert ConnectionsService(ALICE).patch_token("postgres", "legacy", {"status": "needs_reconnect"})

    assert CREATED_BY_FIELD not in _stored(ALICE, "legacy")


def test_the_operator_view_never_stamps_a_creator(trusted):
    persist_connection("postgres", None, "ops", {**ERP, CREATED_BY_FIELD: SUB_A}, scope=LOCAL_SCOPE)
    assert CREATED_BY_FIELD not in _stored(LOCAL_SCOPE, "ops")


async def test_the_creator_never_reaches_the_agent_or_the_api(trusted):
    persist_connection("postgres", None, "erp", dict(ERP), scope=ALICE)

    vault = turn_vault_for_scope(ALICE)
    env = vault.env_for("postgres", "erp")
    assert env["DS_POSTGRES_ERP__HOST"] == "erp.example"
    assert not any("CREATED_BY" in key for key in env) and SUB_A not in env.values()
    injected = vault.inject_env("postgres", "erp")
    try:
        assert not any("CREATED_BY" in key for key in injected)
        assert SUB_A not in {os.environ[key] for key in injected}
    finally:
        for key in injected:
            os.environ.pop(key, None)
    context = build_datasource_context(vault)
    assert "DS_POSTGRES_ERP__HOST" in context
    assert "CREATED_BY" not in context and SUB_A not in context

    detail = await connections_ep.get_connection("postgres", "erp", ALICE, None)
    assert CREATED_BY_FIELD not in detail.fields
    assert SUB_A not in json.dumps(detail.model_dump())


async def test_a_turn_saves_its_connections_stamped(monkeypatch, trusted):
    async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=ALICE) as turn:
        # What anton's connect_new_datasource does with a pasted connection.
        turn.config.data_vault.save("mysql", "pasted", {"host": "pasted.example"})
    assert _stored(ALICE, "pasted", "mysql")[CREATED_BY_FIELD] == SUB_A


# ── one connection shared ───────────────────────────────────────────────────


async def test_bob_reads_alices_shared_connection_and_cant_change_it(sharing, monkeypatch, stores):
    listed = {(c.engine, c.name): c for c in await connections_ep.list_connections(BOB, None)}
    assert (listed[("postgres", "erp")].managed, listed[("postgres", "erp")].shared) == (True, True)
    detail = await connections_ep.get_connection("postgres", "erp", BOB, None)
    assert (detail.managed, detail.shared) == (True, True)
    assert detail.fields["host"] == "erp.example"
    assert ERP["password"] not in detail.fields.values()

    alices = {(c.engine, c.name): c for c in await connections_ep.list_connections(ALICE, None)}
    assert (alices[("postgres", "erp")].managed, alices[("postgres", "erp")].shared) == (False, True)
    detail = await connections_ep.get_connection("postgres", "erp", ALICE, None)
    assert (detail.managed, detail.shared) == (False, True)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB) as turn:
        vault = turn.config.data_vault
        assert ("postgres", "erp") in _names(vault)
        assert vault.load("postgres", "erp")["password"] == ERP["password"]
        assert vault.env_for("postgres", "erp")["DS_POSTGRES_ERP__PASSWORD"] == ERP["password"]
    # Nothing was copied into Bob's own vault.
    assert own_vault_for_scope(BOB).list_connections() == []


async def test_every_change_bob_tries_is_refused_before_any_revoke(sharing, monkeypatch):
    revoked = _no_revoke(monkeypatch)
    before = own_vault_for_scope(ALICE).read_record("postgres", "erp")

    await _every_change_is_refused(BOB)

    assert revoked == []
    assert own_vault_for_scope(ALICE).read_record("postgres", "erp") == before
    assert own_vault_for_scope(BOB).list_connections() == []


def test_bobs_save_whose_derived_name_lands_on_the_shared_one_answers_403(sharing, monkeypatch):
    """The /save route with no name: the derived slug is Alice's shared
    connection, for the same account, so the vault refuses the save and
    create_app's handler answers it."""
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    values = {"email": "team@example.com", "app_password": "team-app-password"}
    slug = persist_connection("gmail", "app-password", "", dict(values), scope=ALICE)
    _write_registry(
        sharing,
        [_entry(sub=SUB_A, partition=PARTITION_A, name=slug, engine="gmail")],
    )
    client = TestClient(
        cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )

    response = client.post(
        "/api/v1/connectors/connections/save",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "X-User-Id": SUB_B,
            "X-Organization-Id": DEPLOYMENT,
        },
        json={"connector_id": "gmail", "method": "app-password", "name": "", "values": values},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": SHARED_CONNECTION_REFUSAL}
    assert own_vault_for_scope(BOB).list_connections() == []


async def test_alice_changes_her_shared_connection_and_bob_sees_each_change(sharing, monkeypatch):
    revoked = []
    monkeypatch.setattr(
        connections_ep.oauth_service,
        "revoke",
        lambda engine, name, *args, **kwargs: revoked.append((engine, name)),
    )

    await connections_ep.patch_access_mode(
        "postgres", "erp", connections_ep.PatchAccessModeBody(access_mode="write"), ALICE, None
    )
    connections_ep.patch_connection_token(
        "postgres", "erp", connections_ep.PatchTokenBody(access_token="fresh"), ALICE
    )
    assert ConnectionsService(ALICE).patch_token("postgres", "erp", {"host": "erp2.example"})

    seen = vault_for_scope(BOB).load("postgres", "erp")
    assert (seen["_access_mode"], seen["access_token"], seen["host"]) == ("write", "fresh", "erp2.example")
    assert _stored(ALICE)[CREATED_BY_FIELD] == SUB_A

    await connections_ep.delete_connection("postgres", "erp", ALICE, None)

    assert revoked == [("postgres", "erp")]
    assert ("postgres", "erp") not in _names(vault_for_scope(BOB))
    assert vault_for_scope(BOB).read_record("postgres", "erp") is None
    assert [c.name for c in await connections_ep.list_connections(BOB, None)] == []


def test_only_registry_entries_of_existing_connections_are_read(sharing):
    persist_connection("postgres", None, "private", dict(ERP), scope=ALICE)
    _write_registry(
        sharing,
        [
            _entry(sub=SUB_A, partition=PARTITION_A, name="erp"),
            _entry(sub=SUB_A, partition=PARTITION_A, name="gone"),
        ],
    )

    vault = vault_for_scope(BOB)

    assert _names(vault) == {("postgres", "erp")}
    assert vault.read_record("postgres", "gone") is None
    assert vault.read_record("postgres", "private") is None
    assert vault.can_change("postgres", "gone")


def test_an_earlier_layer_hides_a_later_same_named_one(sharing):
    # Saved before the share: a save after it would see Alice's erp and the
    # operator's warehouse, and take another name.
    own_vault_for_scope(BOB).save("postgres", "erp", {**ERP, "host": "bob.example"})
    LocalDataVault(sharing.vault_dir).save("postgres", "warehouse", {"host": "operator.example"})
    own_vault_for_scope(ALICE).save("postgres", "warehouse", {**ERP, "host": "alice-wh.example"})
    _write_registry(
        sharing,
        [
            _entry(sub=SUB_A, partition=PARTITION_A, name="erp"),
            _entry(sub=SUB_A, partition=PARTITION_A, name="warehouse"),
        ],
    )

    bob = vault_for_scope(BOB)

    # Bob's own erp hides Alice's, and stays his to change.
    assert bob.load("postgres", "erp")["host"] == "bob.example"
    assert bob.can_change("postgres", "erp") and not bob.is_shared("postgres", "erp")
    # The operator's own warehouse hides Alice's shared one.
    assert bob.load("postgres", "warehouse")["host"] == "operator.example"
    assert not bob.can_change("postgres", "warehouse") and not bob.is_shared("postgres", "warehouse")
    assert sorted(_names(bob)) == [("postgres", "erp"), ("postgres", "warehouse")]


def test_bobs_own_account_saved_under_the_shared_name_takes_another_name(sharing):
    """A different account is a new connection of Bob's own, not a change to
    Alice's: it lands in his vault beside hers, stamped as his."""
    slug = persist_connection("postgres", None, "erp", {**ERP, "host": "bob.example"}, scope=BOB)

    assert slug != "erp"
    assert _stored(BOB, slug)[CREATED_BY_FIELD] == SUB_B
    assert vault_for_scope(BOB).load("postgres", "erp")["host"] == "erp.example"
    assert _stored(ALICE)["host"] == "erp.example"


def test_a_malformed_registry_shares_nothing_and_says_so(sharing, caplog, owned_logger):
    sharing.shared_connections_file.write_text('{"not": "a list"}')
    owned_logger("cowork.services.connectors.persist", level=logging.ERROR)

    assert _names(vault_for_scope(BOB)) == set()
    assert _names(vault_for_scope(ALICE)) == {("postgres", "erp")}
    assert not vault_for_scope(ALICE).is_shared("postgres", "erp")

    errors = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.services.connectors.persist" and record.levelno == logging.ERROR
    ]
    assert errors and str(sharing.shared_connections_file) in errors[0]


def test_only_a_scope_with_personal_connections_reads_the_registry(sharing):
    """Where connections are shared the base vault already holds everyone's:
    a person's scope that keeps connections shared reads no registry."""
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=SHARED_RESOURCES)
    assert _names(vault_for_scope(bob)) == set()
    assert _names(vault_for_scope(LOCAL_SCOPE)) == set()


async def test_the_operators_own_entry_answers_the_shared_sentence(trusted, managed_dir, monkeypatch):
    get_app_settings.cache_clear()
    LocalDataVault(managed_dir.vault_dir).save("postgres", "erp", dict(ERP))
    revoked = _no_revoke(monkeypatch)

    await _every_change_is_refused(ALICE)

    assert revoked == []
    listed = await connections_ep.list_connections(ALICE, None)
    assert [(c.name, c.managed, c.shared) for c in listed] == [("erp", True, False)]


# ── every connection shared ─────────────────────────────────────────────────


ALICE_SHARED = _person(partition=PARTITION_A, sub=SUB_A, resources=SHARED_RESOURCES)
BOB_SHARED = _person(partition=PARTITION_B, sub=SUB_B, resources=SHARED_RESOURCES)


async def test_where_all_are_shared_only_the_creator_changes_an_entry(all_shared, monkeypatch):
    persist_connection("postgres", None, "erp", dict(ERP), scope=ALICE_SHARED)
    assert _stored(LOCAL_SCOPE)[CREATED_BY_FIELD] == SUB_A
    revoked = _no_revoke(monkeypatch)

    await _every_change_is_refused(BOB_SHARED)
    await _every_change_is_refused(LOCAL_SCOPE)
    assert revoked == []

    # Everyone uses it, and the API says who can't change it.
    assert vault_for_scope(BOB_SHARED).load("postgres", "erp")["host"] == "erp.example"
    bobs = await connections_ep.list_connections(BOB_SHARED, None)
    alices = await connections_ep.list_connections(ALICE_SHARED, None)
    assert [(c.name, c.managed, c.shared) for c in bobs] == [("erp", True, False)]
    assert [(c.name, c.managed, c.shared) for c in alices] == [("erp", False, False)]

    await connections_ep.patch_access_mode(
        "postgres", "erp", connections_ep.PatchAccessModeBody(access_mode="write"), ALICE_SHARED, None
    )
    assert vault_for_scope(BOB_SHARED).load("postgres", "erp")["_access_mode"] == "write"
    await connections_ep.delete_connection("postgres", "erp", ALICE_SHARED, None)
    assert own_vault_for_scope(LOCAL_SCOPE).read_record("postgres", "erp") is None


async def test_where_all_are_shared_an_entry_with_no_creator_is_the_operators(all_shared, monkeypatch):
    own_vault_for_scope(LOCAL_SCOPE).save("postgres", "erp", dict(ERP))
    _no_revoke(monkeypatch)

    await _every_change_is_refused(ALICE_SHARED)
    await _every_change_is_refused(BOB_SHARED)

    assert ConnectionsService(LOCAL_SCOPE).patch_token("postgres", "erp", {"status": "ok"})
    assert CREATED_BY_FIELD not in _stored(LOCAL_SCOPE)
    await connections_ep.delete_connection("postgres", "erp", LOCAL_SCOPE, None)
    assert own_vault_for_scope(LOCAL_SCOPE).read_record("postgres", "erp") is None


# ── the desktop and hosted org tenancy ─────────────────────────────────────


@pytest.fixture
def desktop(monkeypatch, stores):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()
    return stores


async def test_the_desktop_keeps_every_connections_route(desktop, monkeypatch):
    revoked = []
    monkeypatch.setattr(
        connections_ep.oauth_service,
        "revoke",
        lambda engine, name, *args, **kwargs: revoked.append((engine, name)),
    )
    saved = connections_ep.save_connection_direct(
        DirectSaveRequest(
            connector_id="postgres", name="erp", values={**ERP, CREATED_BY_FIELD: SUB_B}
        ),
        LOCAL_SCOPE,
    )
    assert saved["name"] == "erp"
    vault = vault_for_scope(None)
    assert not isinstance(vault, LayeredDataVault)
    # Nothing stamps on the desktop: the vault keeps the fields as sent.
    assert _stored(LOCAL_SCOPE)[CREATED_BY_FIELD] == SUB_B

    connections_ep.patch_connection_token(
        "postgres", "erp", connections_ep.PatchTokenBody(access_token="t"), LOCAL_SCOPE
    )
    await connections_ep.patch_picked_files(
        "postgres", "erp", PatchPickedFilesBody(files=[{"id": "f1", "name": "f", "projects": ["p"]}]),
        LOCAL_SCOPE,
        None,
    )
    connections_ep.delete_picked_file("postgres", "erp", "f1", "p", LOCAL_SCOPE)
    await connections_ep.patch_access_mode(
        "postgres", "erp", connections_ep.PatchAccessModeBody(access_mode="none"), LOCAL_SCOPE, None
    )
    stored = _stored(LOCAL_SCOPE)
    assert (stored["access_token"], stored["_access_mode"]) == ("t", "none")
    assert json.loads(stored["_picked_files"]) == [{"id": "f1", "name": "f", "projects": []}]

    listed = await connections_ep.list_connections(LOCAL_SCOPE, None)
    assert [(c.name, c.managed, c.shared) for c in listed] == [("erp", False, False)]
    detail = await connections_ep.get_connection("postgres", "erp", LOCAL_SCOPE, None)
    assert (detail.managed, detail.shared) == (False, False)

    await connections_ep.delete_connection("postgres", "erp", LOCAL_SCOPE, None)
    assert revoked == [("postgres", "erp")]
    assert vault.list_connections() == []


ORG_ID = "33333333-3333-4333-8333-333333333333"
ORG = TenantScope(org_mode=True, org_id=ORG_ID, user_id="org-user")


@pytest.fixture
def hosted(org_deployment, stores):
    get_app_settings.cache_clear()
    return stores


async def test_hosted_org_tenancy_keeps_every_connections_route(hosted, monkeypatch):
    proxied: list[str] = []

    async def proxy(name: str, result):
        proxied.append(name)
        return result

    proxy_module = connections_ep.auth_proxy
    monkeypatch.setattr(proxy_module, "proxy_catalogue", lambda *a: proxy("catalogue", {"items": []}))
    monkeypatch.setattr(proxy_module, "proxy_connection_detail", lambda *a: proxy("detail", {}))
    monkeypatch.setattr(proxy_module, "proxy_delete", lambda *a: proxy("delete", None))
    monkeypatch.setattr(proxy_module, "proxy_picked_files", lambda *a: proxy("picked", []))
    monkeypatch.setattr(proxy_module, "proxy_access_mode", lambda *a: proxy("access", {"ok": True}))
    request = FakeRequest()

    assert await connections_ep.list_connections(ORG, request) == []
    await connections_ep.get_connection("postgres", "erp", ORG, request)
    await connections_ep.delete_connection("postgres", "erp", ORG, request)
    await connections_ep.patch_picked_files("postgres", "erp", PatchPickedFilesBody(files=[]), ORG, request)
    await connections_ep.patch_access_mode(
        "postgres", "erp", connections_ep.PatchAccessModeBody(access_mode="read"), ORG, request
    )
    assert proxied == ["catalogue", "detail", "delete", "picked", "access"]
    with pytest.raises(HTTPException) as refused:
        connections_ep.patch_connection_token(
            "postgres", "erp", connections_ep.PatchTokenBody(access_token="t"), ORG
        )
    assert refused.value.status_code == 501

    # The loopback save and the picked-file untag still use the org's vault,
    # unlayered and unstamped.
    saved = connections_ep.save_connection_direct(
        DirectSaveRequest(connector_id="postgres", name="erp", values=dict(ERP)), ORG
    )
    vault = vault_for_scope(ORG)
    assert not isinstance(vault, LayeredDataVault)
    assert Path(vault._dir) == hosted / "shared" / ORG_ID / "data-vault"
    assert CREATED_BY_FIELD not in vault.read_record("postgres", saved["name"])["fields"]
    ConnectionsService(ORG).merge_picked_files("postgres", "erp", [{"id": "f1", "projects": ["p"]}])
    assert connections_ep.delete_picked_file("postgres", "erp", "f1", "p", ORG)["ok"] is True
