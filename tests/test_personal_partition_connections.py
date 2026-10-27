"""Saved connections on a deployment that trusts identity headers.

Each person's turn reads their own vault: the connections the agent sees and
the DS_* secrets it can use, the labels it sets, and the secrets an artifact's
live backend is started with. A turn bound to no scope is refused there rather
than handed the base vault. Connections left shared keep today's one vault,
and the desktop reads its vault as it always has.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import use_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness.tools import _cowork_label_connection
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.services.connectors.persist import turn_vault_for_scope, vault_for_scope
from tests._turns import built_turn

TOKEN = "partition-connections-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
BEARER = {"Authorization": f"Bearer {TOKEN}"}
A = {**BEARER, "X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT}


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)


@pytest.fixture
def stack(personal_deployment, monkeypatch, tmp_path):
    """A trusted-identity deployment whose stores live under the test's folder.
    The two people's rows are removed afterwards, so later modules that list
    every row don't see them."""
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "base-vault"))
    get_app_settings.cache_clear()
    yield tmp_path
    with get_open_session() as db:
        db.exec(delete(Conversation).where(Conversation.org_id.in_([PARTITION_A, PARTITION_B])))
        db.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        db.commit()
    get_app_settings.cache_clear()


def _save_sales(scope: TenantScope | None, *, secret: str) -> None:
    vault_for_scope(scope).save(
        "postgres", "sales", {"host": "db.example", "password": secret}, secure_keys=["password"]
    )


def _names(vault) -> set[tuple[str, str]]:
    return {(c["engine"], c["name"]) for c in vault.list_connections()}


# ── the read path every turn and backend uses ───────────────────────────────


def test_each_person_reads_only_their_own_vault(stack):
    _save_sales(ALICE, secret="alice-secret")

    assert _names(turn_vault_for_scope(ALICE)) == {("postgres", "sales")}
    assert _names(turn_vault_for_scope(BOB)) == set()
    # The operator view reads the base vault, which holds nobody's.
    assert _names(turn_vault_for_scope(LOCAL_SCOPE)) == set()


def test_an_unbound_read_is_refused_on_a_trusted_deployment(stack):
    with pytest.raises(MissingTenantScopeError):
        turn_vault_for_scope(None)


def test_shared_connections_keep_one_vault(stack):
    shared = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS})
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=shared)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=shared)
    _save_sales(alice, secret="team-secret")

    assert _names(turn_vault_for_scope(bob)) == {("postgres", "sales")}
    assert _names(turn_vault_for_scope(LOCAL_SCOPE)) == {("postgres", "sales")}


def test_desktop_reads_the_configured_vault(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    get_app_settings.cache_clear()
    try:
        _save_sales(None, secret="desktop-secret")
        assert turn_vault_for_scope(None)._dir == tmp_path / "vault"
        assert _names(turn_vault_for_scope(None)) == {("postgres", "sales")}
    finally:
        get_app_settings.cache_clear()


# ── one record lock per vault ───────────────────────────────────────────────


def _start(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def test_a_persons_delete_never_drops_another_persons_record_lock(stack):
    """A and B each own `postgres/sales`, two records in two vaults. B's
    delete of B's own must not drop the lock A's in-flight write holds, or
    A's next write would run beside it and whichever saved last would revert
    the other. Nor does B's write wait on A's."""
    from cowork.services.connectors.connections import ConnectionsService
    from cowork.services.connectors.persist import own_vault_root
    from cowork.services.connectors.vault_lock import lock_for

    _save_sales(ALICE, secret="alice-secret")
    _save_sales(BOB, secret="bob-secret")
    alices = ConnectionsService(ALICE)
    bobs = ConnectionsService(BOB)

    with lock_for(vault_root=own_vault_root(ALICE), engine="postgres", name="sales"):
        # A's first write is in flight.
        b_write = _start(bobs.patch_token, "postgres", "sales", {"status": "b"})
        b_write.join(timeout=5)
        assert not b_write.is_alive(), "B's write waited on A's record lock"
        assert bobs.delete("postgres", "sales")
        a_second = _start(alices.patch_token, "postgres", "sales", {"status": "a"})
        a_second.join(timeout=0.3)
        assert a_second.is_alive(), "A's second write ran beside the first"
    a_second.join(timeout=5)

    assert not a_second.is_alive()
    assert vault_for_scope(ALICE).read_record("postgres", "sales")["fields"]["status"] == "a"


# ── an in-process turn ──────────────────────────────────────────────────────


async def test_a_turn_sees_only_the_askers_connections(monkeypatch, stack):
    _save_sales(ALICE, secret="alice-secret")

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stack, scope=ALICE) as turn:
        assert _names(turn.config.data_vault) == {("postgres", "sales")}
    async with built_turn(monkeypatch=monkeypatch, tmp_path=stack, scope=BOB) as turn:
        assert _names(turn.config.data_vault) == set()


async def test_same_named_private_connections_resolve_to_the_asker_in_each_turn(monkeypatch, stack):
    _save_sales(ALICE, secret="alice-secret")
    _save_sales(BOB, secret="bob-secret")
    for person, own, other in ((ALICE, "alice-secret", "bob-secret"), (BOB, "bob-secret", "alice-secret")):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=stack, scope=person) as turn:
            env = turn.config.data_vault.env_for("postgres", "sales")
            assert env["DS_POSTGRES_SALES__PASSWORD"] == own
            assert other not in env.values()


async def test_a_turn_bound_to_no_scope_is_refused(monkeypatch, stack):
    """Only connections are personal here, so the vault read is the one step
    of the turn that needs a scope: reading the base vault instead would let
    this turn start with the operator view's connection."""
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "connections")
    get_app_settings.cache_clear()
    _save_sales(LOCAL_SCOPE, secret="operator-secret")

    with pytest.raises(MissingTenantScopeError):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=stack, scope=None):
            pass


# ── labels the agent sets ───────────────────────────────────────────────────


async def test_the_label_tool_labels_only_the_askers_connection(stack):
    _save_sales(ALICE, secret="alice-secret")
    request = {"engine": "postgres", "name": "sales", "label": "Finance"}

    with use_settings_scope(BOB):
        refused = await _cowork_label_connection(MagicMock(), request)
    assert "no connection" in refused.lower()
    assert "_user_label" not in (vault_for_scope(ALICE).read_record("postgres", "sales")["fields"])

    with use_settings_scope(ALICE):
        labeled = await _cowork_label_connection(MagicMock(), request)
    assert "Finance" in labeled
    assert vault_for_scope(ALICE).read_record("postgres", "sales")["fields"]["_user_label"] == "Finance"


async def test_the_label_tool_refuses_an_unbound_turn(stack):
    _save_sales(LOCAL_SCOPE, secret="operator-secret")

    result = await _cowork_label_connection(
        MagicMock(), {"engine": "postgres", "name": "sales", "label": "Finance"}
    )

    assert result.startswith("label_connection: could not set label")
    assert "_user_label" not in vault_for_scope(LOCAL_SCOPE).read_record("postgres", "sales")["fields"]


# ── an artifact's live backend ──────────────────────────────────────────────


def _backend_artifact(project_dir: Path, slug: str) -> Path:
    folder = project_dir / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    (folder / "backend.py").write_text("print('hi')\n")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": uuid4().hex,
                "slug": slug,
                "type": "fullstack-stateful-app",
                "datasources": [{"engine": "postgres", "name": "sales"}],
            }
        )
    )
    return folder


async def test_a_backend_gets_only_its_owners_connection_secrets(monkeypatch, stack):
    from cowork.services import artifacts

    _save_sales(ALICE, secret="alice-secret")
    launched: dict[str, dict[str, str]] = {}

    async def fake_launch(*, slug, artifact_folder, scratchpad_pool, tracked_backends, ds_env, health_timeout):
        launched[str(artifact_folder)] = ds_env
        return {"port": 40001, "pid": 1}

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", fake_launch)
    monkeypatch.setattr(artifacts, "_LAUNCHED_BACKENDS", {})
    a_folder = _backend_artifact(stack / "shared" / PARTITION_A / "projects" / "general", "app")
    b_folder = _backend_artifact(stack / "shared" / PARTITION_B / "projects" / "general", "app")

    await artifacts._launch_backend_locked(a_folder, project_root=a_folder.parents[2], scope=ALICE)
    await artifacts._launch_backend_locked(b_folder, project_root=b_folder.parents[2], scope=BOB)

    assert "alice-secret" in launched[str(a_folder)].values()
    assert launched[str(b_folder)] == {}

    # The recipient's copy attaches its own same-named connection. Resolving
    # by the author's scope would pass the missing-connection check above
    # only if the launch wiring were mocked too broadly.
    _save_sales(BOB, secret="bob-secret")
    await artifacts._launch_backend_locked(b_folder, project_root=b_folder.parents[2], scope=BOB)
    assert "bob-secret" in launched[str(b_folder)].values()
    assert "alice-secret" not in launched[str(b_folder)].values()


# ── local OAuth sign-in ─────────────────────────────────────────────────────


async def test_local_oauth_start_is_refused_for_a_person(stack, monkeypatch):
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.connectors import oauth
    from cowork.schemas.connectors import OAuthStartRequest

    monkeypatch.setattr(
        oauth.oauth_service, "start",
        lambda *_a, **_k: pytest.fail("a person's sign-in must not start the local flow"),
    )

    with pytest.raises(HTTPException) as refused:
        await oauth.start_oauth("gmail", MagicMock(), ALICE, OAuthStartRequest())

    assert refused.value.status_code == 403


async def test_local_oauth_start_is_refused_where_connections_are_shared(stack, monkeypatch):
    """Even with connections left shared, a trusted-identity deployment refuses
    the local sign-in: the provider's redirect to the callback carries no
    identity, so the connection would be saved with no creator."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.connectors import oauth
    from cowork.schemas.connectors import OAuthStartRequest

    monkeypatch.setattr(
        oauth.oauth_service, "start",
        lambda *_a, **_k: pytest.fail("the sign-in must not start where identity headers are trusted"),
    )
    shared = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS})
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=shared)

    with pytest.raises(HTTPException) as refused:
        await oauth.start_oauth("gmail", MagicMock(), alice, OAuthStartRequest())

    assert refused.value.status_code == 403


async def test_local_oauth_start_is_refused_for_the_operator_view(stack, monkeypatch):
    """The operator view (no person) is refused too: the gate is the trusted
    deployment, not the caller's partition."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.connectors import oauth
    from cowork.schemas.connectors import OAuthStartRequest

    monkeypatch.setattr(
        oauth.oauth_service, "start",
        lambda *_a, **_k: pytest.fail("the operator view's sign-in must not start the local flow"),
    )

    with pytest.raises(HTTPException) as refused:
        await oauth.start_oauth("gmail", MagicMock(), LOCAL_SCOPE, OAuthStartRequest())

    assert refused.value.status_code == 403


@pytest.fixture
def trusted_app(stack, monkeypatch):
    """A trusted stack create_app accepts: the bearer token, a home folder,
    and the operator view for a request with no identity."""
    monkeypatch.setenv("COWORK_HOME", str(stack / "home"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    return stack


@pytest.fixture
def pending_gmail_sign_in(trusted_app, monkeypatch):
    """A trusted_app stack with a Gmail sign-in pending, as a start would
    leave it, and the provider's side stubbed."""
    from datetime import datetime, timezone

    from cowork.common.settings.app_settings import OAuthSettings
    from cowork.services.connectors.oauth import google

    google.oauth_service._store(OAuthSettings()).set_pending(
        "gmail",
        state="s-1",
        verifier="verifier",
        redirect_uri="http://cowork.example/callback",
        started_at=datetime.now(timezone.utc).isoformat(),
        client_id="cid",
        client_secret="csecret",
    )
    monkeypatch.setattr(
        google.OAuthService, "_exchange_code", lambda self, **_kw: {"access_token": "at-1", "expires_in": 3600}
    )
    monkeypatch.setitem(google._USERINFO_FETCHERS, "gmail", lambda _token: {"email": "team@example.com"})


def _finish_gmail_sign_in(headers: dict) -> str:
    # No `with`: skipping the lifespan skips boot migrations and workers.
    page = TestClient(cowork_server.create_app()).get(
        "/api/v1/connectors/oauth/gmail/callback", params={"code": "c", "state": "s-1"}, headers=headers
    )
    assert page.status_code == 200
    return page.text


def test_a_shared_connections_connection_is_the_creators_own(trusted_app, monkeypatch):
    """Connections shared: a connection records the person who created it, so
    their delete goes through and someone else's answers 403. The connection
    is saved through the person's vault, as a save with their credentials
    would."""
    from cowork.api.v1.endpoints.connectors import connections as connections_ep
    from cowork.services.connectors.layered_vault import CREATED_BY_FIELD, SHARED_CONNECTION_REFUSAL

    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "memory")
    get_app_settings.cache_clear()
    revoked = []
    monkeypatch.setattr(connections_ep.oauth_service, "revoke", lambda engine, name, *_a, **_k: revoked.append(name))

    shared = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS})
    alice_shared = _person(partition=PARTITION_A, sub=SUB_A, resources=shared)
    engine, name = "gmail", "team@example.com"
    vault_for_scope(alice_shared).save(engine, name, {"access_token": "at-1"})

    assert vault_for_scope(LOCAL_SCOPE).read_record(engine, name)["fields"][CREATED_BY_FIELD] == SUB_A
    client = TestClient(cowork_server.create_app())
    path = f"/api/v1/connectors/connections/{engine}/{name}"
    bob = {**BEARER, "X-User-Id": SUB_B, "X-Organization-Id": DEPLOYMENT}
    refused = client.delete(path, headers=bob)
    assert (refused.status_code, refused.json()) == (403, {"detail": SHARED_CONNECTION_REFUSAL})
    assert revoked == []
    assert client.delete(path, headers=A).status_code == 204
    assert revoked == [name]
    assert _names(vault_for_scope(LOCAL_SCOPE)) == set()


def test_the_operator_views_sign_in_saves_into_the_base_vault(pending_gmail_sign_in):
    """Every resource personal: the callback, for a request with no person,
    saves into the base vault, which no person's turn reads."""
    page = _finish_gmail_sign_in(BEARER)

    assert "Gmail connected" in page
    assert {engine for engine, _name in _names(vault_for_scope(LOCAL_SCOPE))} == {"gmail"}
    assert _names(vault_for_scope(ALICE)) == set()


async def test_local_oauth_start_still_runs_without_a_person(monkeypatch):
    from cowork.api.v1.endpoints.connectors import oauth
    from cowork.schemas.connectors import OAuthStartRequest

    started = []
    monkeypatch.setattr(oauth.oauth_service, "start", lambda service, *_a, **_k: started.append(service) or "ok")

    assert await oauth.start_oauth("gmail", MagicMock(), LOCAL_SCOPE, OAuthStartRequest()) == "ok"
    assert started == ["gmail"]
