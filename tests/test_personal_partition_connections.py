"""Saved connections on a deployment that trusts identity headers.

Each person's turn reads their own vault: the connections the agent sees and
the DS_* secrets it can use, the labels it sets, and the secrets an artifact's
live backend is started with. A turn bound to no scope is refused there rather
than handed the base vault. Connections left shared keep today's one vault,
and the desktop reads its vault as it always has.
"""
from __future__ import annotations

import json
import shutil
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import delete

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import Provider, UserSettings, use_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, MissingTenantScopeError, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.tools import _cowork_label_connection
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.services.connectors.persist import turn_vault_for_scope, vault_for_scope

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)


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


# ── an in-process turn ──────────────────────────────────────────────────────


@asynccontextmanager
async def _turn_config(*, monkeypatch, tmp_path, scope: TenantScope | None, conversation_org: str | None):
    """Build one in-process turn's session config, bound to ``scope``."""
    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI,
        planning_model="p", coding_model="c", router_model="r",
        openai_api_key=SecretStr("test-only-key"),
        episodic_memory=False,
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    project_path = tmp_path / "projects" / f"p-{uuid4().hex[:6]}"
    project_path.mkdir(parents=True)
    with get_open_session() as db:
        project = Project(name=project_path.name, path=str(project_path), org_id=conversation_org)
        db.add(project)
        db.commit()
        conversation = Conversation(topic="connections", project_id=project.id, org_id=conversation_org)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        with use_settings_scope(scope) if scope is not None else nullcontext():
            config, scratch, _ = await harness.AntonHarness()._build_chat_session(conversation)
        try:
            yield config
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()


async def test_a_turn_sees_only_the_askers_connections(monkeypatch, stack):
    _save_sales(ALICE, secret="alice-secret")

    async with _turn_config(
        monkeypatch=monkeypatch, tmp_path=stack, scope=ALICE, conversation_org=PARTITION_A
    ) as config:
        assert _names(config.data_vault) == {("postgres", "sales")}
    async with _turn_config(
        monkeypatch=monkeypatch, tmp_path=stack, scope=BOB, conversation_org=PARTITION_B
    ) as config:
        assert _names(config.data_vault) == set()


async def test_a_turn_bound_to_no_scope_is_refused(monkeypatch, stack):
    """Only connections are personal here, so the vault read is the one step
    of the turn that needs a scope: reading the base vault instead would let
    this turn start with the operator view's connection."""
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "connections")
    get_app_settings.cache_clear()
    _save_sales(LOCAL_SCOPE, secret="operator-secret")

    with pytest.raises(MissingTenantScopeError):
        async with _turn_config(monkeypatch=monkeypatch, tmp_path=stack, scope=None, conversation_org=None):
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


async def test_local_oauth_start_still_runs_without_a_person(monkeypatch):
    from cowork.api.v1.endpoints.connectors import oauth
    from cowork.schemas.connectors import OAuthStartRequest

    started = []
    monkeypatch.setattr(oauth.oauth_service, "start", lambda service, *_a, **_k: started.append(service) or "ok")

    assert await oauth.start_oauth("gmail", MagicMock(), LOCAL_SCOPE, OAuthStartRequest()) == "ok"
    assert started == ["gmail"]
