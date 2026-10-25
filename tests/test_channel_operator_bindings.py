"""Operator setup, including old bindings with no reliable author record."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import cowork.channels.runtime as runtime_module
import cowork.ops.channels as channel_ops
import cowork.server as cowork_server
from cowork.channels.registry import load_first_party_plugins
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_session
from cowork.models.channel import ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import ORG_MANAGE_ROLE
from cowork.schemas.channels import BindingCreateRequest, BindingUpdateRequest
from cowork.services.channel_bindings import ChannelBindingService
from cowork.services.projects import GENERAL_PROJECT_ID

SETUP_REQUIRED = "This chat group needs operator setup."
TOKEN = "personal-deployment-test-token"
PERSON = {
    "Authorization": f"Bearer {TOKEN}",
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}


@pytest.fixture
def db(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as raw:
        raw.add(Project(id=GENERAL_PROJECT_ID, name="general", path=str(tmp_path / "general")))
        raw.commit()
    monkeypatch.setattr(runtime_module, "get_open_session", lambda: Session(engine))
    monkeypatch.setattr(channel_ops, "get_open_session", lambda: Session(engine))
    load_first_party_plugins()
    yield engine
    engine.dispose()


@pytest.fixture
def client(db, personal_deployment, monkeypatch, request):
    mode = getattr(request.node, "callspec", None)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", mode.params.get("mode", "audit") if mode else "audit")
    get_app_settings.cache_clear()
    app = cowork_server.create_app()

    def session():
        with Session(db) as raw:
            yield raw

    app.dependency_overrides[get_session] = session
    test_client = TestClient(app)
    yield test_client
    test_client.close()


def seed_binding(db, *, partition=None, configured=False, private_project=False, private_conversation=False):
    with Session(db) as raw:
        project = Project(name="a-private", path="/tmp/a-private", org_id="private" if private_project else None)
        raw.add(project)
        raw.commit()
        conv = Conversation(topic="private history", project_id=project.id,
                            org_id="private" if private_conversation else None)
        raw.add(conv)
        raw.commit()
        row = ChannelBinding(channel_type="telegram", external_group_id="group", external_thread_key="__default__",
                             org_id=partition, created_by="person" if partition else None,
                             anton_project_id=project.id, anton_conversation_id=conv.id,
                             instructions="Always answer in French.", trigger_rule="always",
                             operator_configured=configured)
        raw.add(row)
        raw.commit()
        raw.refresh(row)
        raw.add(ChannelSession(binding_id=row.id, external_session_key="__default__", anton_session_id=str(conv.id)))
        raw.commit()
        return row.id, project.id, conv.id


def event(text="Which files are in this project?", thread=None):
    return SimpleNamespace(address=SimpleNamespace(platform_id="group", thread_id=thread),
                           message=SimpleNamespace(content=text, is_group=True, is_mention=True))


@pytest.mark.parametrize("mode", ["audit", "enforce"])
@pytest.mark.parametrize("role", ["", ORG_MANAGE_ROLE])
@pytest.mark.parametrize("method", ["post", "patch", "delete"])
def test_person_cannot_change_group_binding(client, db, monkeypatch, mode, role, method):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", mode)
    get_app_settings.cache_clear()
    binding_id, _, _ = seed_binding(db)
    headers = {**PERSON, "X-User-Roles": role}
    path = "/api/v1/channels/bindings"
    body = {"channel_type": "telegram", "external_group_id": "other", "instructions": "changed"}
    if method != "post":
        path += f"/{binding_id}"
        body = {"instructions": "changed"} if method == "patch" else None
    response = client.request(method, path, headers=headers, json=body)
    assert (response.status_code, response.json()) == (403, {"detail": "requires the operator"})
    with Session(db) as raw:
        rows = raw.exec(select(ChannelBinding)).all()
        assert len(rows) == 1
        assert rows[0].instructions == "Always answer in French."


@pytest.mark.parametrize("partition", [None, "private"])
@pytest.mark.parametrize("text", ["Which files?", "/new"])
@pytest.mark.asyncio
async def test_legacy_group_requires_setup_before_any_turn(db, personal_deployment, monkeypatch, partition, text):
    binding_id, project_id, conversation_id = seed_binding(db, partition=partition)
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    run = AsyncMock(return_value=("unsafe reply", False))
    deliver = AsyncMock()
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", deliver)
    await runtime.handle("telegram", event(text))
    run.assert_not_awaited()
    assert deliver.await_args.args[2] == SETUP_REQUIRED
    with Session(db) as raw:
        row = raw.get(ChannelBinding, binding_id)
        assert row.anton_conversation_id == conversation_id
        assert row.instructions == "Always answer in French."
        assert raw.get(Project, project_id) is not None
        assert raw.get(Conversation, conversation_id) is not None


@pytest.mark.parametrize("private_project,private_conversation", [(True, False), (False, True)])
@pytest.mark.asyncio
async def test_operator_marker_does_not_authorize_private_references(db, personal_deployment, monkeypatch,
                                                                   private_project, private_conversation):
    seed_binding(db, configured=True, private_project=private_project, private_conversation=private_conversation)
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    run, deliver = AsyncMock(), AsyncMock()
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", deliver)
    await runtime.handle("telegram", event())
    run.assert_not_awaited()
    assert deliver.await_args.args[2] == SETUP_REQUIRED


def test_operator_full_setup_replaces_legacy_tuple_preserving_private_work(db, personal_deployment):
    binding_id, project_id, conversation_id = seed_binding(db, partition="private", private_project=True,
                                                         private_conversation=True)
    with Session(db) as raw:
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        saved = svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group",
                                                  anton_project_id=GENERAL_PROJECT_ID, instructions="Use English."))
        assert saved.id == binding_id
        row = raw.get(ChannelBinding, binding_id)
        assert row.operator_configured
        assert row.org_id is None and row.created_by is None
        assert row.anton_conversation_id is None
        assert row.instructions == "Use English."
        assert raw.exec(select(ChannelSession)).all() == []
        assert raw.get(Project, project_id) is not None
        assert raw.get(Conversation, conversation_id) is not None


def test_metadata_patch_cannot_bless_legacy_binding(db, personal_deployment):
    binding_id, _, _ = seed_binding(db)
    with Session(db) as raw:
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        with pytest.raises(ValueError, match="operator setup"):
            svc.update(binding_id, BindingUpdateRequest(display_name="innocent rename"))
        assert raw.get(ChannelBinding, binding_id).instructions == "Always answer in French."


def test_operator_cli_set_list_delete_and_private_target_refusal(db, personal_deployment, capsys):
    binding_id, private_id, _ = seed_binding(db, partition="private", private_project=True, private_conversation=True)
    refused = {"channel_type": "telegram", "external_group_id": "group", "anton_project_id": str(private_id)}
    assert channel_ops.main(["bindings", "set"], stdin=io.StringIO(json.dumps(refused))) == 1
    assert "deployment-owned project" in capsys.readouterr().err
    with Session(db) as raw:
        assert raw.get(ChannelBinding, binding_id).instructions == "Always answer in French."
    body = {"channel_type": "telegram", "external_group_id": "group", "anton_project_id": str(GENERAL_PROJECT_ID),
            "trigger_rule": "regex", "trigger_pattern": "hello", "instructions": "Answer briefly."}
    assert channel_ops.main(["bindings", "set"], stdin=io.StringIO(json.dumps(body))) == 0
    assert "next message" in capsys.readouterr().out
    assert channel_ops.main(["bindings", "list"]) == 0
    assert str(binding_id) in capsys.readouterr().out
    assert channel_ops.main(["bindings", "delete", str(binding_id)]) == 0
    capsys.readouterr()
    with Session(db) as raw:
        assert raw.get(ChannelBinding, binding_id) is None


@pytest.mark.asyncio
async def test_operator_setup_controls_next_message_and_retains_safe_conversation(db, personal_deployment, monkeypatch):
    with Session(db) as raw:
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        saved = svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group",
                                                  anton_project_id=GENERAL_PROJECT_ID,
                                                  trigger_rule="regex", trigger_pattern="hello", instructions="Use English."))
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    projects = []

    async def answer(scoped, conversation, *args, **kwargs):
        projects.append(conversation.project_id)
        return "operator reply", False

    run = AsyncMock(side_effect=answer)
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", AsyncMock())
    await runtime.handle("telegram", event("skip"))
    run.assert_not_awaited()
    await runtime.handle("telegram", event("hello"))
    assert projects == [GENERAL_PROJECT_ID]
    assert run.await_args.kwargs["channel_context"].instructions == "Use English."
    with Session(db) as raw:
        prior = raw.get(ChannelBinding, saved.id).anton_conversation_id
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        svc.update(saved.id, BindingUpdateRequest(instructions="Use Spanish."))
        assert raw.get(ChannelBinding, saved.id).anton_conversation_id == prior
    await runtime.handle("telegram", event("hello"))
    assert run.await_args.kwargs["channel_context"].instructions == "Use Spanish."


@pytest.mark.parametrize("partition", [None, "private"])
@pytest.mark.parametrize("thread", [None, "thread-1"])
@pytest.mark.asyncio
async def test_missing_group_never_creates_a_binding_even_with_legacy_ingress_scope(db, personal_deployment,
                                                                                 monkeypatch, partition, thread):
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    run, deliver = AsyncMock(), AsyncMock()
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", deliver)
    await runtime.handle("telegram", event(thread=thread), org_id=partition)
    run.assert_not_awaited()
    assert deliver.await_args.args[2:] == (SETUP_REQUIRED, None)
    with Session(db) as raw:
        assert raw.exec(select(ChannelBinding)).all() == []


@pytest.mark.asyncio
async def test_null_binding_with_private_conversation_project_is_refused(db, personal_deployment, monkeypatch):
    binding_id, _, _ = seed_binding(db, configured=True, private_project=True)
    with Session(db) as raw:
        row = raw.get(ChannelBinding, binding_id)
        row.anton_project_id = None
        raw.add(row)
        raw.commit()
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    run, deliver = AsyncMock(), AsyncMock()
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", deliver)
    await runtime.handle("telegram", event())
    run.assert_not_awaited()
    assert deliver.await_args.args[2] == SETUP_REQUIRED


def test_new_operator_http_binding_is_marked_and_private_target_is_refused(client, db, monkeypatch):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    headers = {"Authorization": f"Bearer {TOKEN}"}
    _, private_id, private_conversation = seed_binding(db, partition="private", private_project=True,
                                                     private_conversation=True)
    for extra in ({"anton_project_id": str(private_id)}, {"anton_conversation_id": str(private_conversation)}):
        response = client.post("/api/v1/channels/bindings", headers=headers,
                               json={"channel_type": "telegram", "external_group_id": "new", **extra})
        assert response.status_code == 400
    response = client.post("/api/v1/channels/bindings", headers=headers,
                           json={"channel_type": "telegram", "external_group_id": "new"})
    assert response.status_code == 201
    from uuid import UUID
    with Session(db) as raw:
        row = raw.get(ChannelBinding, UUID(response.json()["id"]))
        assert row.operator_configured and row.org_id is None and row.created_by is None


@pytest.mark.parametrize("mode", ["local", "org"])
def test_desktop_and_org_member_mutations_remain_available(db, monkeypatch, mode):
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    app = cowork_server.create_app()

    def session():
        with Session(db) as raw:
            yield raw

    app.dependency_overrides[get_session] = session
    headers = PERSON if mode == "org" else {}
    test_client = TestClient(app)
    try:
        response = test_client.post("/api/v1/channels/bindings", headers=headers,
                                    json={"channel_type": "slack", "external_group_id": "member-group"})
        assert response.status_code == 201
        binding_id = response.json()["id"]
        response = test_client.patch(f"/api/v1/channels/bindings/{binding_id}", headers=headers,
                                     json={"instructions": "member instructions"})
        assert response.status_code == 200
        assert response.json()["instructions"] == "member instructions"
        assert test_client.delete(f"/api/v1/channels/bindings/{binding_id}", headers=headers).status_code == 204
    finally:
        test_client.close()
        get_app_settings.cache_clear()


@pytest.mark.parametrize("resources", ["", "conversations,files,pins,schedules"])
@pytest.mark.asyncio
async def test_shared_legacy_binding_stays_untrusted_after_resource_change(db, personal_deployment, monkeypatch, resources):
    from cowork.db.scoped import scope_for_identity

    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "")
    get_app_settings.cache_clear()
    with Session(db) as raw:
        person = ScopedSession(raw, scope_for_identity(org_id="private", user_id="person"))
        binding = ChannelBinding(channel_type="telegram", external_group_id="group", instructions="Always answer in French.")
        person.add(binding)
        person.commit()
        assert binding.org_id is None and binding.created_by is None
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", resources)
    get_app_settings.cache_clear()
    runtime = AntonChannelRuntime(LiveAdapterRegistry())
    run, deliver = AsyncMock(), AsyncMock()
    monkeypatch.setattr(runtime, "_run_anton", run)
    monkeypatch.setattr(runtime, "_deliver", deliver)
    await runtime.handle("telegram", event())
    run.assert_not_awaited()
    assert deliver.await_args.args[2] == SETUP_REQUIRED


def test_invalid_operator_configuration_preserves_every_legacy_field_and_session(db, personal_deployment):
    binding_id, _, conversation_id = seed_binding(db, partition="private")
    with Session(db) as raw:
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        with pytest.raises(ValueError, match="trigger_pattern"):
            svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group", trigger_rule="regex"))
        row = raw.get(ChannelBinding, binding_id)
        assert not row.operator_configured and row.org_id == "private"
        assert row.anton_conversation_id == conversation_id
        assert row.instructions == "Always answer in French."
        assert len(raw.exec(select(ChannelSession)).all()) == 1


def test_operator_same_project_setup_preserves_conversation_but_new_project_detaches(db, personal_deployment):
    binding_id, project_id, conversation_id = seed_binding(db, configured=True)
    with Session(db) as raw:
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group",
                                          anton_project_id=project_id, instructions="New instructions."))
        assert raw.get(ChannelBinding, binding_id).anton_conversation_id == conversation_id
        svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group",
                                          anton_project_id=GENERAL_PROJECT_ID))
        row = raw.get(ChannelBinding, binding_id)
        assert row.anton_conversation_id is None and row.instructions is None
        assert raw.exec(select(ChannelSession)).all() == []
        assert raw.get(Conversation, conversation_id) is not None


def test_operator_setup_uses_default_project_id_not_same_named_private_project(db, personal_deployment):
    with Session(db) as raw:
        raw.add(Project(name="general", path="/tmp/private-general", org_id="private"))
        raw.commit()
        svc = ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE))
        saved = svc.configure(BindingCreateRequest(channel_type="telegram", external_group_id="group"))
        row = raw.get(ChannelBinding, saved.id)
        runtime = AntonChannelRuntime(LiveAdapterRegistry())
        assert runtime._ensure_conversation(ScopedSession(raw, LOCAL_SCOPE), row).project_id == GENERAL_PROJECT_ID


def test_turns_ignore_forged_operator_marker_on_personal_binding(db, personal_deployment):
    binding_id, _, _ = seed_binding(db, partition="private", configured=True)
    with Session(db) as raw:
        assert not ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE)).operator_ready(raw.get(ChannelBinding, binding_id))


def test_marker_is_internal_and_off_mode_edit_revokes_it(db, personal_deployment, monkeypatch):
    assert "operator_configured" not in BindingCreateRequest.model_fields
    assert "operator_configured" not in BindingUpdateRequest.model_fields
    binding_id, _, _ = seed_binding(db, configured=True)
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    with Session(db) as raw:
        ChannelBindingService(ScopedSession(raw, LOCAL_SCOPE)).update(binding_id, BindingUpdateRequest(instructions="changed"))
        assert not raw.get(ChannelBinding, binding_id).operator_configured
