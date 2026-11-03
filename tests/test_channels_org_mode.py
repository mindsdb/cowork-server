"""Channels actually dispatching in org (SaaS) mode.

Everything here was already built org-aware (installations, credentials,
webhook resolution, per-org adapters) but unreachable: `_install_channels`
loaded no plugins and mounted no routes outside local mode, and
`_require_local_channels()` 501-gated the config/lifecycle endpoints. This
file proves the activation itself, not the underlying mechanisms (see
test_channels_tenancy.py / test_channel_webhooks.py for those).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.encoders import jsonable_encoder
from sqlalchemy import event
from sqlmodel import select

from cowork.channels.plugins.discord import DiscordBridge
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, scope_for_org
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.conversations import ConversationService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"
OWNER_A = "33333333-3333-4333-8333-333333333333"

A = {"X-User-Id": USER_A, "X-Organization-Id": ORG_A}
B = {"X-User-Id": USER_B, "X-Organization-Id": ORG_B}
A_ADMIN = {**A, "X-User-Roles": "manage-organization"}


@pytest.fixture(scope="module")
def client():
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from cowork.server import create_app

        # No `with`: skipping the lifespan skips boot migrations (they would
        # collide with the schema conftest already created) and never starts
        # background workers in this suite. Mirrors test_org_isolation_e2e.py.
        test_client = TestClient(create_app())
        try:
            yield test_client
        finally:
            test_client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


# --- activation: plugins load and webhook routes mount in org mode too -----

def test_plugins_endpoint_lists_first_party_plugins_in_org_mode(client):
    res = client.get("/api/v1/channels/plugins", headers=A)
    assert res.status_code == 200
    types = {p["channel_type"] for p in res.json()}
    assert {"slack", "discord"} <= types


def test_slack_webhook_route_is_mounted_in_org_mode(client):
    # Nothing claims this team_id, so the route exists but safely ACK-ignores.
    # A 404 here would mean the route never got mounted at all.
    res = client.post(
        "/api/v1/channels/slack/events",
        json={"type": "event_callback", "team_id": "T-UNMATCHED", "event": {}},
    )
    assert res.status_code == 204


def test_webhook_paths_are_exempt_from_identity_enforcement(client):
    # No identity headers — a normal API path would 401 here (see
    # test_no_identity_is_401 in test_org_isolation_e2e.py); the webhook path
    # must not, since external platforms authenticate with their own signature.
    res = client.post("/api/v1/channels/discord/interactions", json={"type": 1})
    assert res.status_code != 401


# --- config/status/lifecycle: no longer gated by tenancy mode alone --------

def test_channel_status_available_in_org_mode(client):
    res = client.get("/api/v1/channels/status", headers=A)
    assert res.status_code == 200


def test_channel_config_get_available_in_org_mode(client):
    res = client.get("/api/v1/channels/slack/config", headers=A)
    assert res.status_code == 200
    assert res.json()["configured"] is False


def test_channel_config_put_and_get_are_org_scoped(client):
    put = client.put(
        "/api/v1/channels/slack/config", headers=A_ADMIN,
        json={"values": {"bot_token": "xoxb-a", "signing_secret": "secret-a"}},
    )
    assert put.status_code == 200, put.text
    try:
        # Org B never configured slack — its own state, not org A's leaking in.
        other = client.get("/api/v1/channels/slack/config", headers=B)
        assert other.json()["configured"] is False

        mine = client.get("/api/v1/channels/slack/config", headers=A)
        assert mine.json()["configured"] is True
    finally:
        client.delete("/api/v1/channels/slack/config", headers=A_ADMIN)


# --- channels without a per-org routing-key extractor (Telegram, WhatsApp)
# stay locked out of mutation in org mode — configuring them would silently
# never deliver, since nothing resolves their webhook to an org yet ---------

def test_plugins_endpoint_flags_org_readiness_per_channel(client):
    res = client.get("/api/v1/channels/plugins", headers=A)
    by_type = {p["channel_type"]: p for p in res.json()}
    assert by_type["slack"]["org_ready"] is True
    assert by_type["discord"]["org_ready"] is True
    assert by_type["telegram"]["org_ready"] is False
    assert by_type["whatsapp"]["org_ready"] is False


def test_telegram_config_get_stays_open_in_org_mode(client):
    # Read-only, harmless either way — only mutation is gated.
    res = client.get("/api/v1/channels/telegram/config", headers=A)
    assert res.status_code == 200
    assert res.json()["configured"] is False


def test_telegram_config_put_is_gated_in_org_mode(client):
    res = client.put(
        "/api/v1/channels/telegram/config", headers=A_ADMIN,
        json={"values": {"bot_token": "T:tok"}},
    )
    assert res.status_code == 501
    assert "not yet available in org deployments" in res.json()["detail"]


def test_telegram_reload_is_gated_in_org_mode(client):
    res = client.post("/api/v1/channels/telegram/reload", headers=A_ADMIN)
    assert res.status_code == 501
    assert "not yet available in org deployments" in res.json()["detail"]


def test_telegram_setup_is_gated_in_org_mode(client):
    # Telegram DOES implement setup/teardown (unlike Slack/Discord) — this
    # must be the readiness gate, not LifecycleNotImplementedError's 501.
    res = client.post("/api/v1/channels/telegram/setup", headers=A_ADMIN)
    assert res.status_code == 501
    assert "not yet available in org deployments" in res.json()["detail"]


def test_channel_setup_available_in_org_mode(client):
    # Slack has no setup/teardown lifecycle at all, so this still 501s — but
    # for THAT reason, not the tenancy gate ("not available in org deployments").
    res = client.post("/api/v1/channels/slack/setup", headers=A_ADMIN)
    assert res.status_code == 501
    assert "not implemented" in res.json()["detail"]


# --- the channel-agent (harness) setting is per-org, not one shared row ----

def test_channel_agent_endpoints_available_in_org_mode(client):
    res = client.get("/api/v1/channels/agent", headers=A)
    assert res.status_code == 200

    put = client.put("/api/v1/channels/agent", headers=A_ADMIN, json={"harness": "anton"})
    assert put.status_code == 200, put.text


def test_set_channel_agent_write_does_not_leak_to_other_orgs(monkeypatch):
    """The precise thing this change adds: set_channel_agent must write the
    caller's org row, not the one global row every org would otherwise share."""
    import cowork.api.v1.endpoints.channels as channels_ep
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.db.session import get_open_session
    from cowork.schemas.channels import ChannelAgentUpdateRequest
    from cowork.services.settings import SettingService

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    import cowork.common.settings.user_settings as user_settings_mod
    monkeypatch.setattr(channels_ep, "available_harness_ids", lambda: ["anton", "other"])
    monkeypatch.setattr(user_settings_mod, "_harness_options", lambda: ["anton", "other"])

    session = get_open_session()
    scope_a = TenantScope(org_mode=True, org_id=ORG_A)
    scoped_a = ScopedSession(session, scope_a)
    try:
        result = channels_ep.set_channel_agent(
            ChannelAgentUpdateRequest(harness="other"), session, scoped_a
        )
        assert result.harness == "other"
        assert get_user_settings(TenantScope(org_mode=True, org_id=ORG_A)).channels_harness == "other"
        assert get_user_settings(TenantScope(org_mode=True, org_id=ORG_B)).channels_harness != "other"
    finally:
        SettingService(session, scope_a).delete_setting("channels_harness")
        session.close()
        get_app_settings.cache_clear()


# --- streaming/polling ingress is now per-org: a config change reconciles
# the caller's own org's ingress, not local mode's one deployment-global slot.

def test_config_writes_land_in_the_calling_orgs_adapter_slot(client, monkeypatch):
    """sync_channel_ingress looks the adapter up as (channel_type, org_id), so
    a config write that lands in the local slot is invisible to it."""
    adapters = client.app.state.channel_adapters
    # Starting real ingress here would open a Discord Gateway socket; the
    # adapter cache is what this test is about.
    monkeypatch.setattr(client.app.state, "channel_ingress", None)

    try:
        put = client.put(
            "/api/v1/channels/discord/config", headers=A_ADMIN,
            json={"values": {"bot_token": "discord-bot-token"}},
        )
        assert put.status_code == 200, put.text
        assert adapters.get("discord", ORG_A) is not None
        assert adapters.get("discord") is None
    finally:
        delete = client.delete("/api/v1/channels/discord/config", headers=A_ADMIN)

    assert delete.status_code == 204
    assert adapters.get("discord", ORG_A) is None


def test_reload_rebuilds_the_calling_orgs_adapter_slot(client, monkeypatch):
    import asyncio

    adapters = client.app.state.channel_adapters
    monkeypatch.setattr(client.app.state, "channel_ingress", None)

    put = client.put(
        "/api/v1/channels/discord/config", headers=A_ADMIN,
        json={"values": {"bot_token": "discord-bot-token"}},
    )
    assert put.status_code == 200, put.text
    try:
        asyncio.run(adapters.remove("discord", ORG_A))
        assert adapters.get("discord", ORG_A) is None

        res = client.post("/api/v1/channels/discord/reload", headers=A_ADMIN)
        assert res.status_code == 200, res.text
        assert res.json()["active"] is True
        assert adapters.get("discord", ORG_A) is not None
        assert adapters.get("discord") is None
    finally:
        client.delete("/api/v1/channels/discord/config", headers=A_ADMIN)


def test_channel_config_put_reconciles_ingress_for_the_calling_org(client, monkeypatch):
    import cowork.api.v1.endpoints.channels as channels_ep

    calls = []

    async def spy(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(channels_ep, "sync_channel_ingress", spy)
    try:
        res = client.put(
            "/api/v1/channels/discord/config", headers=A_ADMIN,
            json={"values": {"bot_token": "discord-bot-token"}},
        )
        assert res.status_code == 200, res.text
        assert len(calls) == 1
        args, kwargs = calls[0]
        # (manager, adapters, channel_type, org_id) — org_id must be ORG_A,
        # never None, in org mode.
        assert args[2] == "discord"
        assert args[3] == ORG_A
    finally:
        client.delete("/api/v1/channels/discord/config", headers=A_ADMIN)


# --- configuring channels (credentials, lifecycle, the shared channel agent)
# is admin-owned in org mode, same rule settings.py already applies to org
# settings writes — bindings (chat routing) stay open to any org member -----

def test_set_config_requires_org_admin(client):
    res = client.put(
        "/api/v1/channels/slack/config", headers=A,
        json={"values": {"bot_token": "xoxb-a"}},
    )
    assert res.status_code == 403
    assert "org admin" in res.json()["detail"]


def test_delete_config_requires_org_admin(client):
    client.put(
        "/api/v1/channels/slack/config", headers=A_ADMIN,
        json={"values": {"bot_token": "xoxb-a"}},
    )
    try:
        res = client.delete("/api/v1/channels/slack/config", headers=A)
        assert res.status_code == 403
        assert "org admin" in res.json()["detail"]
    finally:
        client.delete("/api/v1/channels/slack/config", headers=A_ADMIN)


def test_reload_requires_org_admin(client):
    res = client.post("/api/v1/channels/slack/reload", headers=A)
    assert res.status_code == 403
    assert "org admin" in res.json()["detail"]


def test_setup_and_teardown_require_org_admin_before_the_readiness_check(client):
    # Telegram is also not org-ready (501) — the admin check must still win,
    # proving it runs first rather than only mattering for ready channels.
    setup = client.post("/api/v1/channels/telegram/setup", headers=A)
    assert setup.status_code == 403
    assert "org admin" in setup.json()["detail"]

    teardown = client.post("/api/v1/channels/telegram/teardown", headers=A)
    assert teardown.status_code == 403
    assert "org admin" in teardown.json()["detail"]


def test_set_channel_agent_requires_org_admin(client):
    res = client.put("/api/v1/channels/agent", headers=A, json={"harness": "anton"})
    assert res.status_code == 403
    assert "org admin" in res.json()["detail"]


def test_bindings_do_not_require_org_admin(client):
    # Chat-routing assignment, not channel configuration — any org member.
    res = client.get("/api/v1/channels/bindings", headers=A)
    assert res.status_code == 200


# --- test-connection: read-only (spends the stored creds on a live API call,
# writes nothing) — any org member, same tier as GET config/status ----------

def test_test_connection_does_not_require_org_admin(client):
    res = client.post("/api/v1/channels/slack/test-connection", headers=A)
    assert res.status_code == 200


@pytest.fixture()
def enforced_foreign_keys():
    """Enforce SQLite foreign keys on the app database for one test, as
    Postgres always does. The pool is disposed after so no connection keeps
    the setting into later tests."""
    session = get_open_session()
    engine = session.get_bind()
    session.close()
    engine.dispose()

    def _fk_on(dbapi_conn: Any, _record: Any, _proxy: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    event.listen(engine, "checkout", _fk_on)
    try:
        yield
    finally:
        event.remove(engine, "checkout", _fk_on)
        engine.dispose()


def test_owner_removes_a_members_route_after_its_first_message(client, tmp_path, enforced_foreign_keys):
    owner = {"X-User-Id": OWNER_A, "X-Organization-Id": ORG_A, "X-User-Roles": "manage-organization"}
    project_id = _org_project(tmp_path)
    created = client.post(
        "/api/v1/channels/bindings", headers=A,
        json={"channel_type": "discord", "external_group_id": "route-1", "anton_project_id": str(project_id)},
    )
    assert created.status_code == 201, created.text
    binding_id = UUID(created.json()["id"])
    conversation_id = _serve_first_discord_message(binding_id, project_id)
    try:
        res = client.delete(f"/api/v1/channels/bindings/{binding_id}", headers=owner)
        assert res.status_code == 204, res.text

        listed = client.get("/api/v1/channels/bindings", headers=owner)
        assert str(binding_id) not in {b["id"] for b in listed.json()}
        session = get_open_session()
        try:
            assert not session.exec(select(ChannelSession).where(ChannelSession.binding_id == binding_id)).all()
        finally:
            session.close()
    finally:
        _drop_rows(binding_id, conversation_id, project_id)


def _org_project(tmp_path) -> UUID:
    """A project in ORG_A for the route to point at."""
    session = get_open_session()
    try:
        project = Project(name=f"routes-{uuid4().hex[:8]}", path=str(tmp_path), org_id=ORG_A)
        session.add(project)
        session.commit()
        return project.id
    finally:
        session.close()


def _serve_first_discord_message(binding_id: UUID, project_id: UUID) -> UUID:
    """The runtime's first-message bookkeeping, under its org scope with no user."""
    runtime = AntonChannelRuntime(LiveAdapterRegistry(), default_project_id=project_id)
    message = DiscordBridge({})._normalize_message(
        {"id": "m-route-1", "channel_id": "route-1", "content": "hello", "author": {"id": "42"}}
    )
    session = get_open_session()
    try:
        scoped = ScopedSession(session, scope_for_org(ORG_A))
        binding = scoped.get(ChannelBinding, binding_id)
        conversation = runtime._ensure_conversation(scoped, binding)
        runtime._touch_channel_session(scoped, binding, conversation, message)
        return conversation.id
    finally:
        session.close()


def _drop_rows(binding_id: UUID, conversation_id: UUID, project_id: UUID) -> None:
    """Remove what the test wrote, on the raw session: nobody owns the conversation."""
    session = get_open_session()
    try:
        for row in session.exec(select(ChannelSession).where(ChannelSession.binding_id == binding_id)).all():
            session.delete(row)
        session.flush()
        for model, row_id in ((ChannelBinding, binding_id), (Conversation, conversation_id), (Project, project_id)):
            row = session.get(model, row_id)
            if row is not None:
                session.delete(row)
                session.flush()
        session.commit()
    finally:
        session.close()


OWNER_A_ADMIN = {"X-User-Id": OWNER_A, "X-Organization-Id": ORG_A, "X-User-Roles": "manage-organization"}
B_ADMIN = {**B, "X-User-Roles": "manage-organization"}
CHANNEL_ITEM_FIELDS = {"id", "title", "channelType", "createdAt", "updatedAt"}


@pytest.fixture()
def channel_conversation(client, tmp_path):
    """A route a member added, the conversation the runtime created on its
    first message, and that message saved the way the runtime saves it."""
    project_id = _org_project(tmp_path)
    created = client.post(
        "/api/v1/channels/bindings", headers=A,
        json={"channel_type": "discord", "external_group_id": "route-2", "anton_project_id": str(project_id)},
    )
    assert created.status_code == 201, created.text
    binding_id = UUID(created.json()["id"])
    conversation_id = _serve_first_discord_message(binding_id, project_id)
    session = get_open_session()
    try:
        ConversationService(ScopedSession(session, scope_for_org(ORG_A))).save_user_message(
            conversation_id, "hello", created_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
    finally:
        session.close()
    yield conversation_id
    _drop_messages(conversation_id)
    _drop_rows(binding_id, conversation_id, project_id)


def _drop_messages(conversation_id: UUID) -> None:
    """Remove the conversation's messages, on the raw session: nobody owns it."""
    session = get_open_session()
    try:
        for row in session.exec(select(Message).where(Message.conversation_id == conversation_id)).all():
            session.delete(row)
        session.commit()
    finally:
        session.close()


def _runtime_view_of_messages(conversation_id: UUID) -> list[dict]:
    """The history as the runtime's own scope reads it back."""
    session = get_open_session()
    try:
        return ConversationService(ScopedSession(session, scope_for_org(ORG_A))).get_messages(conversation_id)
    finally:
        session.close()


def test_org_admin_lists_and_reads_a_channel_conversation(client, channel_conversation):
    cid = str(channel_conversation)

    listed = client.get("/api/v1/channel-conversations/", headers=OWNER_A_ADMIN)
    assert listed.status_code == 200, listed.text
    item = next(i for i in listed.json()["conversations"] if i["id"] == cid)
    assert set(item) == CHANNEL_ITEM_FIELDS
    assert (item["title"], item["channelType"]) == ("discord: route-2", "discord")

    one = client.get(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN)
    assert one.status_code == 200, one.text
    assert one.json() == item

    items = client.get(f"/api/v1/channel-conversations/{cid}/items", headers=OWNER_A_ADMIN)
    assert items.status_code == 200, items.text
    assert items.json() == jsonable_encoder(_runtime_view_of_messages(channel_conversation))
    assert [m["content"] for m in items.json()] == ["hello"]

    page = client.get(f"/api/v1/channel-conversations/{cid}/items?limit=1", headers=OWNER_A_ADMIN)
    assert page.status_code == 200, page.text
    assert set(page.json()) == {"items", "hasMore", "nextBefore"}


def test_channel_conversations_list_newest_activity_first(client, channel_conversation, tmp_path):
    older = str(channel_conversation)
    project_id = _org_project(tmp_path)
    created = client.post(
        "/api/v1/channels/bindings", headers=A,
        json={"channel_type": "discord", "external_group_id": "route-3", "anton_project_id": str(project_id)},
    )
    binding_id = UUID(created.json()["id"])
    newer_id = _serve_first_discord_message(binding_id, project_id)
    session = get_open_session()
    try:
        ConversationService(ScopedSession(session, scope_for_org(ORG_A))).save_user_message(
            newer_id, "later", created_at=datetime.now(timezone.utc)
        )
    finally:
        session.close()
    try:
        listed = client.get("/api/v1/channel-conversations/", headers=OWNER_A_ADMIN).json()["conversations"]
        assert [i["id"] for i in listed if i["id"] in {older, str(newer_id)}] == [str(newer_id), older]
        first = client.get("/api/v1/channel-conversations/?limit=1", headers=OWNER_A_ADMIN).json()["conversations"]
        assert [i["id"] for i in first] == [str(newer_id)]
        for bad in (0, 201):
            assert client.get(f"/api/v1/channel-conversations/?limit={bad}", headers=OWNER_A_ADMIN).status_code == 422
    finally:
        _drop_messages(newer_id)
        _drop_rows(binding_id, newer_id, project_id)


def test_members_and_other_orgs_cannot_reach_channel_conversations(client, channel_conversation):
    cid = str(channel_conversation)
    for path in ("/", f"/{cid}", f"/{cid}/items"):
        refused = client.get(f"/api/v1/channel-conversations{path}", headers=A)
        assert refused.status_code == 403, refused.text

    other_list = client.get("/api/v1/channel-conversations/", headers=B_ADMIN)
    assert cid not in {i["id"] for i in other_list.json()["conversations"]}
    for path in (f"/{cid}", f"/{cid}/items"):
        assert client.get(f"/api/v1/channel-conversations{path}", headers=B_ADMIN).status_code == 404


def test_only_marked_ownerless_conversations_are_channel_conversations(client, tmp_path):
    project_id = _org_project(tmp_path)
    own = client.post(
        "/api/v1/conversations/", headers=A, json={"topic": "discord: plans", "projectId": str(project_id)},
    )
    assert own.status_code == 201, own.text
    session = get_open_session()
    try:
        unmarked = Conversation(topic="discord: old", project_id=project_id, org_id=ORG_A, created_by=None)
        session.add(unmarked)
        session.commit()
        unmarked_id = unmarked.id
    finally:
        session.close()
    try:
        listed = {i["id"] for i in client.get("/api/v1/channel-conversations/", headers=OWNER_A_ADMIN).json()["conversations"]}
        for cid in (own.json()["id"], str(unmarked_id)):
            assert cid not in listed
            assert client.get(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN).status_code == 404
            assert client.delete(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN).status_code == 404
        session = get_open_session()
        try:
            assert session.get(Conversation, UUID(own.json()["id"])) is not None
            assert session.get(Conversation, unmarked_id) is not None
        finally:
            session.close()
    finally:
        session = get_open_session()
        try:
            for cid in (UUID(own.json()["id"]), unmarked_id):
                row = session.get(Conversation, cid)
                if row is not None:
                    session.delete(row)
            session.flush()
            session.delete(session.get(Project, project_id))
            session.commit()
        finally:
            session.close()


def test_org_admin_still_cannot_write_to_a_channel_conversation(client, channel_conversation, granted_product_permissions):
    cid = str(channel_conversation)
    assert client.get(f"/api/v1/conversations/{cid}", headers=OWNER_A_ADMIN).status_code == 404
    assert client.patch(f"/api/v1/conversations/{cid}", headers=OWNER_A_ADMIN, json={"topic": "x"}).status_code == 404
    moved = client.post(f"/api/v1/conversations/{cid}/move", headers=OWNER_A_ADMIN, json={"projectId": str(uuid4())})
    assert moved.status_code == 404
    first = _runtime_view_of_messages(channel_conversation)[0]["id"]
    assert client.delete(f"/api/v1/conversations/{cid}/turns/{first}", headers=OWNER_A_ADMIN).status_code == 404
    turn = client.post(
        "/api/v1/responses/", headers=OWNER_A_ADMIN, json={"input": "hi", "stream": True, "conversation": cid},
    )
    assert turn.status_code == 404, turn.text
    assert [m["content"] for m in _runtime_view_of_messages(channel_conversation)] == ["hello"]


def test_channel_conversation_routes_do_not_exist_outside_org_mode(client, channel_conversation, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        cid = str(channel_conversation)
        for path in ("/", f"/{cid}", f"/{cid}/items"):
            assert client.get(f"/api/v1/channel-conversations{path}", headers=OWNER_A_ADMIN).status_code == 404
        assert client.delete(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN).status_code == 404
    finally:
        monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
        get_app_settings.cache_clear()
    assert _runtime_view_of_messages(channel_conversation), "the conversation must survive outside org mode"


def test_org_admin_deletes_a_channel_conversation(client, channel_conversation, enforced_foreign_keys):
    cid = str(channel_conversation)
    session = get_open_session()
    try:
        binding_id = session.exec(
            select(ChannelBinding.id).where(ChannelBinding.anton_conversation_id == channel_conversation)
        ).one()
    finally:
        session.close()

    assert client.delete(f"/api/v1/channel-conversations/{cid}", headers=A).status_code == 403
    assert client.delete(f"/api/v1/channel-conversations/{cid}", headers=B_ADMIN).status_code == 404

    deleted = client.delete(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN)
    assert deleted.status_code == 204, deleted.text

    session = get_open_session()
    try:
        assert session.get(Conversation, channel_conversation) is None
        assert not session.exec(select(Message).where(Message.conversation_id == channel_conversation)).all()
        binding = session.get(ChannelBinding, binding_id)
        assert binding is not None and binding.anton_conversation_id is None
        assert not session.exec(select(ChannelSession).where(ChannelSession.binding_id == binding_id)).all()
    finally:
        session.close()
    assert client.get(f"/api/v1/channel-conversations/{cid}", headers=OWNER_A_ADMIN).status_code == 404
