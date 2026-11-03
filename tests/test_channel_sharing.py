"""Which inbound messages count as a shared channel.

Only a conversation that came from a shared channel is ever readable to org
admins; a direct message to the bot, one-to-one or a group DM, is readable by
nobody in the app. Each adapter decides from its own platform's fields, and
anything it cannot place counts as not shared.
"""
from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.channels.plugins.discord import DiscordBridge
from cowork.channels.plugins.slack import SlackBridge
from cowork.channels.plugins.telegram import TelegramBridge
from cowork.channels.plugins.whatsapp import WhatsAppBridge
from cowork.channels.runtime import AntonChannelRuntime, LiveAdapterRegistry
from cowork.channels.sharing import is_shared
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, scope_for_org
from cowork.models.channel import ChannelBinding
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


def _discord_message(**extra) -> dict:
    return {"id": "1", "channel_id": "42", "content": "hi", "author": {"id": "7"}, **extra}


@pytest.mark.parametrize(
    ("payload", "shared"),
    [
        (_discord_message(guild_id="g1"), True),
        (_discord_message(), False),
        (_discord_message(guild_id=None), False),
        (_discord_message(guild_id=""), False),
    ],
    ids=["server channel", "dm", "null guild", "empty guild"],
)
def test_discord_message(payload, shared):
    event = DiscordBridge({"bot_token": "t"})._normalize_message(payload)

    assert is_shared(event) is shared


@pytest.mark.parametrize(("guild_id", "shared"), [("g1", True), (None, False)], ids=["server", "dm"])
def test_discord_slash_command(guild_id, shared):
    interaction = {
        "type": 2, "id": "i1", "channel_id": "42", "data": {"name": "ask", "options": [{"value": "hi"}]},
        "user": {"id": "7"},
    }
    if guild_id:
        interaction["guild_id"] = guild_id
    events = asyncio.run(DiscordBridge({"bot_token": "t"}).parse_inbound(
        body=json.dumps(interaction).encode(), headers={}, route_name=None,
    ))

    assert [is_shared(e) for e in events] == [shared]


@pytest.mark.parametrize(
    ("event", "shared"),
    [
        ({"type": "message", "channel": "C1", "channel_type": "channel"}, True),
        ({"type": "message", "channel": "G1", "channel_type": "group"}, True),
        ({"type": "message", "channel": "D1", "channel_type": "im"}, False),
        ({"type": "message", "channel": "G2", "channel_type": "mpim"}, False),
        ({"type": "message", "channel": "C2", "channel_type": "mpim"}, False),
        # app_mention carries no channel_type: only a C channel is placed as shared.
        ({"type": "app_mention", "channel": "C3"}, True),
        ({"type": "app_mention", "channel": "G3"}, False),
        ({"type": "app_mention", "channel": "D3"}, False),
    ],
    ids=["public channel", "private channel", "im", "mpim", "mpim with C id", "mention in C", "mention in G", "mention in D"],
)
def test_slack_event(event, shared):
    normalized = SlackBridge({"signing_secret": "s", "bot_token": "x"})._normalize_event(
        {"user": "U1", "text": "hi", "ts": "1700000000.0001", **event}
    )

    assert is_shared(normalized) is shared


@pytest.mark.parametrize(
    ("chat_type", "shared"),
    [("group", True), ("supergroup", True), ("channel", True), ("private", False), (None, False), ("Group", False)],
)
def test_telegram_update(chat_type, shared):
    chat = {"id": -5}
    if chat_type is not None:
        chat["type"] = chat_type
    event = TelegramBridge({"bot_token": "x", "secret_token": "s", "bot_username": "b"})._normalize_update(
        {"update_id": 1, "message": {"message_id": 1, "from": {"id": 42}, "chat": chat, "date": 1700000000, "text": "hi"}}
    )

    assert is_shared(event) is shared


def test_whatsapp_is_never_shared():
    bridge = WhatsAppBridge({"phone_number_id": "p", "access_token": "tok", "app_secret": "k", "verify_token": "v"})
    event = bridge._normalize_message({"id": "w1", "from": "15550001", "type": "text", "text": {"body": "hi"}, "timestamp": "1700000000"})

    assert is_shared(event) is False


ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
MEMBER = "11111111-1111-4111-8111-111111111111"


@pytest.fixture()
def db(monkeypatch):
    """An org-mode database with one project, and a runtime that serves it."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        project = Project(name="routes", path="/routes", org_id=ORG)
        session.add(project)
        session.commit()
        project_id = project.id
    yield engine, AntonChannelRuntime(LiveAdapterRegistry(), default_project_id=project_id), project_id
    get_app_settings.cache_clear()


def _route(engine, conversation_id=None) -> UUID:
    """A route as it may exist from before pinning was refused: written raw."""
    with Session(engine) as session:
        binding = ChannelBinding(
            channel_type="discord", external_group_id=f"chan-{uuid4().hex[:6]}", external_thread_key="__default__",
            org_id=ORG, anton_conversation_id=conversation_id,
        )
        session.add(binding)
        session.commit()
        return binding.id


def _serve(engine, runtime, binding_id, *, guild_id) -> UUID:
    """The runtime's first-message bookkeeping for one Discord message."""
    payload = {"id": uuid4().hex, "channel_id": "c", "content": "hi", "author": {"id": "7"}}
    if guild_id:
        payload["guild_id"] = guild_id
    event = DiscordBridge({"bot_token": "t"})._normalize_message(payload)
    with Session(engine) as session:
        scoped = ScopedSession(session, scope_for_org(ORG))
        binding = scoped.get(ChannelBinding, binding_id)
        return runtime._ensure_conversation(scoped, binding, shared=is_shared(event)).id


def _conversation(engine, conversation_id) -> Conversation:
    with Session(engine) as session:
        return session.get(Conversation, conversation_id)


def test_a_dm_never_lands_in_a_shared_conversation(db):
    engine, runtime, _ = db
    shared_route = _route(engine)
    shared_id = _serve(engine, runtime, shared_route, guild_id="g1")
    dm_route = _route(engine, conversation_id=shared_id)

    dm_id = _serve(engine, runtime, dm_route, guild_id=None)

    assert dm_id != shared_id
    assert _conversation(engine, dm_id).channel_type is None
    with Session(engine) as session:
        assert session.get(ChannelBinding, dm_route).anton_conversation_id == dm_id


def test_a_shared_message_never_lands_in_a_members_own_conversation(db):
    engine, runtime, project_id = db
    with Session(engine) as session:
        own = ConversationService(
            ScopedSession(session, TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER))
        ).create_conversation(topic="mine", project_id=project_id)
    route = _route(engine, conversation_id=own.id)

    served = _serve(engine, runtime, route, guild_id="g1")

    assert served != own.id
    assert _conversation(engine, served).channel_type == "discord"
    assert _conversation(engine, own.id).channel_type is None


def test_a_dm_never_lands_in_a_members_own_conversation(db):
    # A member pinning a route for someone else's DM chat id (a Telegram
    # private chat id is the user's own id) to their own conversation would
    # read that person's DMs; the DM gets a fresh conversation instead.
    engine, runtime, project_id = db
    with Session(engine) as session:
        own = ConversationService(
            ScopedSession(session, TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER))
        ).create_conversation(topic="mine", project_id=project_id)
    route = _route(engine, conversation_id=own.id)

    served = _serve(engine, runtime, route, guild_id=None)

    assert served != own.id
    assert _conversation(engine, served).created_by is None


def test_a_routes_own_conversation_is_reused(db):
    engine, runtime, _ = db
    route = _route(engine)

    first = _serve(engine, runtime, route, guild_id="g1")

    assert _serve(engine, runtime, route, guild_id="g1") == first


def test_desktop_keeps_reusing_a_pinned_conversation(db, monkeypatch):
    engine, runtime, _ = db
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    with Session(engine) as session:
        local = Conversation(topic="desktop chat", project_id=GENERAL_PROJECT_ID)
        session.add(local)
        session.commit()
        local_id = local.id
    with Session(engine) as session:
        binding = ChannelBinding(channel_type="discord", external_group_id="d1", external_thread_key="__default__",
                                 anton_conversation_id=local_id)
        session.add(binding)
        session.commit()
        binding_id = binding.id
    event = DiscordBridge({"bot_token": "t"})._normalize_message(
        {"id": "x", "channel_id": "d1", "content": "hi", "author": {"id": "7"}, "guild_id": "g1"}
    )
    with Session(engine) as session:
        scoped = ScopedSession(session, LOCAL_SCOPE)
        served = runtime._ensure_conversation(scoped, scoped.get(ChannelBinding, binding_id), shared=is_shared(event))

    assert served.id == local_id


def test_an_event_no_adapter_placed_is_not_shared():
    event = DiscordBridge({"bot_token": "t"})._normalize_message(_discord_message(guild_id="g1"))
    del event._shared

    assert is_shared(event) is False
