"""The Teams channel's webhook (cowork/channels/plugins/teams.py).

Tokens are signed here with a throwaway RSA key standing in for Microsoft's,
so every check runs against real signatures; replies go to a stand-in
TeamsClient.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from anton.core.dispatch import OutboundMessage
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlmodel import select

from cowork.api.v1.endpoints import channels as channel_routes
from cowork.channels import teams as teams_routes
from cowork.channels.plugins import teams as teams_plugin
from cowork.channels.plugins.teams import SigningKeys, TeamsBridge
from cowork.channels.registry import PluginRegistry, load_first_party_plugins
from cowork.channels.teams import TeamsBot, TeamsSettings
from cowork.channels.webhooks import SignatureError, build_channel_webhook_router
from cowork.models.channel import ChannelEvent
from cowork.db.session import get_open_session

APP_ID = "bot-app-id"
TENANT = "tenant-1"
BOT = f"28:{APP_ID}"
SERVICE_URL = "https://smba.trafficmanager.net/amer/tenant-1/"
CHANNEL = "19:abc@thread.tacv2"
CONVERSATION = f"{CHANNEL};messageid=1001"
KID = "key-1"
WEBHOOK = "/api/v1/channels/teams/api/messages"

MICROSOFT_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
FORGER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks(endorsements=("msteams",)) -> dict:
    entry = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(MICROSOFT_KEY.public_key()))
    entry.update(kid=KID, use="sig", endorsements=list(endorsements))
    return {"keys": [entry]}


def _token(*, key=MICROSOFT_KEY, kid=KID, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": "https://api.botframework.com",
        "aud": APP_ID,
        "exp": now + 3600,
        "nbf": now - 60,
        "serviceurl": SERVICE_URL,
    }
    claims.update(overrides)
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


def _activity(**overrides) -> dict:
    activity = {
        "type": "message",
        "id": "1001",
        "channelId": "msteams",
        "serviceUrl": SERVICE_URL,
        "from": {"id": "29:user-1", "name": "Ada Lovelace", "aadObjectId": "aad-1"},
        "recipient": {"id": BOT, "name": "MindsHub"},
        "conversation": {"id": CONVERSATION, "conversationType": "channel", "tenantId": TENANT},
        "channelData": {"tenant": {"id": TENANT}},
        "text": "<at>MindsHub</at> what were  last week's sales?",
        "entities": [{"type": "mention", "mentioned": {"id": BOT, "name": "MindsHub"}, "text": "<at>MindsHub</at>"}],
    }
    activity.update(overrides)
    return activity


def _headers(token: str | None = None, **extra) -> dict:
    headers = {"authorization": f"Bearer {token or _token()}", "x-ms-tenant-id": TENANT}
    headers.update(extra)
    return {name: value for name, value in headers.items() if value is not None}


class StandInTeams:
    def __init__(self) -> None:
        self.replies: list[tuple] = []

    async def reply(self, service_url, conversation_id, activity_id, text):
        self.replies.append((service_url, conversation_id, activity_id, text))
        return "reply-1"

    async def aclose(self) -> None:
        pass


@pytest.fixture
def identity_headers(monkeypatch):
    state = SimpleNamespace(trust_identity_headers=False)
    monkeypatch.setattr(teams_plugin, "get_app_settings", lambda: state)
    return state


@pytest.fixture
def stand_in():
    return StandInTeams()


def _bridge(stand_in, *, tenant_id=TENANT, endorsements=("msteams",), allowed_users="AAD-1") -> TeamsBridge:
    settings = TeamsSettings(
        bot_app_id=APP_ID, bot_app_secret="secret", tenant_id=tenant_id, allowed_users=allowed_users
    )
    keys = SigningKeys()
    keys.load(_jwks(endorsements))
    return TeamsBridge(settings, keys, TeamsBot(settings, make_client=lambda: stand_in))


@pytest.fixture
def bridge(identity_headers, stand_in):
    return _bridge(stand_in)


def _verify(bridge, activity=None, headers=None):
    bridge.verify_signature(body=json.dumps(activity or _activity()).encode(), headers=headers or _headers())


def test_a_message_from_microsoft_for_this_bot_and_tenant_passes(bridge):
    _verify(bridge)
    assert sum(bridge.refused.values()) == 0


@pytest.mark.parametrize(
    ("activity", "headers", "reason"),
    [
        (None, {"x-ms-tenant-id": TENANT}, "no Microsoft token"),
        (None, _headers(token="not-a-jwt"), "malformed Microsoft token"),
        (None, _headers(token=_token(kid="other-key")), "token signed with an unknown key"),
        (None, _headers(token=_token(key=FORGER_KEY)), "invalid Microsoft token: InvalidSignatureError"),
        (None, _headers(token=_token(aud="another-bot")), "invalid Microsoft token: InvalidAudienceError"),
        (None, _headers(token=_token(iss="https://sts.windows.net/x/")), "invalid Microsoft token: InvalidIssuerError"),
        (None, _headers(token=_token(exp=int(time.time()) - 3600)), "invalid Microsoft token: ExpiredSignatureError"),
        (None, _headers(token=_token(exp=None)), "invalid Microsoft token: MissingRequiredClaimError"),
        (_activity(channelId="slack"), None, "signing key not endorsed for this channel"),
        (_activity(serviceUrl="https://attacker.example/"), None, "serviceUrl doesn't match the token"),
        (_activity(conversation={"id": CONVERSATION, "tenantId": "tenant-2"}), None, "message from another tenant"),
        (_activity(channelData={"tenant": {"id": "tenant-2"}}), None, "message from another tenant"),
        (None, _headers(**{"x-ms-tenant-id": "tenant-2"}), "message from another tenant"),
        (_activity(conversation={"id": CONVERSATION}, channelData={}), {"authorization": f"Bearer {_token()}"},
         "message from another tenant"),
    ],
)
def test_a_message_microsoft_didnt_send_us_is_refused(bridge, activity, headers, reason):
    with pytest.raises(SignatureError, match=reason):
        _verify(bridge, activity, headers)
    assert bridge.refused[reason] == 1


def test_without_a_tenant_setting_every_message_is_refused(identity_headers, stand_in):
    bridge = _bridge(stand_in, tenant_id="")
    with pytest.raises(SignatureError, match="COWORK_TEAMS_TENANT_ID is not set"):
        _verify(bridge)


def test_with_identity_headers_trusted_every_message_is_refused(bridge, identity_headers):
    identity_headers.trust_identity_headers = True
    with pytest.raises(SignatureError, match="identity headers are trusted"):
        _verify(bridge)


def test_a_sender_not_on_the_allowed_list_is_refused(bridge, caplog):
    activity = _activity(**{"from": {"id": "29:user-2", "name": "Someone Else", "aadObjectId": "aad-2"}})
    with caplog.at_level(logging.INFO), pytest.raises(SignatureError, match="sender not on COWORK_TEAMS_ALLOWED_USERS"):
        _verify(bridge, activity)
    assert "message from aad-2, who isn't on COWORK_TEAMS_ALLOWED_USERS" in caplog.text


def test_with_no_allowed_senders_nobody_is_answered(identity_headers, stand_in):
    bridge = _bridge(stand_in, allowed_users="")
    with pytest.raises(SignatureError, match="sender not on COWORK_TEAMS_ALLOWED_USERS"):
        _verify(bridge)


def test_allowed_senders_are_read_from_a_comma_separated_setting(identity_headers, stand_in):
    bridge = _bridge(stand_in, allowed_users="aad-9, AAD-1")
    _verify(bridge)


async def test_the_same_message_id_in_two_conversations_isnt_a_duplicate(bridge):
    other = _activity(conversation={"id": "19:other@thread.tacv2;messageid=1001", "tenantId": TENANT})
    [first] = await bridge.parse_inbound(body=json.dumps(_activity()).encode(), headers=_headers(), route_name=None)
    [second] = await bridge.parse_inbound(body=json.dumps(other).encode(), headers=_headers(), route_name=None)
    assert bridge.dedupe_key(first) != bridge.dedupe_key(second)


def test_a_key_endorsed_only_for_another_channel_is_refused(identity_headers, stand_in):
    bridge = _bridge(stand_in, endorsements=("skype",))
    with pytest.raises(SignatureError, match="not endorsed"):
        _verify(bridge)


async def test_the_message_becomes_a_question_without_the_bots_mention(bridge):
    [event] = await bridge.parse_inbound(body=json.dumps(_activity()).encode(), headers=_headers(), route_name=None)
    assert event.message.content == "what were last week's sales?"
    assert event.message.is_mention is True
    assert event.message.is_group is True
    assert event.message.sender_id == "aad-1"
    assert event.address.platform_id == CHANNEL
    assert event.address.thread_id == "1001"
    assert bridge.dedupe_key(event) == f"{CHANNEL}:1001"


async def test_a_personal_chat_has_no_thread(bridge):
    activity = _activity(conversation={"id": "a:personal-1", "conversationType": "personal", "tenantId": TENANT})
    [event] = await bridge.parse_inbound(body=json.dumps(activity).encode(), headers=_headers(), route_name=None)
    assert event.address.platform_id == "a:personal-1"
    assert event.address.thread_id is None
    assert event.message.is_group is False


async def test_activities_other_than_messages_start_nothing(bridge):
    activity = _activity(type="conversationUpdate")
    assert await bridge.parse_inbound(body=json.dumps(activity).encode(), headers=_headers(), route_name=None) == []


async def test_the_reply_goes_to_the_thread_it_came_from(bridge, stand_in):
    [event] = await bridge.parse_inbound(body=json.dumps(_activity()).encode(), headers=_headers(), route_name=None)
    await bridge.deliver(OutboundMessage(address=event.address, text="Sales were up 4%."))
    assert stand_in.replies == [(SERVICE_URL, CONVERSATION, "1001", "Sales were up 4%.")]


async def test_a_reply_never_goes_to_a_host_off_the_list(bridge, stand_in):
    activity = _activity(serviceUrl="https://attacker.example/amer/")
    [event] = await bridge.parse_inbound(body=json.dumps(activity).encode(), headers=_headers(), route_name=None)
    with pytest.raises(HTTPException):
        await bridge.deliver(OutboundMessage(address=event.address, text="answer"))
    assert stand_in.replies == []


async def test_a_reply_with_nothing_to_reply_to_is_dropped(bridge, stand_in):
    address = teams_plugin.PlatformAddress(channel_type="teams", platform_id="19:unknown", thread_id=None)
    await bridge.deliver(OutboundMessage(address=address, text="answer"))
    assert stand_in.replies == []


def _route(bridge, sunk):
    plugin = teams_plugin.ChannelPlugin(
        channel_type="teams",
        display_name="Microsoft Teams",
        factory=None,
        credentials=teams_plugin.CredentialSchema(),
        webhooks=(teams_plugin.WebhookRoute(path="/api/messages", name="messages"),),
    )
    app = FastAPI()
    app.include_router(
        build_channel_webhook_router(
            plugin, resolver=lambda channel_type: bridge, sink=None, scheduler=lambda job: sunk.append(job) or job.close()
        ),
        prefix="/api/v1/channels",
    )
    return TestClient(app)


def test_through_the_webhook_a_forged_message_gets_401_and_runs_nothing(bridge):
    sunk: list = []
    response = _route(bridge, sunk).post(
        WEBHOOK, content=json.dumps(_activity()).encode(), headers=_headers(token=_token(key=FORGER_KEY))
    )
    assert response.status_code == 401
    assert sunk == []


@pytest.fixture
def stored_events():
    """Removes the inbound events a test stores, which other tests count."""
    keys: list[str] = []
    yield keys
    session = get_open_session()
    try:
        for event in session.exec(select(ChannelEvent).where(ChannelEvent.dedupe_key.in_(keys))).all():
            session.delete(event)
        session.commit()
    finally:
        session.close()


def test_through_the_webhook_a_message_from_microsoft_is_accepted_once(bridge, stored_events):
    sunk: list = []
    client = _route(bridge, sunk)
    activity = _activity(id=f"dedupe-{time.time_ns()}")
    stored_events.append(f"{CHANNEL}:{activity['id']}")
    for _ in range(2):
        response = client.post(WEBHOOK, content=json.dumps(activity).encode(), headers=_headers())
        assert response.status_code == 200
    assert len(sunk) == 1


async def test_signing_keys_load_from_microsofts_metadata():
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL(teams_plugin.OPENID_CONFIG_URL):
            return httpx.Response(200, json={"jwks_uri": "https://login.botframework.com/v1/.well-known/keys"})
        return httpx.Response(200, json=_jwks())

    keys = SigningKeys(httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    await keys.refresh()
    key, endorsements = keys.get(KID)
    assert endorsements == {"msteams"}


async def test_unknown_keys_start_a_fetch_at_most_every_five_minutes(monkeypatch):
    fetches = []

    def answer(request: httpx.Request) -> httpx.Response:
        fetches.append(request.url)
        return httpx.Response(503)

    keys = SigningKeys(httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    keys.load(_jwks())
    clock = [time.time()]
    monkeypatch.setattr(teams_plugin.time, "time", lambda: clock[0])
    clock[0] += teams_plugin.KEYS_MIN_REFRESH_S + 1
    for _ in range(5):
        assert keys.get("unknown") is None
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert len(fetches) == 1
    clock[0] += teams_plugin.KEYS_MIN_REFRESH_S + 1
    assert keys.get("unknown") is None
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(fetches) == 2


async def test_a_failed_key_fetch_keeps_the_keys_it_had(caplog):
    keys = SigningKeys(httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))))
    keys.load(_jwks())
    with caplog.at_level(logging.WARNING):
        await keys.refresh()
    assert keys.get(KID) is not None
    assert "couldn't load Microsoft's signing keys" in caplog.text


def test_the_config_routes_refuse_a_channel_set_in_the_environment(monkeypatch):
    registry = PluginRegistry()
    registry.register(
        teams_plugin.ChannelPlugin(
            channel_type="teams", display_name="Microsoft Teams", factory=None,
            credentials=teams_plugin.CredentialSchema(), credentials_from_env=True,
        )
    )
    monkeypatch.setattr("cowork.channels.registry.default_registry", registry)
    with pytest.raises(HTTPException) as refused:
        channel_routes._refuse_env_channel("teams")
    assert refused.value.status_code == 409
    channel_routes._refuse_env_channel("slack")


def test_every_route_that_changes_a_channel_checks_it():
    import inspect

    for handler in (
        channel_routes.set_config,
        channel_routes.delete_config,
        channel_routes.reload_channel,
        channel_routes.setup_channel,
        channel_routes.teardown_channel,
    ):
        assert "_refuse_env_channel(channel_type)" in inspect.getsource(handler), handler.__name__


def test_teams_registers_only_when_configured_and_available(monkeypatch, identity_headers):
    monkeypatch.setattr(teams_plugin, "teams_available", lambda: True)
    monkeypatch.delenv("COWORK_TEAMS_BOT_APP_ID", raising=False)
    assert teams_plugin._build() is None
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_ID", APP_ID)
    plugin = teams_plugin._build()
    assert plugin.credentials_from_env
    assert [route.path for route in plugin.webhooks] == ["/api/messages"]
    monkeypatch.setattr(teams_plugin, "teams_available", lambda: False)
    assert teams_plugin._build() is None


def test_teams_doesnt_register_where_identity_headers_are_trusted(monkeypatch, identity_headers):
    monkeypatch.setattr(teams_plugin, "teams_available", lambda: True)
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_ID", APP_ID)
    identity_headers.trust_identity_headers = True
    assert teams_plugin._build() is None


def test_registering_says_what_the_channel_still_needs(monkeypatch, identity_headers, caplog):
    monkeypatch.setattr(teams_plugin, "teams_available", lambda: True)
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_ID", APP_ID)
    for name in ("COWORK_TEAMS_BOT_APP_SECRET", "COWORK_TEAMS_TENANT_ID", "COWORK_TEAMS_ALLOWED_USERS"):
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level(logging.WARNING):
        assert teams_plugin._build() is not None
    assert (
        "COWORK_TEAMS_BOT_APP_SECRET, COWORK_TEAMS_TENANT_ID, COWORK_TEAMS_ALLOWED_USERS not set" in caplog.text
    )


def test_an_install_without_teams_loads_the_other_channels_quietly(monkeypatch, caplog):
    monkeypatch.setattr(teams_plugin, "plugin", None)
    registry = PluginRegistry()
    with caplog.at_level(logging.WARNING):
        loaded = load_first_party_plugins(registry)
    assert "teams" not in loaded
    assert {"slack", "discord", "telegram", "whatsapp"} <= set(loaded)
    assert "teams" not in caplog.text


def test_the_routes_module_and_the_plugin_share_one_settings_class():
    assert teams_plugin.TeamsSettings is teams_routes.TeamsSettings
