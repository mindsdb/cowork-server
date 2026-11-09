"""Teams as a channel: Microsoft posts each @mention straight to Cowork.

This is the path for a developer's tunnel. On a self-hosted Cowork, Teams
questions come through the recording proxy instead (cowork/channels/teams.py),
and this webhook refuses every message: a channel turn runs with no person,
and a deployment that trusts identity headers keeps each person's connections
apart only for turns that name one.

Every message must carry Microsoft's token, which ``verify_signature`` checks
before anything is stored or run: signed with one of Microsoft's keys
(login.botframework.com) endorsed for the message's channel, issued by the Bot
Framework, for this bot's App ID, in date, and naming the message's
serviceUrl. Then the tenant: Teams can send the bot messages from any
organization, so every tenant ID the message carries must be the configured
one, and with none configured every message is refused.

The bot is set in the environment (``COWORK_TEAMS_*``, see
cowork/channels/teams.py), not in Settings → Channels, so the channel config
routes refuse Teams. The plugin is registered only when the Teams packages are
installed and ``COWORK_TEAMS_BOT_APP_ID`` is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import jwt
from anton.core.dispatch import InboundEvent, InboundMessage, PlatformAddress

from cowork.channels.plugin import ChannelCapabilities, ChannelPlugin, CredentialSchema, WebhookRoute
from cowork.channels.teams import TeamsBot, TeamsSettings
from cowork.channels.teams_client import teams_available
from cowork.channels.webhooks import SignatureError
from cowork.common.settings.app_settings import get_app_settings

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter, ChannelSetup, OutboundMessage

log = logging.getLogger(__name__)

CHANNEL_TYPE = "teams"
OPENID_CONFIG_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"
TOKEN_ISSUER = "https://api.botframework.com"
CLOCK_SKEW_S = 5 * 60
KEYS_MAX_AGE_S = 24 * 60 * 60
# Reply addresses kept for the turns in flight; the oldest go first.
MAX_REPLY_ADDRESSES = 1000

_MESSAGE_ID = ";messageid="


class SigningKeys:
    """Microsoft's signing keys, fetched off the request path and refreshed daily.

    ``verify_signature`` is synchronous, so it only reads what's cached. A key
    it doesn't know starts a refresh, and that message is refused; Microsoft
    retries it.
    """

    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self._http = http
        self._keys: dict[str, tuple[jwt.PyJWK, frozenset[str]]] = {}
        self._fetched_at = 0.0
        self._refreshing: asyncio.Task[None] | None = None

    def load(self, jwks: Mapping[str, Any]) -> None:
        keys = {}
        for entry in jwks.get("keys", []):
            try:
                key = jwt.PyJWK.from_dict(entry)
            except jwt.PyJWTError:
                continue
            keys[entry.get("kid")] = (key, frozenset(entry.get("endorsements") or ()))
        self._keys = keys
        self._fetched_at = time.time()

    async def refresh(self) -> None:
        http = self._http or httpx.AsyncClient(timeout=10)
        try:
            config = (await http.get(OPENID_CONFIG_URL)).raise_for_status().json()
            self.load((await http.get(config["jwks_uri"])).raise_for_status().json())
            log.info("Teams: loaded %d Microsoft signing keys", len(self._keys))
        except Exception as error:
            log.warning("Teams: couldn't load Microsoft's signing keys: %s", type(error).__name__)
        finally:
            if self._http is None:
                await http.aclose()

    def _refresh_soon(self) -> None:
        if self._refreshing is None or self._refreshing.done():
            try:
                self._refreshing = asyncio.get_running_loop().create_task(self.refresh())
            except RuntimeError:
                pass

    def get(self, kid: str) -> tuple[jwt.PyJWK, frozenset[str]] | None:
        if time.time() - self._fetched_at > KEYS_MAX_AGE_S:
            self._refresh_soon()
        found = self._keys.get(kid)
        if found is None:
            self._refresh_soon()
        return found


def _mention_free_text(activity: Mapping[str, Any]) -> str:
    """The message without the bot's own @mention tags."""
    text = activity.get("text") or ""
    bot_id = (activity.get("recipient") or {}).get("id")
    for entity in activity.get("entities") or []:
        if entity.get("type") == "mention" and (entity.get("mentioned") or {}).get("id") == bot_id:
            text = text.replace(entity.get("text") or "", "")
    return re.sub(r"\s+", " ", text).strip()


def _mentions_bot(activity: Mapping[str, Any]) -> bool:
    bot_id = (activity.get("recipient") or {}).get("id")
    return any(
        entity.get("type") == "mention" and (entity.get("mentioned") or {}).get("id") == bot_id
        for entity in activity.get("entities") or []
    )


def _address(activity: Mapping[str, Any]) -> PlatformAddress:
    # A channel thread's conversation id is "<channel>;messageid=<thread root>".
    conversation_id = (activity.get("conversation") or {}).get("id") or ""
    channel, _, root = conversation_id.partition(_MESSAGE_ID)
    return PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=channel, thread_id=root or None)


class TeamsBridge:
    def __init__(self, settings: TeamsSettings, keys: SigningKeys, bot: TeamsBot) -> None:
        self._app_id = settings.bot_app_id
        self._tenant_id = settings.tenant_id
        self._keys = keys
        self._bot = bot
        self._reply_to: OrderedDict[PlatformAddress, tuple[str, str, str]] = OrderedDict()
        self.refused: Counter[str] = Counter()

    @property
    def channel_type(self) -> str:
        return CHANNEL_TYPE

    async def setup(self, setup: ChannelSetup) -> None:
        return None

    async def shutdown(self) -> None:
        await self._bot.aclose()

    def _refuse(self, reason: str) -> SignatureError:
        self.refused[reason] += 1
        return SignatureError(reason)

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None:
        if get_app_settings().trust_identity_headers:
            raise self._refuse("identity headers are trusted, so the Teams webhook is closed")
        if not self._tenant_id:
            raise self._refuse("COWORK_TEAMS_TENANT_ID is not set")
        try:
            activity = json.loads(body)
        except ValueError:
            raise self._refuse("body is not JSON") from None
        if not isinstance(activity, dict):
            raise self._refuse("body is not a Teams activity")

        scheme, _, token = (headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise self._refuse("no Microsoft token")
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError:
            raise self._refuse("malformed Microsoft token") from None
        found = self._keys.get(kid) if kid else None
        if found is None:
            raise self._refuse("token signed with an unknown key")
        key, endorsements = found
        if activity.get("channelId") not in endorsements:
            raise self._refuse("signing key not endorsed for this channel")
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self._app_id,
                issuer=TOKEN_ISSUER,
                leeway=CLOCK_SKEW_S,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as error:
            raise self._refuse(f"invalid Microsoft token: {type(error).__name__}") from None
        if claims.get("serviceurl") != activity.get("serviceUrl"):
            raise self._refuse("serviceUrl doesn't match the token")

        tenants = [
            (activity.get("conversation") or {}).get("tenantId"),
            ((activity.get("channelData") or {}).get("tenant") or {}).get("id"),
            headers.get("x-ms-tenant-id"),
        ]
        named = [tenant for tenant in tenants if tenant]
        if not named or any(tenant != self._tenant_id for tenant in named):
            raise self._refuse("message from another tenant")

    async def parse_inbound(self, *, body: bytes, headers: Mapping[str, str], route_name: str | None) -> list[InboundEvent]:
        activity = json.loads(body)
        if activity.get("type") != "message":
            return []
        address = _address(activity)
        sender = activity.get("from") or {}
        conversation = activity.get("conversation") or {}
        self._remember_reply_to(address, (activity["serviceUrl"], conversation["id"], activity["id"]))
        return [
            InboundEvent(
                address=address,
                message=InboundMessage(
                    id=activity["id"],
                    content=_mention_free_text(activity),
                    timestamp=datetime.now(timezone.utc),
                    sender_id=sender.get("aadObjectId") or sender.get("id"),
                    sender_name=sender.get("name"),
                    is_mention=_mentions_bot(activity),
                    is_group=conversation.get("conversationType") in ("channel", "groupChat"),
                ),
            )
        ]

    def _remember_reply_to(self, address: PlatformAddress, reference: tuple[str, str, str]) -> None:
        self._reply_to[address] = reference
        self._reply_to.move_to_end(address)
        while len(self._reply_to) > MAX_REPLY_ADDRESSES:
            self._reply_to.popitem(last=False)

    def dedupe_key(self, event: InboundEvent) -> str | None:
        return event.message.id

    async def deliver(self, message: OutboundMessage) -> None:
        reference = self._reply_to.get(message.address)
        if reference is None:
            log.warning("Teams: no message to reply to in %s; reply not sent", message.address.platform_id)
            return
        service_url, conversation_id, activity_id = reference
        self._bot.check_service_url(service_url)
        client = await self._bot.client()
        await client.reply(service_url, conversation_id, activity_id, message.text)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        log.info("Teams: action cards aren't supported; not shown")


async def _factory(credentials: Mapping[str, str]) -> ChannelAdapter | None:
    # The bot comes from the environment; Teams stores nothing in Settings.
    settings = TeamsSettings()
    keys = SigningKeys()
    await keys.refresh()
    return TeamsBridge(settings, keys, TeamsBot(settings))


def _build() -> ChannelPlugin | None:
    if not teams_available() or not TeamsSettings().bot_app_id:
        return None
    return ChannelPlugin(
        channel_type=CHANNEL_TYPE,
        display_name="Microsoft Teams",
        factory=_factory,
        credentials=CredentialSchema(),
        webhooks=(WebhookRoute(path="/api/messages", methods=("POST",), name="messages"),),
        capabilities=ChannelCapabilities(supports_webhook_ingress=True, supports_direct_credentials=False),
        credentials_from_env=True,
    )


plugin = _build()
