from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from anton.core.dispatch import Attachment, InboundEvent, InboundMessage, PlatformAddress

from cowork.channels.plugin import (
    ChannelCapabilities,
    ChannelPlugin,
    CredentialField,
    CredentialSchema,
    WebhookRoute,
)
from cowork.channels.text import split_for_limit
from cowork.channels.webhooks import SignatureError, WebhookAck, WebhookHandshake

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter, ChannelSetup, OutboundMessage

log = logging.getLogger(__name__)

CHANNEL_TYPE = "discord"
DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_MAX_TEXT = 2000
DISCORD_MAX_FILE_BYTES = 20 * 1024 * 1024
DISCORD_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # default bot upload cap

# Gateway intents for receiving normal channel/DM messages. MESSAGE_CONTENT is
# privileged — it must be enabled in the Discord Developer Portal or message
# content arrives empty. GUILDS|GUILD_MESSAGES|DIRECT_MESSAGES|MESSAGE_CONTENT.
_GATEWAY_INTENTS = (1 << 0) | (1 << 9) | (1 << 12) | (1 << 15)
_GATEWAY_DISPATCH = 0
_GATEWAY_HEARTBEAT = 1
_GATEWAY_IDENTIFY = 2
_GATEWAY_RECONNECT = 7
_GATEWAY_INVALID_SESSION = 9
_GATEWAY_HELLO = 10


def extract_media(data: dict) -> list[Attachment]:
    """Attachment-option descriptors from a slash-command interaction; the CDN
    bytes are fetched later in the background via fetch_attachment."""
    media: list[Attachment] = []
    resolved = ((data.get("data") or {}).get("resolved") or {}).get("attachments") or {}
    for raw in resolved.values():
        if not isinstance(raw, dict) or not raw.get("url"):
            continue
        if (raw.get("size") or 0) > DISCORD_MAX_FILE_BYTES:
            log.info("skipping discord attachment over the size cap")
            continue
        attachment = Attachment(
            filename=raw.get("filename") or "file",
            mime_type=raw.get("content_type") or "application/octet-stream",
        )
        attachment.discord_url = raw.get("url")
        media.append(attachment)
    return media
_INTERACTION_PING = 1
_INTERACTION_APPLICATION_COMMAND = 2


def _verify_ed25519(public_key_hex: str, signature_hex: str, message: bytes) -> None:
    """Raise SignatureError unless ``signature`` is a valid Ed25519 signature of
    ``message`` for ``public_key``. Uses ``cryptography`` (a core dependency)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), message)
    except (InvalidSignature, ValueError) as exc:
        raise SignatureError("discord ed25519 verify failed") from exc


class DiscordBridge:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        self._secrets = dict(credentials)
        self._setup: ChannelSetup | None = None
        # Learned from the Gateway READY frame; used to skip our own messages
        # and detect @-mentions of the bot.
        self._bot_user_id: str | None = None

    @property
    def channel_type(self) -> str:
        return CHANNEL_TYPE

    async def setup(self, setup: ChannelSetup) -> None:
        self._setup = setup

    async def shutdown(self) -> None:
        self._setup = None

    async def deliver(self, message: OutboundMessage) -> None:
        for chunk in split_for_limit(message.text, DISCORD_MAX_TEXT):
            await self.send_text(address=message.address, text=chunk)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"**{getattr(card, 'prompt', '')}**\n{bullets}".strip()
        for chunk in split_for_limit(text, DISCORD_MAX_TEXT):
            await self.send_text(address=address, text=chunk)

    async def set_typing(self, *, address: PlatformAddress) -> None:
        # Best-effort: Discord shows the indicator for ~10s per trigger.
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{DISCORD_API_BASE}/channels/{address.platform_id}/typing",
                    headers={"Authorization": f"Bot {bot_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError):
            pass

    def try_handshake(
        self, *, method: str, body: bytes, headers: Mapping[str, str], query: Mapping[str, str]
    ) -> WebhookHandshake:
        # Discord PING (type 1) — must be answered with a signed PONG. Discord
        # requires a valid signature even on the PING, so verify first.
        if method != "POST" or not body:
            return WebhookHandshake(handled=False)
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return WebhookHandshake(handled=False)
        if not (isinstance(data, dict) and data.get("type") == _INTERACTION_PING):
            return WebhookHandshake(handled=False)
        try:
            self.verify_signature(body=body, headers=headers)
        except SignatureError:
            return WebhookHandshake(handled=True, response_body="invalid signature", status_code=401)
        return WebhookHandshake(
            handled=True,
            response_body=json.dumps({"type": _INTERACTION_PING}),
            content_type="application/json",
        )

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None:
        public_key = (self._secrets.get("public_key") or "").strip()
        signature = headers.get("x-signature-ed25519", "")
        timestamp = headers.get("x-signature-timestamp", "")
        if not public_key:
            raise SignatureError("discord public_key not configured")
        if not signature or not timestamp:
            raise SignatureError("missing discord signature headers")
        _verify_ed25519(public_key, signature, timestamp.encode("utf-8") + body)

    async def parse_inbound(
        self, *, body: bytes, headers: Mapping[str, str], route_name: str | None
    ) -> list[InboundEvent]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(data, dict) or data.get("type") != _INTERACTION_APPLICATION_COMMAND:
            return []
        channel_id = str(data.get("channel_id", "") or "")
        if not channel_id:
            return []
        user_obj = (data.get("member") or {}).get("user") or data.get("user") or {}
        interaction_id = str(data.get("id", "") or "")
        opts = (data.get("data") or {}).get("options") or []
        utterance = " ".join(
            str(o.get("value", "")).strip()
            for o in opts
            if isinstance(o, dict) and o.get("value")
        ).strip()
        if not utterance:
            utterance = "/" + str((data.get("data") or {}).get("name", "")).strip()
        event = InboundEvent(
            address=PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=channel_id, thread_id=None),
            message=InboundMessage(
                id=interaction_id,
                content=utterance,
                timestamp=datetime.now(timezone.utc),
                kind="chat",
                sender_id=str(user_obj.get("id", "")) or None,
                is_mention=True,
                is_group=bool(data.get("guild_id")),
                attachments=extract_media(data),
            ),
        )
        event._dedupe_key = f"discord:interaction:{interaction_id}"  # type: ignore[attr-defined]
        return [event]

    def dedupe_key(self, event: InboundEvent) -> str | None:
        key = getattr(event, "_dedupe_key", None)
        if key:
            return key
        mid = event.message.id
        return f"discord:{event.address.platform_id}:{mid}" if mid else None

    async def stream_events(self) -> AsyncIterator[list[InboundEvent]]:
        """Connect to the Discord Gateway and yield inbound message events. This
        is the primary ingress for normal channel/DM messages (the interactions
        webhook only carries slash commands and needs a public URL). One
        connection lifecycle per call — the ingress manager reconnects when it
        returns. No session resume; dedupe absorbs any replays on reconnect."""
        import aiohttp

        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            return
        async with aiohttp.ClientSession(headers={"Authorization": f"Bot {bot_token}"}) as session:
            async with session.get(f"{DISCORD_API_BASE}/gateway/bot") as resp:
                if resp.status != 200:
                    raise ConnectionError(f"discord gateway/bot HTTP {resp.status}")
                gateway_url = (await resp.json())["url"]
            async with session.ws_connect(f"{gateway_url}?v=10&encoding=json", heartbeat=None) as ws:
                hello = await ws.receive_json()
                interval = float(hello["d"]["heartbeat_interval"]) / 1000.0
                seq: dict[str, Any] = {"s": None}
                heartbeat = asyncio.create_task(self._heartbeat(ws, interval, seq))
                try:
                    await ws.send_json({
                        "op": _GATEWAY_IDENTIFY,
                        "d": {
                            "token": bot_token,
                            "intents": _GATEWAY_INTENTS,
                            "properties": {"os": "linux", "browser": "cowork", "device": "cowork"},
                        },
                    })
                    async for frame in ws:
                        if frame.type != aiohttp.WSMsgType.TEXT:
                            break
                        payload = frame.json()
                        if payload.get("s") is not None:
                            seq["s"] = payload["s"]
                        op = payload.get("op")
                        if op == _GATEWAY_DISPATCH:
                            t = payload.get("t")
                            if t == "READY":
                                self._bot_user_id = ((payload.get("d") or {}).get("user") or {}).get("id")
                            elif t == "MESSAGE_CREATE":
                                event = self._normalize_message(payload.get("d") or {})
                                if event is not None:
                                    yield [event]
                        elif op == _GATEWAY_HEARTBEAT:
                            await ws.send_json({"op": _GATEWAY_HEARTBEAT, "d": seq["s"]})
                        elif op in (_GATEWAY_RECONNECT, _GATEWAY_INVALID_SESSION):
                            break  # let the manager reconnect with a fresh IDENTIFY
                finally:
                    heartbeat.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat

    async def _heartbeat(self, ws: Any, interval: float, seq: dict[str, Any]) -> None:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json({"op": _GATEWAY_HEARTBEAT, "d": seq["s"]})

    def _normalize_message(self, d: dict) -> InboundEvent | None:
        """One Gateway MESSAGE_CREATE → InboundEvent, or None to skip (bot/self
        messages, empty non-attachment messages, missing channel)."""
        author = d.get("author") or {}
        if author.get("bot"):
            return None
        author_id = str(author.get("id", "") or "")
        if self._bot_user_id and author_id == str(self._bot_user_id):
            return None
        channel_id = str(d.get("channel_id", "") or "")
        if not channel_id:
            return None
        content = (d.get("content") or "").strip()
        attachments = self._message_attachments(d)
        if not content and not attachments:
            return None

        is_group = bool(d.get("guild_id"))
        mentions = d.get("mentions") or []
        is_mention = (not is_group) or (
            bool(self._bot_user_id)
            and any(str(m.get("id")) == str(self._bot_user_id) for m in mentions if isinstance(m, dict))
        )
        raw_ts = d.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)

        message_id = str(d.get("id", "") or "")
        event = InboundEvent(
            address=PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=channel_id, thread_id=None),
            message=InboundMessage(
                id=message_id,
                content=content,
                timestamp=timestamp,
                kind="chat",
                sender_id=author_id or None,
                sender_name=author.get("global_name") or author.get("username") or None,
                is_mention=is_mention,
                is_group=is_group,
                attachments=attachments,
            ),
        )
        event._dedupe_key = f"discord:message:{message_id}"  # type: ignore[attr-defined]
        return event

    @staticmethod
    def _message_attachments(d: dict) -> list[Attachment]:
        out: list[Attachment] = []
        for raw in (d.get("attachments") or []):
            if not isinstance(raw, dict) or not raw.get("url"):
                continue
            if (raw.get("size") or 0) > DISCORD_MAX_FILE_BYTES:
                log.info("skipping discord attachment over the size cap")
                continue
            attachment = Attachment(
                filename=raw.get("filename") or "file",
                mime_type=raw.get("content_type") or "application/octet-stream",
            )
            attachment.discord_url = raw.get("url")
            out.append(attachment)
        return out

    def ack_response(self, events: list[InboundEvent]) -> WebhookAck | None:
        if not events:
            return None
        return WebhookAck(
            body=json.dumps({"type": 5}),
            content_type="application/json",
        )

    async def send_attachment(self, *, address: PlatformAddress, path: str, filename: str | None = None) -> str:
        """Post one file to the channel; returns the platform message id."""
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("discord bot_token not configured")
        source = Path(path)
        if source.stat().st_size > DISCORD_MAX_UPLOAD_BYTES:
            raise RuntimeError("discord attachment exceeds the upload size cap")
        name = (filename or source.name).strip() or "file"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{DISCORD_API_BASE}/channels/{address.platform_id}/messages",
                    headers={"Authorization": f"Bot {bot_token}"},
                    files={"files[0]": (name, source.read_bytes())},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"discord upload transport error: {exc!r}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ConnectionError(f"discord transient HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise RuntimeError(f"discord upload failed: HTTP {resp.status_code}")
        return str((resp.json() or {}).get("id", ""))

    async def fetch_attachment(self, attachment: Attachment) -> bytes | None:
        """Best-effort CDN download (attachment URLs need no auth)."""
        url = getattr(attachment, "discord_url", None)
        if not url:
            return None
        return await self.download_url(url)

    @staticmethod
    async def download_url(url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        return resp.content if resp.status_code == 200 else None

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("discord bot_token not configured")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{DISCORD_API_BASE}/channels/{address.platform_id}/messages",
                    json={"content": text},
                    headers={"Authorization": f"Bot {bot_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"discord transport error: {exc!r}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ConnectionError(f"discord transient HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise RuntimeError(f"discord send failed: HTTP {resp.status_code}")
        return str((resp.json() or {}).get("id", ""))


async def _factory(credentials: Mapping[str, str]) -> ChannelAdapter | None:
    # Only the bot token is required: it powers the Gateway (inbound) and the
    # REST API (outbound). public_key is needed solely for the interactions
    # webhook (slash commands), so it stays optional.
    if not (credentials.get("bot_token") or "").strip():
        return None
    return DiscordBridge(credentials)


plugin = ChannelPlugin(
    channel_type=CHANNEL_TYPE,
    display_name="Discord",
    factory=_factory,
    credentials=CredentialSchema(
        fields=(
            CredentialField(name="bot_token", label="Bot token", secret=True, required=True,
                            description="Bot token — powers the Gateway (receive) and sending messages"),
            CredentialField(name="public_key", label="Public key", secret=False, required=False,
                            description="Application public key (hex) — only for the slash-command interactions webhook"),
        )
    ),
    webhooks=(WebhookRoute(path="/interactions", methods=("POST",), name="interactions", needs_raw_body=True),),
    capabilities=ChannelCapabilities(
        supports_webhook_ingress=True,
        supports_webhook_setup=False,
        supports_teardown=False,
        supports_oauth=False,
        supports_direct_credentials=True,
        supports_custom_ack=True,
    ),
)
