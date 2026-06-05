from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
from anton.core.dispatch import Attachment, InboundEvent, InboundMessage, PlatformAddress

from cowork.channels.plugin import (
    ChannelCapabilities,
    ChannelPlugin,
    CredentialField,
    CredentialSchema,
    OAuthSpec,
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
_OAUTH_SCOPES = ("bot", "applications.commands")


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

    def ack_response(self, events: list[InboundEvent]) -> WebhookAck | None:
        if not events:
            return None
        return WebhookAck(
            body=json.dumps({"type": 5}),
            content_type="application/json",
        )

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
    if not (credentials.get("public_key") or "").strip():
        return None
    if not (credentials.get("bot_token") or "").strip():
        return None
    return DiscordBridge(credentials)


plugin = ChannelPlugin(
    channel_type=CHANNEL_TYPE,
    display_name="Discord",
    factory=_factory,
    credentials=CredentialSchema(
        fields=(
            CredentialField(name="public_key", label="Public key", secret=False, required=True,
                            description="Application public key (hex); verifies interaction signatures"),
            CredentialField(name="bot_token", label="Bot token", secret=True, required=True,
                            description="Bot token used to post messages"),
            CredentialField(name="application_id", label="Application id", secret=False, required=False,
                            description="App id (OAuth install)"),
            CredentialField(name="client_secret", label="OAuth client secret", secret=True, required=False,
                            description="App client secret (OAuth install)"),
        )
    ),
    webhooks=(WebhookRoute(path="/interactions", methods=("POST",), name="interactions", needs_raw_body=True),),
    oauth=OAuthSpec(scopes=_OAUTH_SCOPES),
    capabilities=ChannelCapabilities(
        supports_webhook_ingress=True,
        supports_webhook_setup=False,
        supports_teardown=False,
        supports_oauth=True,
        supports_direct_credentials=True,
        supports_custom_ack=True,
    ),
)
