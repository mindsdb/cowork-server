from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
from anton.core.dispatch import InboundEvent, InboundMessage, PlatformAddress

from cowork.channels.plugin import (
    ChannelPlugin,
    CredentialField,
    CredentialSchema,
    WebhookRoute,
)
from cowork.channels.webhooks import SignatureError, WebhookHandshake

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter, ChannelSetup, OutboundMessage

log = logging.getLogger(__name__)

CHANNEL_TYPE = "telegram"
TELEGRAM_API_BASE = "https://api.telegram.org/bot"
TELEGRAM_MAX_TEXT = 4096
_SECRET_TOKEN_HEADER = "x-telegram-bot-api-secret-token"


def _split_for_limit(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` chars, preferring a
    newline boundary so messages don't break mid-line."""
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


class TelegramBridge:
    """Telegram adapter: WebhookBridge (ingress) + ChannelAdapter (egress)."""

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
        for chunk in _split_for_limit(message.text, TELEGRAM_MAX_TEXT):
            await self.send_text(address=message.address, text=chunk)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"*{getattr(card, 'prompt', '')}*\n{bullets}".strip()
        for chunk in _split_for_limit(text, TELEGRAM_MAX_TEXT):
            await self.send_text(address=address, text=chunk)

    def try_handshake(
        self, *, method: str, body: bytes, headers: Mapping[str, str], query: Mapping[str, str]
    ) -> WebhookHandshake:
        return WebhookHandshake(handled=False)

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None:
        expected = (self._secrets.get("secret_token") or "").strip()
        if not expected:
            raise SignatureError(
                "telegram secret_token not configured; webhook ingress refuses "
                "unauthenticated payloads"
            )
        provided = headers.get(_SECRET_TOKEN_HEADER, "")
        if not hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8")):
            raise SignatureError("telegram secret_token mismatch")

    async def parse_inbound(
        self, *, body: bytes, headers: Mapping[str, str], route_name: str | None
    ) -> list[InboundEvent]:
        try:
            update = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        event = self._normalize_update(update)
        return [event] if event is not None else []

    def dedupe_key(self, event: InboundEvent) -> str | None:
        key = getattr(event, "_dedupe_key", None)
        if key:
            return key
        message_id = event.message.id
        if not message_id:
            return None
        return f"telegram:message:{event.address.platform_id}:{message_id}"

    def _normalize_update(self, update: dict) -> InboundEvent | None:
        """One Telegram Update → InboundEvent, or None to skip.

        Skips non-message updates (edited_message, polls, …), bot messages
        (avoids echo loops), and non-text messages (media is future work)."""
        msg = update.get("message")
        if not isinstance(msg, dict):
            return None
        sender = msg.get("from") or {}
        if sender.get("is_bot"):
            return None
        text = (msg.get("text") or "").strip()
        if not text:
            return None

        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        chat_type = chat.get("type", "private")
        is_group = chat_type in ("group", "supergroup", "channel")

        date = msg.get("date")
        try:
            timestamp = (
                datetime.fromtimestamp(float(date), timezone.utc)
                if date
                else datetime.now(timezone.utc)
            )
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)

        bot_username = self._secrets.get("bot_username", "")
        is_mention = (not is_group) or bool(bot_username and f"@{bot_username}" in text)

        message_id = str(msg.get("message_id", ""))
        event = InboundEvent(
            address=PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=chat_id, thread_id=None),
            message=InboundMessage(
                id=message_id,
                content=text,
                timestamp=timestamp,
                kind="chat",
                sender_id=str(sender.get("id", "")) or None,
                is_mention=is_mention,
                is_group=is_group,
            ),
        )

        update_id = update.get("update_id")
        event._dedupe_key = (
            f"telegram:update:{update_id}"
            if update_id is not None
            else f"telegram:message:{chat_id}:{message_id}"
        )
        return event

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        """Send one chunk via ``sendMessage``; returns the platform message id.

        Retries once without Markdown on a 400 (unbalanced markup is common),
        and maps 420/429/5xx to ConnectionError so the runtime can retry."""
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("telegram bot_token not configured")

        payload: dict[str, Any] = {
            "chat_id": address.platform_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        result = await self._call(bot_token, "sendMessage", payload)
        if not result.get("ok") and result.get("error_code") == 400:
            payload.pop("parse_mode", None)
            result = await self._call(bot_token, "sendMessage", payload)
        if not result.get("ok"):
            code = result.get("error_code")
            if code in (420, 429) or (isinstance(code, int) and 500 <= code < 600):
                raise ConnectionError(f"telegram transient error code={code}")
            raise RuntimeError(
                f"telegram sendMessage failed: {result.get('description', 'unknown')}"
            )
        return str((result.get("result") or {}).get("message_id", ""))

    @staticmethod
    async def _call(bot_token: str, method: str, payload: dict) -> dict:
        """Await one Telegram Bot API call; returns the parsed JSON body."""
        url = f"{TELEGRAM_API_BASE}{bot_token}/{method}"
        try:
            async with httpx.AsyncClient(
                timeout=15.0, headers={"User-Agent": "Cowork/1.0"}
            ) as client:
                resp = await client.post(url, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"telegram {method} transport error: {exc!r}") from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(f"telegram {method} HTTP {resp.status_code}") from exc


async def _factory(credentials: Mapping[str, str]) -> ChannelAdapter | None:
    """Build a TelegramBridge from resolved credentials, or None if the channel
    is not fully configured.
"""
    if not (credentials.get("bot_token") or "").strip():
        return None
    if not (credentials.get("secret_token") or "").strip():
        return None
    return TelegramBridge(credentials)


plugin = ChannelPlugin(
    channel_type=CHANNEL_TYPE,
    display_name="Telegram",
    factory=_factory,
    credentials=CredentialSchema(
        fields=(
            CredentialField(
                name="bot_token",
                label="Bot token",
                secret=True,
                required=True,
                description="Bot API token from @BotFather",
            ),
            CredentialField(
                name="secret_token",
                label="Webhook secret token",
                secret=True,
                required=True,
                description=(
                    "Passed to setWebhook and echoed in the "
                    "X-Telegram-Bot-Api-Secret-Token header; required to authenticate webhook ingress"
                ),
            ),
            CredentialField(
                name="bot_username",
                label="Bot username",
                secret=False,
                required=False,
                description="Used to detect @mentions in group chats",
            ),
        )
    ),
    webhooks=(WebhookRoute(path="/webhook", methods=("POST",), needs_raw_body=True),),
)
