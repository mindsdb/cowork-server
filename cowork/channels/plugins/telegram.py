from __future__ import annotations

import hmac
import json
import logging
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from anton.core.dispatch import ActionResponse, Attachment, InboundEvent, InboundMessage, PlatformAddress

from cowork.channels.lifecycle import (
    ChannelLifecycle,
    LifecycleContext,
    LifecycleError,
    LifecycleResult,
)
from cowork.channels.plugin import (
    ChannelCapabilities,
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
TELEGRAM_FILE_BASE = "https://api.telegram.org/file/bot"
TELEGRAM_MAX_TEXT = 4096
TELEGRAM_MAX_FILE_BYTES = 20 * 1024 * 1024  # Bot API getFile hard limit
TELEGRAM_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # Bot API sendDocument hard limit
_SECRET_TOKEN_HEADER = "x-telegram-bot-api-secret-token"


def extract_media(msg: dict) -> list[Attachment]:
    """Photo/document → Attachment descriptors; the bytes are fetched later in
    the background via fetch_attachment (never in the pre-ACK parse path)."""
    media: list[Attachment] = []
    if isinstance(msg.get("photo"), list) and msg["photo"]:
        size = msg["photo"][-1]  # largest rendition is last
        attachment = Attachment(filename=f"photo_{size.get('file_unique_id', 'tg')}.jpg", mime_type="image/jpeg")
        attachment.telegram_file_id = size.get("file_id")
        attachment.telegram_file_size = size.get("file_size")
        media.append(attachment)
    document = msg.get("document")
    if isinstance(document, dict):
        attachment = Attachment(
            filename=document.get("file_name") or f"document_{document.get('file_unique_id', 'tg')}",
            mime_type=document.get("mime_type") or "application/octet-stream",
        )
        attachment.telegram_file_id = document.get("file_id")
        attachment.telegram_file_size = document.get("file_size")
        media.append(attachment)
    kept = []
    for attachment in media:
        size = getattr(attachment, "telegram_file_size", None)
        if size and size > TELEGRAM_MAX_FILE_BYTES:
            log.info("skipping telegram attachment over getFile limit")
            continue
        if getattr(attachment, "telegram_file_id", None):
            kept.append(attachment)
    return kept


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
        question_id = getattr(card, "question_id", None)
        options = list(getattr(card, "options", []) or [])
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if question_id and options and bot_token:
            keyboard = [[
                {"text": option.label, "callback_data": f"appr:{question_id}:{option.id}"}
                for option in options
            ]]
            result = await self._call(bot_token, "sendMessage", {
                "chat_id": address.platform_id,
                "text": getattr(card, "prompt", "") or "Approve?",
                "reply_markup": {"inline_keyboard": keyboard},
            })
            if result.get("ok"):
                return
            log.warning("telegram inline card failed; falling back to text")
        bullets = "\n".join(f"  • {o.label}" for o in options)
        text = f"*{getattr(card, 'prompt', '')}*\n{bullets}".strip()
        for chunk in _split_for_limit(text, TELEGRAM_MAX_TEXT):
            await self.send_text(address=address, text=chunk)

    def parse_action_response(self, *, body: bytes, headers: Mapping[str, str]) -> ActionResponse | None:
        """Inline-button presses arrive as callback_query updates; anything else
        falls through to the normal inbound path."""
        try:
            update = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        query = update.get("callback_query") if isinstance(update, dict) else None
        if not isinstance(query, dict):
            return None
        data = query.get("data") or ""
        parts = data.split(":", 2)
        if len(parts) != 3 or parts[0] != "appr":
            return None
        response = ActionResponse(
            question_id=parts[1],
            selected_option_id=parts[2],
            user_id=str((query.get("from") or {}).get("id", "")) or None,
        )
        message = query.get("message") or {}
        response.telegram_callback_query_id = query.get("id")
        response.telegram_chat_id = str((message.get("chat") or {}).get("id", ""))
        response.telegram_message_id = message.get("message_id")
        return response

    async def ack_action_response(self, response: Any, *, resolved: bool, approved: bool) -> None:
        """Clear the button spinner and rewrite the card with the outcome."""
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            return
        note = ("Approved" if approved else "Denied") if resolved else "No longer pending"
        query_id = getattr(response, "telegram_callback_query_id", None)
        if query_id:
            try:
                await self._call(bot_token, "answerCallbackQuery", {"callback_query_id": query_id, "text": note})
            except Exception:
                log.debug("answerCallbackQuery failed", exc_info=True)
        chat_id = getattr(response, "telegram_chat_id", None)
        message_id = getattr(response, "telegram_message_id", None)
        if chat_id and message_id:
            try:
                await self._call(bot_token, "editMessageText", {
                    "chat_id": chat_id, "message_id": message_id, "text": note,
                })
            except Exception:
                log.debug("editMessageText failed", exc_info=True)

    async def set_typing(self, *, address: PlatformAddress) -> None:
        # Best-effort: Telegram shows the indicator for ~5s per call.
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            return
        await self._call(bot_token, "sendChatAction", {"chat_id": address.platform_id, "action": "typing"})

    async def fetch_attachment(self, attachment: Attachment) -> bytes | None:
        """Resolve a parsed attachment to bytes (getFile + download). Best-effort:
        any failure returns None so the turn proceeds with the text alone."""
        file_id = getattr(attachment, "telegram_file_id", None)
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not file_id or not bot_token:
            return None
        try:
            info = await self._call(bot_token, "getFile", {"file_id": file_id})
        except (ConnectionError, RuntimeError):
            return None
        file_path = (info.get("result") or {}).get("file_path") if info.get("ok") else None
        if not file_path:
            return None
        return await self.download_file(bot_token, file_path)

    async def send_attachment(self, *, address: PlatformAddress, path: str, filename: str | None = None) -> str:
        """Upload one file via sendDocument; returns the platform message id."""
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("telegram bot_token not configured")
        source = Path(path)
        if source.stat().st_size > TELEGRAM_MAX_UPLOAD_BYTES:
            raise RuntimeError("telegram attachment exceeds the 50MB upload limit")
        name = (filename or source.name).strip() or "file"
        try:
            async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": "Cowork/1.0"}) as client:
                resp = await client.post(
                    f"{TELEGRAM_API_BASE}{bot_token}/sendDocument",
                    data={"chat_id": address.platform_id},
                    files={"document": (name, source.read_bytes())},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"telegram sendDocument transport error: {exc!r}") from exc
        result = resp.json()
        if not result.get("ok"):
            raise RuntimeError(f"telegram sendDocument failed: {result.get('description', 'unknown')}")
        return str((result.get("result") or {}).get("message_id", ""))

    @staticmethod
    async def download_file(bot_token: str, file_path: str) -> bytes | None:
        # The URL embeds the bot token — never log it.
        url = f"{TELEGRAM_FILE_BASE}{bot_token}/{file_path}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        return resp.content if resp.status_code == 200 else None

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
        text = (msg.get("text") or msg.get("caption") or "").strip()
        attachments = extract_media(msg)
        if not text and not attachments:
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
                attachments=attachments,
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


async def _setup(ctx: LifecycleContext) -> LifecycleResult:
    """Register the Telegram webhook: mint a secret_token if absent, call
    setWebhook with it, then bring the live adapter online."""
    bot_token = (ctx.credentials.get("bot_token") or "").strip()
    if not bot_token:
        raise LifecycleError(400, "telegram bot_token is required before setup")
    if not ctx.webhook_url:
        raise LifecycleError(409, "public base URL is not configured; cannot register a webhook")

    secret_token = (ctx.credentials.get("secret_token") or "").strip()
    if not secret_token:
        secret_token = secrets.token_urlsafe(32)
        ctx.persist_credentials({"secret_token": secret_token})

    result = await TelegramBridge._call(
        bot_token,
        "setWebhook",
        {
            "url": ctx.webhook_url,
            "secret_token": secret_token,
            "allowed_updates": ["message", "callback_query"],
        },
    )
    if not result.get("ok"):
        raise LifecycleError(502, f"telegram setWebhook failed: {result.get('description', 'unknown')}")

    active = await ctx.refresh_adapter()
    return LifecycleResult(active=active, detail="telegram webhook registered")


async def _teardown(ctx: LifecycleContext) -> LifecycleResult:
    """Unregister the Telegram webhook and drop the live adapter. Credentials
    are left intact — teardown stops ingress, it does not forget the channel."""
    bot_token = (ctx.credentials.get("bot_token") or "").strip()
    if bot_token:
        try:
            await TelegramBridge._call(bot_token, "deleteWebhook", {"drop_pending_updates": False})
        except Exception:
            log.warning("telegram deleteWebhook failed during teardown")
    await ctx.remove_adapter()
    return LifecycleResult(active=False, detail="telegram webhook removed")


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
    lifecycle=ChannelLifecycle(setup=_setup, teardown=_teardown),
    capabilities=ChannelCapabilities(
        supports_webhook_ingress=True,
        supports_webhook_setup=True,
        supports_teardown=True,
        supports_oauth=False,
        supports_direct_credentials=True,
        supports_custom_ack=False,
    ),
)
