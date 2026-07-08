from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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
from cowork.channels.webhooks import SignatureError, WebhookHandshake

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter, ChannelSetup, OutboundMessage

log = logging.getLogger(__name__)

CHANNEL_TYPE = "whatsapp"
GRAPH_API_BASE = "https://graph.facebook.com/v21.0"
WHATSAPP_MAX_TEXT = 4096
WHATSAPP_MAX_FILE_BYTES = 20 * 1024 * 1024
WHATSAPP_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # Graph document cap
_CUSTOMER_CARE_WINDOW = timedelta(hours=24)
_TRANSIENT_ERROR_CODES = {4, 17, 32, 80007}


def extract_media(msg: dict) -> tuple[str, list[Attachment]]:
    """(caption, attachments) for image/document messages; bytes are resolved
    later in the background via fetch_attachment (media id → url → download)."""
    mtype = msg.get("type")
    if mtype not in ("image", "document"):
        return "", []
    media = msg.get(mtype) or {}
    media_id = media.get("id")
    if not media_id:
        return (media.get("caption") or "").strip(), []
    mime = media.get("mime_type") or "application/octet-stream"
    if mtype == "image":
        subtype = mime.split("/")[-1].split(";")[0] or "bin"
        filename = f"image_{media_id}.{subtype}"
    else:
        filename = media.get("filename") or f"document_{media_id}"
    attachment = Attachment(filename=filename, mime_type=mime)
    attachment.whatsapp_media_id = media_id
    return (media.get("caption") or "").strip(), [attachment]


class WhatsAppBridge:
    def __init__(self, credentials: Mapping[str, str]) -> None:
        self._secrets = dict(credentials)
        self._setup: ChannelSetup | None = None
        self._last_inbound: dict[str, datetime] = {}

    @property
    def channel_type(self) -> str:
        return CHANNEL_TYPE

    async def setup(self, setup: ChannelSetup) -> None:
        self._setup = setup

    async def shutdown(self) -> None:
        self._setup = None

    async def deliver(self, message: OutboundMessage) -> None:
        for chunk in split_for_limit(message.text, WHATSAPP_MAX_TEXT):
            await self.send_text(address=message.address, text=chunk)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"*{getattr(card, 'prompt', '')}*\n{bullets}".strip()
        for chunk in split_for_limit(text, WHATSAPP_MAX_TEXT):
            await self.send_text(address=address, text=chunk)

    def try_handshake(
        self, *, method: str, body: bytes, headers: Mapping[str, str], query: Mapping[str, str]
    ) -> WebhookHandshake:
        if method != "GET" or query.get("hub.mode") != "subscribe":
            return WebhookHandshake(handled=False)
        expected = (self._secrets.get("verify_token") or "").strip()
        provided = query.get("hub.verify_token", "")
        if not expected or provided != expected:
            return WebhookHandshake(handled=True, response_body="forbidden", status_code=403)
        return WebhookHandshake(handled=True, response_body=query.get("hub.challenge", ""))

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None:
        app_secret = (self._secrets.get("app_secret") or "").strip()
        if not app_secret:
            raise SignatureError("whatsapp app_secret not configured")
        header = headers.get("x-hub-signature-256", "")
        if not header.startswith("sha256="):
            raise SignatureError("missing/invalid whatsapp signature header")
        expected = header[len("sha256="):]
        digest = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, expected):
            raise SignatureError("whatsapp signature mismatch")

    async def parse_inbound(
        self, *, body: bytes, headers: Mapping[str, str], route_name: str | None
    ) -> list[InboundEvent]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        events: list[InboundEvent] = []
        for entry in (payload.get("entry") or []):
            for change in (entry.get("changes") or []):
                value = change.get("value") or {}
                for raw in (value.get("messages") or []):
                    event = self._normalize_message(raw)
                    if event is not None:
                        events.append(event)
        return events

    def dedupe_key(self, event: InboundEvent) -> str | None:
        key = getattr(event, "_dedupe_key", None)
        if key:
            return key
        mid = event.message.id
        return f"whatsapp:{mid}" if mid else None

    def _normalize_message(self, msg: dict) -> InboundEvent | None:
        attachments: list[Attachment] = []
        if msg.get("type") == "text":
            text = ((msg.get("text") or {}).get("body") or "").strip()
        else:
            text, attachments = extract_media(msg)
        sender = (msg.get("from") or "").strip()  # E.164, no '+'
        message_id = str(msg.get("id", ""))
        if (not text and not attachments) or not sender or not message_id:
            return None
        try:
            timestamp = datetime.fromtimestamp(int(msg.get("timestamp", "")), UTC)
        except (TypeError, ValueError):
            timestamp = datetime.now(UTC)
        self._last_inbound[sender] = timestamp
        event = InboundEvent(
            address=PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=sender, thread_id=None),
            message=InboundMessage(
                id=message_id,
                content=text,
                timestamp=timestamp,
                kind="chat",
                sender_id=sender,
                is_mention=True,
                is_group=False,
                attachments=attachments,
            ),
        )
        event._dedupe_key = f"whatsapp:{message_id}"  # type: ignore[attr-defined]
        return event

    async def send_attachment(self, *, address: PlatformAddress, path: str, filename: str | None = None) -> str:
        """Upload to the Graph media endpoint, then send as a document message."""
        phone_number_id = (self._secrets.get("phone_number_id") or "").strip()
        access_token = (self._secrets.get("access_token") or "").strip()
        if not phone_number_id or not access_token:
            raise RuntimeError("whatsapp phone_number_id/access_token not configured")
        recipient = address.platform_id
        self.require_care_window(recipient)
        source = Path(path)
        if source.stat().st_size > WHATSAPP_MAX_UPLOAD_BYTES:
            raise RuntimeError("whatsapp attachment exceeds the upload size cap")
        name = (filename or source.name).strip() or "file"
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"

        media_id = await self.upload_media(access_token, phone_number_id, name, mime, source.read_bytes())
        if not media_id:
            raise ConnectionError("whatsapp media upload failed")
        result = await self.send_media_message(access_token, phone_number_id, recipient, media_id, name)
        if result.get("error"):
            err = result["error"]
            code = err.get("code") if isinstance(err, dict) else None
            if code in _TRANSIENT_ERROR_CODES:
                raise ConnectionError(f"whatsapp rate-limit: code={code}")
            raise RuntimeError("whatsapp media send failed")
        messages = result.get("messages") or []
        return str(messages[0].get("id", "")) if messages else ""

    @staticmethod
    async def upload_media(access_token: str, phone_number_id: str, name: str, mime: str, data: bytes) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{GRAPH_API_BASE}/{phone_number_id}/media",
                    headers={"Authorization": f"Bearer {access_token}"},
                    data={"messaging_product": "whatsapp"},
                    files={"file": (name, data, mime)},
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        if resp.status_code != 200:
            return None
        return (resp.json() or {}).get("id")

    @staticmethod
    async def send_media_message(access_token: str, phone_number_id: str, recipient: str,
                                 media_id: str, name: str) -> dict:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "document",
            "document": {"id": media_id, "filename": name},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GRAPH_API_BASE}/{phone_number_id}/messages",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json=payload,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"whatsapp media send transport error: {exc!r}") from exc
        return resp.json()

    async def fetch_attachment(self, attachment: Attachment) -> bytes | None:
        """Resolve a media id to bytes (Graph media lookup + authed download).
        Best-effort: any failure returns None and the turn proceeds text-only."""
        media_id = getattr(attachment, "whatsapp_media_id", None)
        access_token = (self._secrets.get("access_token") or "").strip()
        if not media_id or not access_token:
            return None
        info = await self.media_info(access_token, media_id)
        if not info or not info.get("url"):
            return None
        if (info.get("file_size") or 0) > WHATSAPP_MAX_FILE_BYTES:
            log.info("skipping whatsapp attachment over the size cap")
            return None
        return await self.download_url(access_token, info["url"])

    @staticmethod
    async def media_info(access_token: str, media_id: str) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{GRAPH_API_BASE}/{media_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        return resp.json() if resp.status_code == 200 else None

    @staticmethod
    async def download_url(access_token: str, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {access_token}"})
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        return resp.content if resp.status_code == 200 else None

    def require_care_window(self, recipient: str) -> None:
        last = self._last_inbound.get(recipient)
        if last is None or (datetime.now(UTC) - last) > _CUSTOMER_CARE_WINDOW:
            raise RuntimeError(
                "whatsapp customer-care window expired for this contact; "
                "only pre-approved templates can be sent"
            )

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        phone_number_id = (self._secrets.get("phone_number_id") or "").strip()
        access_token = (self._secrets.get("access_token") or "").strip()
        if not phone_number_id or not access_token:
            raise RuntimeError("whatsapp phone_number_id/access_token not configured")
        recipient = address.platform_id
        self.require_care_window(recipient)
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"body": text, "preview_url": False},
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{GRAPH_API_BASE}/{phone_number_id}/messages",
                    json=payload,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"whatsapp transport error: {exc!r}") from exc
        result = resp.json()
        if result.get("error"):
            err = result["error"]
            code = err.get("code") if isinstance(err, dict) else None
            if code in _TRANSIENT_ERROR_CODES:
                raise ConnectionError(f"whatsapp rate-limit: code={code}")
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise RuntimeError(f"whatsapp send failed: {msg}")
        msgs = result.get("messages") or []
        if not msgs:
            raise RuntimeError("whatsapp send returned no message id")
        return str(msgs[0].get("id", ""))


async def _factory(credentials: Mapping[str, str]) -> ChannelAdapter | None:
    required = ("phone_number_id", "access_token", "app_secret", "verify_token")
    if any(not (credentials.get(f) or "").strip() for f in required):
        return None
    return WhatsAppBridge(credentials)


plugin = ChannelPlugin(
    channel_type=CHANNEL_TYPE,
    display_name="WhatsApp",
    factory=_factory,
    credentials=CredentialSchema(
        fields=(
            CredentialField(name="phone_number_id", label="Phone number id", secret=False, required=True,
                            description="Meta-issued id of the sending number"),
            CredentialField(name="access_token", label="Access token", secret=True, required=True,
                            description="Graph API token used to send messages"),
            CredentialField(name="app_secret", label="App secret", secret=True, required=True,
                            description="Verifies inbound X-Hub-Signature-256"),
            CredentialField(name="verify_token", label="Verify token", secret=False, required=True,
                            description="Operator-chosen token echoed during webhook subscription; "
                                        "enter the same value in the Meta dashboard"),
        )
    ),
    webhooks=(WebhookRoute(path="/webhook", methods=("GET", "POST"), needs_raw_body=True),),
    capabilities=ChannelCapabilities(
        supports_webhook_ingress=True,
        supports_webhook_setup=False,
        supports_teardown=False,
        supports_oauth=False,
        supports_direct_credentials=True,
        supports_custom_ack=False,
    ),
)
