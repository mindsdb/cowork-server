from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx
from anton.core.dispatch import InboundEvent, InboundMessage, PlatformAddress

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
_CUSTOMER_CARE_WINDOW = timedelta(hours=24)
_TRANSIENT_ERROR_CODES = {4, 17, 32, 80007}


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
        if msg.get("type") != "text":
            return None
        text = ((msg.get("text") or {}).get("body") or "").strip()
        sender = (msg.get("from") or "").strip()  # E.164, no '+'
        message_id = str(msg.get("id", ""))
        if not text or not sender or not message_id:
            return None
        try:
            timestamp = datetime.fromtimestamp(int(msg.get("timestamp", "")), timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)
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
            ),
        )
        event._dedupe_key = f"whatsapp:{message_id}"  # type: ignore[attr-defined]
        return event

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        phone_number_id = (self._secrets.get("phone_number_id") or "").strip()
        access_token = (self._secrets.get("access_token") or "").strip()
        if not phone_number_id or not access_token:
            raise RuntimeError("whatsapp phone_number_id/access_token not configured")
        recipient = address.platform_id
        last = self._last_inbound.get(recipient)
        now = datetime.now(timezone.utc)
        if last is None or (now - last) > _CUSTOMER_CARE_WINDOW:
            raise RuntimeError(
                "whatsapp customer-care window expired for this contact; "
                "only pre-approved templates can be sent"
            )
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
            CredentialField(name="verify_token", label="Verify token", secret=True, required=True,
                            description="Operator-chosen token echoed during webhook subscription"),
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
