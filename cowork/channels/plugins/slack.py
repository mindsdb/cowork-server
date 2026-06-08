from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Mapping
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
    OAuthSpec,
    WebhookRoute,
)
from cowork.channels.text import split_for_limit
from cowork.channels.webhooks import SignatureError, WebhookHandshake

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter, ChannelSetup, OutboundMessage

log = logging.getLogger(__name__)

CHANNEL_TYPE = "slack"
SLACK_API_BASE = "https://slack.com/api"
SLACK_MAX_TEXT = 3900  # Slack docs say 4000 chars, but in practice it seems to truncate at 3900 or so.
SLACK_MAX_FILE_BYTES = 20 * 1024 * 1024
SLACK_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_REPLAY_WINDOW_S = 5 * 60


def extract_media(event: dict) -> list[Attachment]:
    """Shared-file descriptors from a message event; bytes are fetched later in
    the background via fetch_attachment (url_private needs the bot token)."""
    media: list[Attachment] = []
    for f in event.get("files") or []:
        if not isinstance(f, dict):
            continue
        url = f.get("url_private_download") or f.get("url_private")
        if not url:
            continue
        if (f.get("size") or 0) > SLACK_MAX_FILE_BYTES:
            log.info("skipping slack attachment over the size cap")
            continue
        attachment = Attachment(
            filename=f.get("name") or "file",
            mime_type=f.get("mimetype") or "application/octet-stream",
        )
        attachment.slack_url = url
        media.append(attachment)
    return media
_OAUTH_SCOPES = (
    "app_mentions:read", "channels:history", "groups:history",
    "im:history", "im:write", "mpim:history", "chat:write",
)


class SlackBridge:
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
        for chunk in split_for_limit(message.text, SLACK_MAX_TEXT):
            await self.send_text(address=message.address, text=chunk)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"*{getattr(card, 'prompt', '')}*\n{bullets}".strip()
        for chunk in split_for_limit(text, SLACK_MAX_TEXT):
            await self.send_text(address=address, text=chunk)

    def try_handshake(
        self, *, method: str, body: bytes, headers: Mapping[str, str], query: Mapping[str, str]
    ) -> WebhookHandshake:
        if method != "POST" or not body:
            return WebhookHandshake(handled=False)
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return WebhookHandshake(handled=False)
        if isinstance(data, dict) and data.get("type") == "url_verification":
            return WebhookHandshake(handled=True, response_body=str(data.get("challenge", "")))
        return WebhookHandshake(handled=False)

    def verify_signature(self, *, body: bytes, headers: Mapping[str, str]) -> None:
        signing_secret = (self._secrets.get("signing_secret") or "").strip()
        timestamp = headers.get("x-slack-request-timestamp", "")
        signature = headers.get("x-slack-signature", "")
        if not signing_secret:
            raise SignatureError("slack signing_secret not configured")
        if not timestamp or not signature.startswith("v0="):
            raise SignatureError("missing/invalid slack signature headers")
        try:
            ts = int(timestamp)
        except ValueError as exc:
            raise SignatureError("non-integer slack timestamp") from exc
        if abs(int(time.time()) - ts) > _REPLAY_WINDOW_S:
            raise SignatureError("slack timestamp outside replay window")
        base = f"v0:{timestamp}:".encode("utf-8") + body
        digest = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, signature):
            raise SignatureError("slack signature mismatch")

    async def parse_inbound(
        self, *, body: bytes, headers: Mapping[str, str], route_name: str | None
    ) -> list[InboundEvent]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(data, dict) or data.get("type") != "event_callback":
            return []
        event = self._normalize_event(data.get("event") or {})
        if event is None:
            return []
        event_id = str(data.get("event_id", "") or "")
        event._dedupe_key = (  # type: ignore[attr-defined]
            f"slack:event:{event_id}" if event_id
            else f"slack:{event.address.platform_id}:{event.message.id}"
        )
        return [event]

    def dedupe_key(self, event: InboundEvent) -> str | None:
        key = getattr(event, "_dedupe_key", None)
        if key:
            return key
        mid = event.message.id
        return f"slack:{event.address.platform_id}:{mid}" if mid else None

    def _normalize_event(self, event: dict) -> InboundEvent | None:
        kind = event.get("type")
        if kind not in ("message", "app_mention"):
            return None
        if event.get("bot_id") or event.get("subtype") in (
            "bot_message", "message_changed", "message_deleted",
        ):
            return None
        text = (event.get("text") or "").strip()
        attachments = extract_media(event)
        if not text and not attachments:
            return None
        channel = event.get("channel", "") or ""
        thread_ts = event.get("thread_ts")
        ts = event.get("ts", "") or ""
        try:
            timestamp = datetime.fromtimestamp(float(ts), timezone.utc) if ts else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)
        is_group = not channel.startswith("D")
        return InboundEvent(
            address=PlatformAddress(channel_type=CHANNEL_TYPE, platform_id=channel, thread_id=thread_ts),
            message=InboundMessage(
                id=ts,
                content=text,
                timestamp=timestamp,
                kind="chat",
                sender_id=event.get("user") or None,
                is_mention=kind == "app_mention",
                is_group=is_group,
                attachments=attachments,
            ),
        )

    async def send_attachment(self, *, address: PlatformAddress, path: str, filename: str | None = None) -> str:
        """Upload via the external-upload flow (files.upload is deprecated):
        getUploadURLExternal → POST bytes → completeUploadExternal."""
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("slack bot_token not configured")
        source = Path(path)
        if source.stat().st_size > SLACK_MAX_UPLOAD_BYTES:
            raise RuntimeError("slack attachment exceeds the upload size cap")
        name = (filename or source.name).strip() or "file"
        data = source.read_bytes()

        issued = await self.web_api(bot_token, "files.getUploadURLExternal",
                                    data={"filename": name, "length": len(data)})
        if not issued.get("ok") or not issued.get("upload_url"):
            raise RuntimeError(f"slack getUploadURLExternal failed: {issued.get('error', 'unknown')}")
        if not await self.upload_bytes(issued["upload_url"], data):
            raise ConnectionError("slack file upload failed")

        payload: dict[str, Any] = {
            "files": [{"id": issued.get("file_id"), "title": name}],
            "channel_id": address.platform_id,
        }
        if address.thread_id:
            payload["thread_ts"] = address.thread_id
        done = await self.web_api(bot_token, "files.completeUploadExternal", json_body=payload)
        if not done.get("ok"):
            raise RuntimeError(f"slack completeUploadExternal failed: {done.get('error', 'unknown')}")
        return str(issued.get("file_id", ""))

    @staticmethod
    async def web_api(bot_token: str, method: str, *, data: dict | None = None,
                      json_body: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{SLACK_API_BASE}/{method}", data=data, json=json_body,
                    headers={"Authorization": f"Bearer {bot_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"slack {method} transport error: {exc!r}") from exc
        return resp.json()

    @staticmethod
    async def upload_bytes(upload_url: str, data: bytes) -> bool:
        # Pre-signed URL from getUploadURLExternal — no auth header needed.
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(upload_url, content=data)
        except (httpx.TimeoutException, httpx.TransportError):
            return False
        return resp.status_code == 200

    async def fetch_attachment(self, attachment: Attachment) -> bytes | None:
        """Best-effort download of a shared file; failures fall back to text-only."""
        url = getattr(attachment, "slack_url", None)
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not url or not bot_token:
            return None
        return await self.download_url(bot_token, url)

    @staticmethod
    async def download_url(bot_token: str, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {bot_token}"})
        except (httpx.TimeoutException, httpx.TransportError):
            return None
        return resp.content if resp.status_code == 200 else None

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        bot_token = (self._secrets.get("bot_token") or "").strip()
        if not bot_token:
            raise RuntimeError("slack bot_token not configured")
        payload: dict[str, Any] = {"channel": address.platform_id, "text": text}
        if address.thread_id:
            payload["thread_ts"] = address.thread_id
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{SLACK_API_BASE}/chat.postMessage",
                    json=payload,
                    headers={"Authorization": f"Bearer {bot_token}"},
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(f"slack transport error: {exc!r}") from exc
        result = resp.json()
        if not result.get("ok"):
            err = result.get("error", "")
            if err in ("ratelimited", "internal_error", "service_unavailable"):
                raise ConnectionError(f"slack transient error: {err}")
            raise RuntimeError(f"slack chat.postMessage failed: {err}")
        return str(result.get("ts", ""))


async def _factory(credentials: Mapping[str, str]) -> ChannelAdapter | None:
    if not (credentials.get("signing_secret") or "").strip():
        return None
    if not (credentials.get("bot_token") or "").strip():
        return None
    return SlackBridge(credentials)


plugin = ChannelPlugin(
    channel_type=CHANNEL_TYPE,
    display_name="Slack",
    factory=_factory,
    credentials=CredentialSchema(
        fields=(
            CredentialField(name="signing_secret", label="Signing secret", secret=True, required=True,
                            description="Verifies inbound Events API requests"),
            CredentialField(name="bot_token", label="Bot token", secret=True, required=True,
                            description="xoxb- token used to post messages"),
            CredentialField(name="client_id", label="OAuth client id", secret=False, required=False,
                            description="App client id (OAuth install)"),
            CredentialField(name="client_secret", label="OAuth client secret", secret=True, required=False,
                            description="App client secret (OAuth install)"),
            CredentialField(name="app_token", label="App-level token", secret=True, required=False,
                            description="xapp- token for Socket Mode (later)"),
        )
    ),
    webhooks=(WebhookRoute(path="/events", methods=("POST",), name="events", needs_raw_body=True),),
    oauth=OAuthSpec(scopes=_OAUTH_SCOPES),
    capabilities=ChannelCapabilities(
        supports_webhook_ingress=True,
        supports_webhook_setup=False,
        supports_teardown=False,
        supports_oauth=True,
        supports_direct_credentials=True,
        supports_custom_ack=False,
    ),
)
