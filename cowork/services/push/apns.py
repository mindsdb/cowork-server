"""Apple Push Notification service sender: token-based auth over HTTP/2.

Authenticates with an ES256 provider token signed by the team's .p8 key
(Apple Developer > Keys), reused for 50 minutes: Apple rejects one older than
an hour and throttles a provider that signs new ones too often. A device
whose environment is unknown is tried in production first; a 400
``BadDeviceToken`` there is how a development (sandbox) build's token shows,
so it is tried once in sandbox, and the environment that took it is kept on
the device.
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
from dataclasses import dataclass
from enum import Enum

import httpx
import jwt

from cowork.services.push.devices import Device
from cowork.services.push.payloads import PushPayload

logger = logging.getLogger(__name__)

PRODUCTION_HOST = "https://api.push.apple.com"
SANDBOX_HOST = "https://api.sandbox.push.apple.com"
_TOKEN_REFRESH_SECONDS = 50 * 60
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# APNs answers these for a token that will never reach an app again.
_DEAD_TOKEN_REASONS = frozenset({"BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic"})
_STALE_PROVIDER_TOKEN_REASONS = frozenset({"ExpiredProviderToken", "InvalidProviderToken"})


class Delivery(Enum):
    SENT = "sent"
    # The token reaches no app: delete the device.
    DEAD = "dead"
    FAILED = "failed"


@dataclass(frozen=True)
class ApnsResult:
    delivery: Delivery
    # The environment that accepted the token when it differs from the
    # device's recorded one, to save on the device.
    environment: str | None = None


def read_key(value: str) -> str:
    """The .p8 key as PEM, from PEM or base64 of it. ValueError otherwise."""
    text = value.strip()
    if "-----BEGIN" in text:
        return text
    try:
        decoded = base64.b64decode(text, validate=True).decode("utf-8").strip()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ValueError("COWORK_PUSH_APNS_KEY_P8 is neither PEM nor base64 of PEM") from None
    if "-----BEGIN" not in decoded:
        raise ValueError("COWORK_PUSH_APNS_KEY_P8 is neither PEM nor base64 of PEM")
    return decoded


class ApnsSender:
    def __init__(
        self, *, key_pem: str, key_id: str, team_id: str, topic: str,
        transport: httpx.AsyncBaseTransport | None = None,
        clock=time.time,
    ) -> None:
        self._key_pem = key_pem
        self._key_id = key_id
        self._team_id = team_id
        self._topic = topic
        self._transport = transport
        self._clock = clock
        self._token: str | None = None
        self._token_issued_at = 0.0

    def provider_token(self) -> str:
        now = self._clock()
        if self._token is None or now - self._token_issued_at >= _TOKEN_REFRESH_SECONDS:
            self._token = jwt.encode(
                {"iss": self._team_id, "iat": int(now)},
                self._key_pem,
                algorithm="ES256",
                headers={"kid": self._key_id},
            )
            self._token_issued_at = now
        return self._token

    def _client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT)
        return httpx.AsyncClient(http2=True, timeout=_TIMEOUT)

    async def _post(self, client: httpx.AsyncClient, host: str, device: Device, payload: PushPayload) -> tuple[int, str]:
        response = await client.post(
            f"{host}/3/device/{device.token}",
            json=payload.apns_body(),
            headers={
                "authorization": f"bearer {self.provider_token()}",
                "apns-topic": self._topic,
                "apns-push-type": "alert",
                "apns-priority": "10",
            },
        )
        reason = ""
        if response.status_code != 200:
            try:
                reason = str(response.json().get("reason") or "")
            except ValueError:
                reason = ""
        return response.status_code, reason

    async def send(self, device: Device, payload: PushPayload) -> ApnsResult:
        sandbox_first = device.environment == "sandbox"
        async with self._client() as client:
            status, reason = await self._post(
                client, SANDBOX_HOST if sandbox_first else PRODUCTION_HOST, device, payload,
            )
            if status == 400 and reason == "BadDeviceToken" and not sandbox_first:
                # A development build's token: production doesn't know it.
                status, reason = await self._post(client, SANDBOX_HOST, device, payload)
                if status == 200:
                    return ApnsResult(Delivery.SENT, environment="sandbox")
        if status == 200:
            return ApnsResult(Delivery.SENT)
        if status == 410 or reason in _DEAD_TOKEN_REASONS:
            return ApnsResult(Delivery.DEAD)
        if reason in _STALE_PROVIDER_TOKEN_REASONS:
            # Sign a fresh one for the next send.
            self._token = None
        logger.warning(
            "APNs refused a push",
            extra={"event": "push_apns_refused", "status_code": status, "reason": reason},
        )
        return ApnsResult(Delivery.FAILED)
