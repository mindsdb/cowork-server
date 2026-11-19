"""Firebase Cloud Messaging sender (HTTP v1).

Authenticates as a Firebase service account: a JWT signed with the account's
private key is exchanged at its token URI for an OAuth2 access token with the
``firebase.messaging`` scope, which is reused until five minutes before it
expires. The exchange is the documented JWT-bearer grant, so it needs no
Google client library.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass

import httpx
import jwt

from cowork.services.push.apns import Delivery
from cowork.services.push.devices import Device
from cowork.services.push.payloads import PushPayload

logger = logging.getLogger(__name__)

SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
_DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
_SEND_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
_TOKEN_LIFETIME_SECONDS = 3600
_REFRESH_MARGIN_SECONDS = 300
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass(frozen=True)
class ServiceAccount:
    project_id: str
    client_email: str
    private_key: str
    token_uri: str = _DEFAULT_TOKEN_URI


def read_service_account(value: str) -> ServiceAccount:
    """The service account from its JSON, or from base64 of it. ValueError otherwise."""
    text = value.strip()
    if not text.startswith("{"):
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise ValueError("COWORK_PUSH_FCM_SERVICE_ACCOUNT_JSON is neither JSON nor base64 of JSON") from None
    try:
        data = json.loads(text)
        return ServiceAccount(
            project_id=str(data["project_id"]),
            client_email=str(data["client_email"]),
            private_key=str(data["private_key"]),
            token_uri=str(data.get("token_uri") or _DEFAULT_TOKEN_URI),
        )
    except (ValueError, KeyError, TypeError):
        raise ValueError(
            "COWORK_PUSH_FCM_SERVICE_ACCOUNT_JSON needs project_id, client_email and private_key"
        ) from None


class FcmSender:
    def __init__(
        self, account: ServiceAccount, *, transport: httpx.AsyncBaseTransport | None = None,
        clock=time.time,
    ) -> None:
        self._account = account
        self._transport = transport
        self._clock = clock
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT)

    async def access_token(self, client: httpx.AsyncClient) -> str:
        async with self._lock:
            now = self._clock()
            if self._access_token is not None and now < self._expires_at - _REFRESH_MARGIN_SECONDS:
                return self._access_token
            assertion = jwt.encode(
                {
                    "iss": self._account.client_email,
                    "scope": SCOPE,
                    "aud": self._account.token_uri,
                    "iat": int(now),
                    "exp": int(now) + _TOKEN_LIFETIME_SECONDS,
                },
                self._account.private_key,
                algorithm="RS256",
            )
            response = await client.post(
                self._account.token_uri,
                data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            )
            response.raise_for_status()
            body = response.json()
            self._access_token = str(body["access_token"])
            self._expires_at = now + float(body.get("expires_in") or _TOKEN_LIFETIME_SECONDS)
            return self._access_token

    async def send(self, device: Device, payload: PushPayload) -> Delivery:
        async with self._client() as client:
            token = await self.access_token(client)
            response = await client.post(
                _SEND_URL.format(project_id=self._account.project_id),
                json=payload.fcm_message(device.token),
                headers={"authorization": f"Bearer {token}"},
            )
        if response.status_code == 200:
            return Delivery.SENT
        error_code, status = _error_of(response)
        if response.status_code == 404 or error_code == "UNREGISTERED" or status == "NOT_FOUND":
            return Delivery.DEAD
        if response.status_code == 401:
            # Fetch a fresh access token for the next send.
            self._access_token = None
        logger.warning(
            "FCM refused a push",
            extra={"event": "push_fcm_refused", "status_code": response.status_code, "reason": error_code or status},
        )
        return Delivery.FAILED


def _error_of(response: httpx.Response) -> tuple[str, str]:
    """FCM's errorCode (from the error details) and the error's status."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return "", ""
    codes = [
        str(detail.get("errorCode"))
        for detail in error.get("details") or []
        if isinstance(detail, dict) and detail.get("errorCode")
    ]
    return (codes[0] if codes else ""), str(error.get("status") or "")
