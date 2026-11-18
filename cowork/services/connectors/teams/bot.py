"""The Teams bot on a self-hosted Cowork: its settings, the recording proxy's signature, and the
bot's client, which the Teams connector's internal routes use
(cowork/api/v1/endpoints/connectors/teams.py).

On self-hosted, a Teams question reaches Cowork from the recording proxy as a normal turn, run as
the person who asked. What Cowork adds for Teams is the outgoing side, which needs the bot's
credentials: posting a reply in a Teams thread, and looking up a person in a conversation.

Only the recording proxy may call those routes. The agent's code runs in this container and can
reach this server's port, so each call carries an HMAC (``sign``) keyed with
``COWORK_TEAMS_INTERNAL_KEY``, which only the proxy and this server hold. Both routes post only to a
``service_url`` whose host is on ``COWORK_TEAMS_SERVICE_HOSTS``, so the bot's token never goes to an
address a forged message named.

The operator sets the bot in .env (``COWORK_TEAMS_BOT_APP_ID``, ``COWORK_TEAMS_BOT_APP_SECRET``,
``COWORK_TEAMS_TENANT_ID``), since one install has one bot. Every Teams secret is a ``COWORK_*``
name, which anton's sandbox never passes to the agent's code.

``python -m cowork.services.connectors.teams check`` asks Microsoft for the bot's token with those
settings, for the operator after setup.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections.abc import Callable
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, status
from pydantic import SecretStr, field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from cowork.common.settings.app_settings import Settings
from cowork.services.connectors.teams.client import (
    TeamsAuthError,
    TeamsClient,
    teams_available,
)

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Cowork-Teams-Signature"
TIMESTAMP_HEADER = "X-Cowork-Teams-Timestamp"
MAX_CLOCK_SKEW_S = 60


class TeamsSettings(Settings):
    model_config = SettingsConfigDict(env_prefix="COWORK_TEAMS_")

    bot_app_id: str = ""
    bot_app_secret: SecretStr = SecretStr("")
    tenant_id: str = ""
    internal_key: SecretStr = SecretStr("")
    # Teams names the address to reply to in each message (serviceUrl), e.g.
    # https://smba.trafficmanager.net/amer/<tenant ID>/.
    service_hosts: Annotated[tuple[str, ...], NoDecode] = ("smba.trafficmanager.net",)

    @field_validator("service_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(host.strip().lower() for host in value.split(",") if host.strip())
        return value

    def missing(self) -> list[str]:
        """The settings that are unset, by environment name."""
        values = {
            "COWORK_TEAMS_BOT_APP_ID": self.bot_app_id,
            "COWORK_TEAMS_BOT_APP_SECRET": self.bot_app_secret.get_secret_value(),
            "COWORK_TEAMS_TENANT_ID": self.tenant_id,
            "COWORK_TEAMS_INTERNAL_KEY": self.internal_key.get_secret_value(),
            "COWORK_TEAMS_SERVICE_HOSTS": ",".join(self.service_hosts),
        }
        return [name for name, value in values.items() if not value]


def sign(key: bytes, method: str, target: str, timestamp: str, body: bytes) -> str:
    """The proxy's signature over one call: hex HMAC-SHA256 of
    ``METHOD \\n target \\n timestamp \\n body``, where ``target`` is the path
    plus ``?query`` when there is one, and ``timestamp`` is Unix seconds."""
    message = b"\n".join([method.upper().encode(), target.encode(), timestamp.encode(), body])
    return hmac.new(key, message, hashlib.sha256).hexdigest()


class TeamsProxySignature:
    """The credential is the recording proxy's HMAC over the call (``sign``).

    Refuses with 401 a call with no signature, a wrong one, or a timestamp
    more than a minute off. There's no person in these calls: the caller is
    the proxy, and the key is what tells it apart from the agent's code,
    which shares this server's network.
    """

    async def check(self, request: Request) -> None:
        bot: TeamsBot = request.app.state.teams_bot
        timestamp = request.headers.get(TIMESTAMP_HEADER, "")
        given = request.headers.get(SIGNATURE_HEADER, "")
        try:
            fresh = abs(time.time() - int(timestamp)) <= MAX_CLOCK_SKEW_S
        except ValueError:
            fresh = False
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        expected = sign(bot.internal_key, request.method, target, timestamp, await request.body())
        # Bytes: compare_digest raises on a str with non-ASCII characters, and
        # Starlette decodes header values as Latin-1.
        if not (fresh and hmac.compare_digest(given.encode("latin-1"), expected.encode())):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing Teams proxy signature")


class TeamsBot:
    """The bot's client, created on first use and shared by both routes."""

    def __init__(self, settings: TeamsSettings, make_client: Callable[[], TeamsClient] | None = None) -> None:
        self.internal_key = settings.internal_key.get_secret_value().encode()
        self.service_hosts = frozenset(settings.service_hosts)
        self._make_client = make_client or (
            lambda: TeamsClient(
                settings.bot_app_id, settings.bot_app_secret.get_secret_value(), settings.tenant_id
            )
        )
        self._client: TeamsClient | None = None
        self._lock = asyncio.Lock()

    async def client(self) -> TeamsClient:
        async with self._lock:
            if self._client is None:
                # Creating it makes msal look up the tenant over the network, synchronously.
                self._client = await asyncio.to_thread(self._make_client)
            return self._client

    def check_service_url(self, service_url: str) -> None:
        parts = urlsplit(service_url)
        if (
            parts.scheme != "https"
            or parts.hostname not in self.service_hosts
            or parts.username is not None
            or parts.port not in (None, 443)
        ):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "service_url is not an allowed Teams host")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def check(settings: TeamsSettings | None = None, make_client: Callable[[], TeamsClient] | None = None) -> int:
    """Ask Microsoft for the bot's token; prints the result and returns an exit code."""
    settings = settings or TeamsSettings()
    missing = [name for name in settings.missing() if name != "COWORK_TEAMS_INTERNAL_KEY"]
    if missing:
        print(f"Teams is not configured: {', '.join(missing)} not set")
        return 2
    if not teams_available():
        print("Teams is not available: the Teams packages are installed only on Linux")
        return 2

    async def token() -> None:
        bot = TeamsBot(settings, make_client)
        try:
            await (await bot.client()).bot_token()
        finally:
            await bot.aclose()

    try:
        asyncio.run(token())
    except (TeamsAuthError, ValueError) as error:
        # msal raises ValueError for an unknown tenant.
        print(f"Microsoft refused the bot's credentials: {error}")
        return 1
    except Exception as error:
        print(f"Couldn't reach Microsoft: {type(error).__name__}: {error}")
        return 1
    print(f"Microsoft accepted the bot's credentials (App ID {settings.bot_app_id}, tenant {settings.tenant_id})")
    if not settings.internal_key.get_secret_value():
        print("COWORK_TEAMS_INTERNAL_KEY is not set, so the proxy can't reach the Teams routes")
        return 1
    return 0
