"""Outbound calls to Microsoft Teams: the bot's token, thread replies and
member lookups. Not a channel plugin; the Teams plugin builds on it.

microsoft-teams-api talks to the Teams service and msal gets the bot's token.
We don't use microsoft-teams-apps: it brings its own app server we'd bypass
and requires cryptography>=50, and api's own token client is deprecated in
favour of msal. api decodes inbound tokens without checking their signature,
so verifying what Microsoft sends is the receiver's job, not this module's.

Both packages are Linux-only dependencies (Teams runs only on a server
Cowork), so desktop installs import this module without them and
``TeamsClient`` raises ``TeamsUnavailableError``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

try:
    import msal
    from microsoft_teams.api.activities.message import MessageActivityInput
    from microsoft_teams.api.clients.conversation.client import ConversationClient
    from microsoft_teams.api.models import TeamsChannelAccount
    from microsoft_teams.common.http import Client, ClientOptions
except ImportError:
    msal = None

LOGIN_HOST = "https://login.microsoftonline.com"
BOT_SCOPE = "https://api.botframework.com/.default"


class TeamsUnavailableError(RuntimeError):
    """The Teams packages aren't installed (any non-Linux install)."""


class TeamsAuthError(RuntimeError):
    """Microsoft refused the bot's credentials."""


def teams_available() -> bool:
    return msal is not None


class TeamsClient:
    """One bot's outbound Teams calls.

    The bot token is sent to whatever ``service_url`` a call names, so callers
    must check it against the allowed Teams hosts first.
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        tenant_id: str,
        *,
        http: httpx.AsyncClient | None = None,
        token_http: Any = None,
    ) -> None:
        if msal is None:
            raise TeamsUnavailableError("the Teams packages are installed only on Linux")
        # Single-tenant bot: without a tenant, msal would fall back to the
        # multi-tenant botframework.com authority.
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._app = msal.ConfidentialClientApplication(
            app_id,
            client_credential=app_secret,
            authority=f"{LOGIN_HOST}/{tenant_id}",
            http_client=token_http,
        )
        self._http = http or httpx.AsyncClient(timeout=30)

    async def bot_token(self) -> str:
        # msal is synchronous; it serves a cached token until it nears expiry.
        result = await asyncio.to_thread(self._app.acquire_token_for_client, scopes=[BOT_SCOPE])
        token = result.get("access_token")
        if not token:
            raise TeamsAuthError(f"{result.get('error')}: {result.get('error_description')}")
        return token

    def _conversations(self, service_url: str) -> ConversationClient:
        # _http shares one connection pool across calls instead of a new one per call.
        return ConversationClient(service_url, Client(ClientOptions(token=self.bot_token), _http=self._http))

    async def reply(self, service_url: str, conversation_id: str, activity_id: str, text: str) -> str:
        """Post ``text`` as a reply in the thread of ``activity_id``; returns the new message's id."""
        sent = await self._conversations(service_url).reply_to_activity(
            conversation_id, activity_id, MessageActivityInput(text=text)
        )
        return sent.id

    async def get_member(self, service_url: str, conversation_id: str, member_id: str) -> TeamsChannelAccount:
        return await self._conversations(service_url).members_client.get_by_id(conversation_id, member_id)

    async def aclose(self) -> None:
        await self._http.aclose()
