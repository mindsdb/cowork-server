"""Outbound calls to Microsoft Teams: the bot's token, thread replies and
member lookups. The Teams connector's routes (cowork/api/v1/endpoints/connectors/teams.py) build on it.

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
import html
import re
from typing import Any

import httpx

try:
    import msal
    from microsoft_teams.api.activities.message import MessageActivityInput
    from microsoft_teams.api.activities.typing import TypingActivityInput
    from microsoft_teams.api.clients.conversation.client import ConversationClient
    from microsoft_teams.api.models import Account, ChannelData, StreamInfoEntity, TeamsChannelAccount
    from microsoft_teams.common.http import Client, ClientOptions
except ImportError:
    msal = None

LOGIN_HOST = "https://login.microsoftonline.com"
BOT_SCOPE = "https://api.botframework.com/.default"
# Teams' extended markdown shows tables, headings, fenced code and task lists as an agent writes them;
# its default markdown shows only bold, italic and links. The Teams SDK marks it public preview, so
# "markdown" is the fallback if Microsoft changes it.
TEXT_FORMAT = "extendedmarkdown"

# Text that opens with a markdown block: a list item, heading, table, code block or quote.
_STARTS_A_BLOCK = re.compile(r"( {4}|\t)|\s*([-*+][ \t]|\d+[.)][ \t]|#|\||```|~~~|>)")
# A "<" that could open an HTML tag or comment: <img, <svg, </p>, <!--, <?xml.
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")


def without_images(text: str) -> str:
    """TEXT with each markdown image turned into a plain link, and each HTML tag shown as text.

    An agent's answer can carry text a document planted, and an image would make every reader's
    Teams fetch its address, carrying whatever the answer put in it, without a click. That covers
    <img>, <svg><image>, <video poster>, <picture>, <input type=image> and style="url(...)", so no
    tag is let through. Code is filtered too: telling code apart the way Markdown does is easy to
    get wrong (an unclosed ```, escaped backticks, a backtick inside an attribute), so in code an
    image shows as a link and a tag starts with &lt;."""
    return _TAG_START.sub("&lt;", re.sub(r"!+(?=\[)", "", text))


# What Microsoft's client returns as an activity's id when Teams answered with an empty body.
_NO_ID = "DO_NOT_USE_PLACEHOLDER_ID"


def _real_id(activity_id: str | None) -> str | None:
    return activity_id if activity_id and activity_id != _NO_ID else None


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
            # requests waits forever without a timeout; msal ignores it when http_client is set.
            timeout=15,
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

    async def status(
        self, service_url: str, conversation_id: str, activity_id: str, text: str,
        *, stream_id: str | None = None, sequence: int = 1,
    ) -> str | None:
        """Show ``text`` as an informative update under the message ``activity_id``: a status line,
        not a message, with no notification. The first update opens a stream and returns its id;
        later ones name it with a rising ``sequence``, and each replaces the one before. A reply
        with that ``stream_id`` closes the stream and takes the status line's place.

        Returns None when Teams gave the first update no id: no stream opened, so the answer goes
        out as a plain reply.
        """
        update = TypingActivityInput(text=text, reply_to_id=activity_id)
        update.channel_data = ChannelData(stream_id=stream_id, stream_type="informative", stream_sequence=sequence)
        update.add_entity(StreamInfoEntity(stream_id=stream_id, stream_type="informative", stream_sequence=sequence))
        sent = await self._conversations(service_url).create_activity(conversation_id, update)
        # Teams answers a later update with an empty body, so it keeps the stream's first id.
        return stream_id or _real_id(sent.id)

    async def reply(
        self, service_url: str, conversation_id: str, activity_id: str, text: str,
        *, mention: tuple[str, str] | None = None, stream_id: str | None = None,
    ) -> str:
        """Post ``text`` as a reply in the thread of ``activity_id``; returns the new message's id.

        ``mention``, a member's (id, name), @mentions them ahead of the text, so Teams notifies them.
        ``stream_id``, from ``status``, posts it as that stream's final message instead. The text is
        sent in Teams' extended markdown (TEXT_FORMAT).
        """
        text = without_images(text)
        activity = MessageActivityInput(text=text)
        if mention:
            member_id, name = mention
            # The name is markup inside <at>; the entity's text must match the message's exactly.
            tag = f"<at>{html.escape(name, quote=False)}</at>"
            # On a line of its own before a block, or Teams runs the block's first line into the name.
            gap = "\n\n" if _STARTS_A_BLOCK.match(text) else " "
            activity = MessageActivityInput(text=f"{tag}{gap}{text}").add_mention(
                Account(id=member_id, name=name), text=html.escape(name, quote=False), add_text=False
            )
        activity = activity.with_text_format(TEXT_FORMAT)
        if stream_id:
            activity.reply_to_id = activity_id
            activity.channel_data = ChannelData(stream_id=stream_id, stream_type="final")
            activity.add_entity(StreamInfoEntity(stream_id=stream_id, stream_type="final"))
            sent = await self._conversations(service_url).create_activity(conversation_id, activity)
            # Teams usually answers the stream's final message with an empty body.
            return _real_id(sent.id) or stream_id
        sent = await self._conversations(service_url).reply_to_activity(conversation_id, activity_id, activity)
        return sent.id

    async def get_member(self, service_url: str, conversation_id: str, member_id: str) -> TeamsChannelAccount:
        return await self._conversations(service_url).members_client.get_by_id(conversation_id, member_id)

    async def aclose(self) -> None:
        await self._http.aclose()
