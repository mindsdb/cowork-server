# Teams client

`cowork/channels/teams_client.py` makes cowork-server's outbound calls to Microsoft Teams: it gets
the bot's token, replies in a thread and looks up a member. It isn't a channel plugin, and nothing
calls it yet. Receiving messages comes with the Teams plugin.

## Where it runs

Teams runs only on a server Cowork, so `microsoft-teams-api` and `msal` are main dependencies
marked `sys_platform == 'linux'`:

- Linux installs get them, including cowork's web image and the self-hosted release (on its next
  pin bump).
- macOS and Windows desktop installs don't. There, `teams_available()` returns `False` and
  `TeamsClient(...)` raises `TeamsUnavailableError`. Import the module freely, but check
  `teams_available()` before building a client.

## What it needs

Three values from the customer's own bot, which their operator sets up:

| Value | Where it comes from |
|---|---|
| App ID | Entra → App registrations → the bot's app → Application (client) ID |
| Client secret | The same app → Certificates & secrets |
| Tenant ID | Entra → Overview → Tenant ID |

The bot is single tenant, so the tenant ID is required: the token comes from that tenant's
sign-in endpoint, not the multi-tenant one.

## Example

Getting the bot's token is the quickest check that the three values are right. On Linux, or
anywhere with the packages installed:

```python
import asyncio
import os

from cowork.channels.teams_client import TeamsClient, teams_available


async def main() -> None:
    if not teams_available():
        raise SystemExit("The Teams packages are installed only on Linux.")
    client = TeamsClient(os.environ["APP_ID"], os.environ["APP_SECRET"], os.environ["TENANT_ID"])
    try:
        await client.bot_token()
        print("Microsoft accepted the bot's credentials.")
    finally:
        await client.aclose()


asyncio.run(main())
```

A wrong value fails as follows:

- Unknown tenant ID: msal raises `ValueError` ("Unable to get authority configuration") when the
  client is created.
- App ID not found in that tenant: `TeamsAuthError` starting `unauthorized_client: AADSTS700016`.
- Wrong secret: `TeamsAuthError` starting `invalid_client`.

Replying needs three values from an incoming message: its `serviceUrl`, the conversation ID and
the message's ID.

```python
await client.reply(service_url, conversation_id, activity_id, "Hello from Cowork")
member = await client.get_member(service_url, conversation_id, member_id)
```

Each call sends the bot's token to `service_url`. Check it against the allowed Teams hosts before
calling, so a forged message can't send the token elsewhere.
