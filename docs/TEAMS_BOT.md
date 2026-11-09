# Teams bot on a self-hosted Cowork

On self-hosted, a Teams question reaches Cowork from the recording proxy as a normal turn, run as
the person who asked. Cowork adds two internal routes for the outgoing side: posting the reply in
the Teams thread, and looking up a person. Only the recording proxy may call them.

The code is in `cowork/channels/teams.py`; it uses the [Teams client](TEAMS_CLIENT.md).

## Settings

The operator sets the bot in `.env`, and compose passes it to Cowork under these names. This is
the Teams equivalent of `python -m cowork.ops.channels`: one install has one bot, set once.

| Variable | Value |
|---|---|
| `COWORK_TEAMS_BOT_APP_ID` | The bot's App ID |
| `COWORK_TEAMS_BOT_APP_SECRET` | Its client secret |
| `COWORK_TEAMS_TENANT_ID` | The organization's Entra tenant ID |
| `COWORK_TEAMS_INTERNAL_KEY` | A random key shared with the recording proxy only |
| `COWORK_TEAMS_SERVICE_HOSTS` | Hosts replies may go to, comma-separated. Default `smba.trafficmanager.net` |
| `COWORK_TEAMS_ALLOWED_USERS` | The webhook only: Entra object IDs of the people it answers, comma-separated |

Every name starts with `COWORK_`, which anton's sandbox never passes to the agent's code. With
none set, Teams is off. With some missing, or on a non-Linux install, Teams is off and the server
log says why. Cowork starts either way.

## Routes

| Route | Does |
|---|---|
| `POST /api/v1/channels/teams/deliver` | JSON `service_url`, `conversation_id`, `reply_to_id`, `text`. Posts `text` as a reply in that thread; answers `{"id": <new message id>}`. |
| `GET /api/v1/channels/teams/member` | Query `service_url`, `conversation_id`, `member_id`. Answers the person's `id`, `name`, `email`, `user_principal_name`, `aad_object_id` and `tenant_id`. |

Both answer:

- 401 when the call isn't signed as below;
- 403 when `service_url` isn't `https://` on one of `COWORK_TEAMS_SERVICE_HOSTS` (default port, no
  user name), so the bot's token never goes to an address a forged message named;
- 502 when Microsoft refuses the bot's credentials or the Teams service fails.

## Signing a call

The agent's code runs in the Cowork container and can reach Cowork's port, so the network alone
doesn't keep it out. Each call carries two headers:

- `X-Cowork-Teams-Timestamp`: the current Unix time in seconds. Cowork refuses one more than 60
  seconds off.
- `X-Cowork-Teams-Signature`: hex HMAC-SHA256, keyed with `COWORK_TEAMS_INTERNAL_KEY`, of

  ```
  METHOD \n target \n timestamp \n body
  ```

  where `target` is the path plus `?query` exactly as sent, and `body` is the raw request body
  (empty for GET).

```python
import hashlib, hmac, time

def headers(key: bytes, method: str, target: str, body: bytes = b"") -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = b"\n".join([method.upper().encode(), target.encode(), timestamp.encode(), body])
    return {
        "X-Cowork-Teams-Timestamp": timestamp,
        "X-Cowork-Teams-Signature": hmac.new(key, message, hashlib.sha256).hexdigest(),
    }
```

The routes skip Cowork's bearer token and identity headers: the signature is their credential.

Two limits:

- The key keeps the agent's code out only with the code sandbox on (`COWORK_CODE_SANDBOX=landlock`).
  The agent's code doesn't get `COWORK_*` variables, and Landlock stops it reading the server's
  environment from `/proc`. Without the sandbox, it runs as the server's user and could read the
  key there. A self-hosted Cowork always has the sandbox on, since identity headers need it.
- A signed call can be sent again within the 60 seconds, which would post the same reply twice. Only
  the proxy and Cowork see these calls, on their own internal network, so this is accepted.

## Checking the bot

After setup, inside the Cowork container:

```
python -m cowork.channels.teams check
```

It asks Microsoft for the bot's token and prints whether Microsoft accepted the credentials. It
exits 0 when they work and the internal key is set, 1 when Microsoft refuses them or the key is
missing, and 2 when Teams isn't configured or available.

## The webhook, for a developer's tunnel

With `COWORK_TEAMS_BOT_APP_ID` set on Linux, Teams is also a channel: Microsoft can post each
@mention straight to `POST /api/v1/channels/teams/api/messages`, and the agent's answer goes back in
the same thread. Point the bot's messaging endpoint there through a tunnel to try the bot without
the recording proxy. The code is in `cowork/channels/plugins/teams.py`.

Before anything is stored or run, each message must carry Microsoft's token:

- signed with one of Microsoft's keys from `login.botframework.com`, endorsed for the message's
  channel (keys are fetched at start and refreshed daily);
- issued by `https://api.botframework.com` for this bot's App ID, and in date (5 minutes' leeway);
- naming the message's `serviceUrl`.

Then every tenant ID the message carries (`conversation.tenantId`, `channelData.tenant.id`, the
`x-ms-tenant-id` header) must be `COWORK_TEAMS_TENANT_ID`; with no tenant set, every message is
refused.

Last, the sender. A channel turn runs as this install's owner, with their connections (mail,
files and the rest), so anyone in the tenant who could @mention the bot could ask the agent to
read them. The webhook answers only the people on `COWORK_TEAMS_ALLOWED_USERS`, by their Entra
object ID (`from.aadObjectId`), and nobody when it's empty. A refused sender's ID is in the server
log, to copy into the list.

A refused message gets 401, no reply and nothing stored, and the server log says why. Keep a test
bot's app out of the organization's app catalogue.

The address to reply to is kept in memory, so a reply to a question asked before a restart is
lost.

A channel turn runs with no person, so on a Cowork that trusts identity headers the channel isn't
registered, and the webhook would refuse every message anyway; Teams questions go through the
recording proxy there. The channel config
routes (save, delete, reload, set up, tear down) answer 409 for Teams, since the bot is set in the
environment.
