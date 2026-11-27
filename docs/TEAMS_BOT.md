# Teams connector on a self-hosted Cowork

On self-hosted, a Teams question reaches Cowork from the recording proxy as a normal turn, run as
the person who asked. Cowork adds internal routes for the outgoing side, such as posting the reply in
the Teams thread, and looking up a person. Only the recording proxy may call them.

The settings, signature and client are in `cowork/services/connectors/teams/`, and the routes in
`cowork/api/v1/endpoints/connectors/teams.py`; it uses the [Teams client](TEAMS_CLIENT.md).

## Settings

The operator sets the bot in `.env`, and compose passes it to Cowork under these names: one install
has one bot, set once.

| Variable | Value |
|---|---|
| `COWORK_TEAMS_BOT_APP_ID` | The bot's App ID |
| `COWORK_TEAMS_BOT_APP_SECRET` | Its client secret |
| `COWORK_TEAMS_TENANT_ID` | The organization's Entra tenant ID |
| `COWORK_TEAMS_INTERNAL_KEY` | A random key shared with the recording proxy only |
| `COWORK_TEAMS_SERVICE_HOSTS` | Hosts replies may go to, comma-separated. Default `smba.trafficmanager.net` |
| `COWORK_TEAMS_WEB_UI_URL` | The web UI's address, for the link under a scheduled run's result. No link without it |

Every name starts with `COWORK_`, which anton's sandbox never passes to the agent's code. With
none set, Teams is off. With some missing, or on a non-Linux install, Teams is off and the server
log says why. Cowork starts either way.

## Routes

| Route | Does |
|---|---|
| `POST /api/v1/connectors/teams/deliver` | JSON `service_url`, `conversation_id`, `reply_to_id`, `text`, and optionally `mention`: `{"id": <member's Teams id>, "name": <display name>}`. Posts `text` as a reply in that thread, starting with an @mention of that member when given, so Teams notifies them. With `stream_id` from the status route, the reply closes that stream and takes the status line's place, and answers that stream id when Teams gives the reply none; if Teams refuses (it closes a stream after 10 minutes), it goes out as a plain reply. The text goes in Teams' extended markdown, so tables, headings and fenced code show as written; an image becomes a plain link and an HTML tag is shown as text, code included, so no reader's Teams fetches an address the answer put there. Answers `{"id": <new message id>}`. |
| `POST /api/v1/connectors/teams/status` | JSON `service_url`, `conversation_id`, `reply_to_id`, `text` (up to 1,000 characters), and for a later update `stream_id` and a rising `sequence`. Shows `text` as a Teams informative update under that message: a status line with no notification, replaced by each later update and by the reply that names its stream. Answers `{"stream_id": <id>}`, or `{"stream_id": null}` when Teams gave the first update no id: no stream opened, so the answer goes out as a plain reply. |
| `GET /api/v1/connectors/teams/member` | Query `service_url`, `conversation_id`, `member_id`. Answers the person's `id`, `name`, `email`, `user_principal_name`, `aad_object_id` and `tenant_id`. |
| `PUT /api/v1/connectors/teams/contacts` | JSON `user_id` (the person's Cowork user), `tenant_id`, `aad_object_id`, `service_url`. Keeps where to reach that person in Teams, as the proxy pairs them. Answers 204. |
| `DELETE /api/v1/connectors/teams/contacts` | Query `user_id`. Forgets it, as the person disconnects. Answers 204. |

Each answers 401 when the call isn't signed as below, 413 when the body is over 256 KB (refused
before the rest is read), and 422 for a malformed field. Beyond that:

- the routes that take a `service_url` (all but `DELETE /contacts`) answer 403 when it isn't
  `https://` on one of `COWORK_TEAMS_SERVICE_HOSTS` (default port, no user name), so the bot's token
  never goes to an address a forged message named;
- an ID is refused with 422 when it holds `/`, `?`, `#`, `%`, `\` or whitespace, or is only dots, since Microsoft's
  client puts IDs into the URL path unquoted; also when `text` is over 40,000 characters (the proxy
  sends parts of at most 38,000 UTF-16 units);
- the routes that call Teams (`/deliver`, `/status`, `/member`) answer 502 when Microsoft refuses the
  bot's credentials or the Teams service fails. The contacts routes only write Cowork's database.

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

## Scheduled results in Teams

A schedule with `deliverToTeams` posts each completed run's result to its owner's personal chat with
the bot: the task's name, the run's answer and, with `COWORK_TEAMS_WEB_UI_URL`, a link to the run.
A long answer goes in parts. A run that failed posts a short notice with the link instead, so someone
who reads results only in Teams hears when a task breaks; trial runs post nothing. It needs the
owner's contact from the proxy, so the owner must have connected Teams. Where the Teams bot isn't set up (desktop, cloud), the option does nothing and runs record no delivery. A Microsoft ID reaches one person: when the proxy pairs it with someone else, the earlier owner's results stop going there.

The proxy sends the contact with each message a paired person sends the bot. After upgrading to this
release, someone who connected Teams before it has no contact until they next message the bot, and
until then their runs record `failed`: ask people to message the bot once after the upgrade.

Each run's `teamsDelivery` says how it went: `delivered`, `failed`, or `unknown` when it may have
reached Teams: the server stopped while sending, Teams timed out or answered 5xx after the post was
sent, or a later part failed after an earlier one went out. A result is tried once and never sent again on its own, so nobody gets one
twice.

## Checking the bot

After setup, inside the Cowork container:

```
python -m cowork.services.connectors.teams check
```

It asks Microsoft for the bot's token and prints whether Microsoft accepted the credentials. It
exits 0 when they work and the internal key is set, 1 when Microsoft refuses them or the key is
missing, and 2 when Teams isn't configured or available.

## The connector

With every setting above and the Teams packages, the connector directory also lists **Microsoft
Teams (Bot)** (`cowork/services/connectors/specs/microsoft_teams_bot.json`), beside the Graph
"Microsoft Teams" connector. It is a bot, not a data source: its one method, `teams_connection`,
saves no credential, and the agent's connector lookup never offers it. Where Teams isn't set up,
it isn't listed or served at all. The self-hosted stack's recording proxy answers its Connect and
Disconnect, since the pairing between a Teams user and a person lives there.
