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
    message = b"\n".join([method.encode(), target.encode(), timestamp.encode(), body])
    return {
        "X-Cowork-Teams-Timestamp": timestamp,
        "X-Cowork-Teams-Signature": hmac.new(key, message, hashlib.sha256).hexdigest(),
    }
```

The routes skip Cowork's bearer token and identity headers: the signature is their credential.

## Checking the bot

After setup, inside the Cowork container:

```
python -m cowork.channels.teams check
```

It asks Microsoft for the bot's token and prints whether Microsoft accepted the credentials. It
exits 0 when they work and the internal key is set, 1 when Microsoft refuses them or the key is
missing, and 2 when Teams isn't configured or available.
