# Scheduled tasks API

Everything the Scheduled Tasks page does goes through a REST API under
`/api/v1/schedules/`. A script can create a task, run it now, follow its runs
and read what each run wrote. The same routes serve the web app, the desktop
app and self-hosted deployments; only the credential differs.

## Routes

| Method and path | What it does |
|---|---|
| `GET /api/v1/schedules/` | Lists your tasks: `{"schedules": [...]}` |
| `POST /api/v1/schedules/` | Creates a task. Answers 201 with the task |
| `GET /api/v1/schedules/{id}` | One task |
| `PATCH /api/v1/schedules/{id}` | Changes the fields you send. `PUT` does the same |
| `DELETE /api/v1/schedules/{id}` | Deletes the task and its run history. Answers 204 |
| `POST /api/v1/schedules/{id}/pause` | Stops it running until resumed |
| `POST /api/v1/schedules/{id}/resume` | Starts it running again |
| `POST /api/v1/schedules/{id}/run-now` | Starts a run now. Answers 202 with the run's `conversation_id` |
| `GET /api/v1/schedules/{id}/runs` | Its runs, newest first: `{"runs": [...]}`. `?limit=` defaults to 100 |
| `GET /api/v1/conversations/{conversation_id}/items` | What a run wrote: the prompt, then the agent's answer |

Requests take camelCase or snake_case field names. Responses use camelCase.

A task has:

- `title` and `prompt`. Each run starts a new conversation from `prompt` alone,
  with no other context, so write it as complete instructions.
- `cadence`: `once`, `hourly`, `daily`, `weekdays` or `weekly`.
- `next_run_at`: the first run, ISO 8601 with a UTC offset. Recurring tasks
  repeat at that time of day, minute or weekday.
- `timezone`: the IANA zone the cadence follows, such as `Europe/London`.
  Defaults to `UTC`.
- `project_id` (optional): the project the runs belong to. Defaults to your
  general project.
- `model` (optional): leave it out to run on your account's default models.

A run has a `status` of `running`, `success`, `failed` or `cancelled`, its
`conversationId`, `startedAt`, `finishedAt`, `durationMs` and, when it failed,
`error`.

## Credentials

| Deployment | Base URL | Credential |
|---|---|---|
| Hosted (MindsHub) | `https://cowork.mindshub.ai` (staging: `https://cowork.staging.mindshub.ai`) | An `mdb_` API key from your MindsHub account |
| Desktop app | `http://127.0.0.1:26866` | The app's local token: `COWORK_AUTH_TOKEN` in `~/.cowork/.env` |
| Self-hosted | Your deployment's address | A tester's Keycloak access token, as the runbook's scripted questions get it |

Send it as `Authorization: Bearer <credential>`.

- **Hosted.** The gateway checks the key and tells cowork-server whose it is.
  Every call acts as the key's owner, in their organization. Identity headers a
  caller sends are replaced, so a key can't act as someone else.
- **Desktop.** The app writes the token the first time it starts its server.
  Only processes on your machine can reach the port.
- **Self-hosted.** Sign in with `tests/e2e/keycloak_login.py` from the
  cowork-self-hosted repository: call `login()` once and `refresh_if_needed()`
  before each call, as its runbook's
  [Tokens and cost per answer](https://github.com/mindsdb/cowork-self-hosted/blob/main/docs/runbook.md#tokens-and-cost-per-answer)
  section does for scripted questions. Password sign-in is off for testers, so
  a script can't trade an email and password for a token.

Keep keys and tokens out of shell history and logs. The examples read one from
a file.

## Create a task, run it, read its output

```sh
BASE=https://cowork.mindshub.ai/api/v1
AUTH="Authorization: Bearer $(cat ~/.config/cowork/api-key)"

# Create a daily task that starts tomorrow at 09:00 UTC.
curl -sS -X POST "$BASE/schedules/" -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "title": "Daily pipeline summary",
  "prompt": "Summarize yesterday'\''s closed deals from the sales-db connection: count, total value and the three largest.",
  "cadence": "daily",
  "timezone": "UTC",
  "next_run_at": "2026-11-01T09:00:00+00:00"
}'
# → 201 {"id": "<task id>", ...}

# Run it now instead of waiting.
curl -sS -X POST "$BASE/schedules/<task id>/run-now" -H "$AUTH"
# → 202 {"detail": "Run triggered", "conversation_id": "<conversation id>"}

# Poll until the newest run is no longer "running".
curl -sS "$BASE/schedules/<task id>/runs?limit=1" -H "$AUTH"
# → {"runs": [{"status": "success", "durationMs": 12613, "conversationId": "...", ...}]}

# Read what the run wrote.
curl -sS "$BASE/conversations/<conversation id>/items" -H "$AUTH"
# → [{"role": "user", "content": "<the prompt>"}, {"role": "assistant", "content": "<the answer>"}]

# Delete the task.
curl -sS -X DELETE "$BASE/schedules/<task id>" -H "$AUTH"
# → 204
```

On the desktop, use `BASE=http://127.0.0.1:26866/api/v1` and read the token
from `~/.cowork/.env`:

```sh
AUTH="Authorization: Bearer $(sed -n 's/^COWORK_AUTH_TOKEN=//p' ~/.cowork/.env)"
```

## Full schema

The server describes every route in OpenAPI.

- **Desktop:** `curl -H "$AUTH" http://127.0.0.1:26866/openapi.json`. The
  interactive page at `/docs` needs the same token, so a browser can't open it
  directly.
- **Hosted:** not published. The hosted address serves only `/api` from
  cowork-server, and `/openapi.json` there is the web app's page.

## What it doesn't do yet

- **No push.** There are no webhooks or callbacks: poll `/runs` to learn when a
  run finishes.
- **No cost on a run.** A run doesn't report its cost or tokens, and its output
  is only in its conversation.
- **Only four statuses.** A run is `success`, `failed`, `cancelled` or
  `running`. It doesn't yet say whether a failed run acted before it failed, or
  which version of the task it ran. The scheduled-task reliability work adds
  both.
