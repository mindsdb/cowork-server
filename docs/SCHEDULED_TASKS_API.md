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
| `POST /api/v1/schedules/{id}/resume` | Starts it running again. Answers 409 for a one-time task whose time has passed |
| `POST /api/v1/schedules/{id}/run-now` | Starts a run now. Answers 202 with the run's `conversation_id`, or 409 if the task is already running or too many tasks are running |
| `GET /api/v1/schedules/{id}/runs` | Its runs, newest first: `{"runs": [...]}`. `?limit=` defaults to 100 |
| `GET /api/v1/conversations/{conversation_id}/items` | What a run wrote: the prompt, then the agent's answer |

Requests take camelCase or snake_case field names. Responses use camelCase.

A task has:

- `title` and `prompt`. Each run starts a new conversation from `prompt` alone,
  with no other context, so write it as complete instructions.
- `cadence`: `once`, `every_15_minutes`, `every_30_minutes`, `hourly`,
  `daily`, `weekdays` or `weekly`. A server older than the 15 and 30 minute
  cadences answers 422 for them.
- `next_run_at`: the first run, ISO 8601 with a UTC offset. Recurring tasks
  repeat at that time of day, minute or weekday.
- `timezone`: the IANA zone the cadence follows, such as `Europe/London`.
  Defaults to `UTC`.
- `project_id` (optional): the project the runs belong to. Defaults to your
  general project.
- `model` (optional): leave it out to run on your account's default models.

A task in a response also has:

- `state`: `active`, `paused`, `auto_paused`, `missed` or `done`. A task
  pauses itself (`auto_paused`) after three failed runs in a row. A one-time
  task is `missed` when its time passed while nothing could run it, and `done`
  once it has run. Resume a paused task with `/resume`.
- `running`: whether a run is going now.
- `runCount` and `lastRun`: how many runs it has had, and the latest finished
  one.

A run has:

- `status`: `running`, `success`, `failed` or `cancelled`.
- `outcome`, once it ends: `completed`, `failed_before_execution`,
  `failed_during_execution`, `interrupted`, `cancelled` or `uncertain`. Only
  `failed_before_execution` means the agent did nothing, so a failed run with
  any other outcome may have acted.
- `trigger`: `cron`, `manual` (run-now) or `trial`.
- `conversationId`, `startedAt`, `finishedAt`, `durationMs` and, when it
  failed, `error`.

A run is stopped once it passes the deployment's time limit, 30 minutes by
default. It then ends `failed` with outcome `interrupted`.

## Credentials

| Deployment | Base URL | Credential |
|---|---|---|
| Hosted (MindsHub) | `https://cowork.mindshub.ai` (staging: `https://cowork.staging.mindshub.ai`) | An `mdb_` API key from your MindsHub account |
| Desktop app | None a script can use | The hosted API with an `mdb_` key, or a `cowork-server` you start from a terminal |
| `cowork-server` started from a terminal | `http://127.0.0.1:26866` | Its local token: `COWORK_AUTH_TOKEN` in `~/.cowork/.env`, or in `~/.cowork-scripts/.env` when it has its own home beside the desktop app |
| Self-hosted | Your deployment's address | A tester's Keycloak access token |

Send it as `Authorization: Bearer <credential>`.

- **Hosted.** The gateway checks the key and tells cowork-server whose it is.
  Every call acts as the key's owner, in their organization. Identity headers a
  caller sends are replaced, so a key can't act as someone else.
- **Desktop.** The desktop app hands the server it starts a new token for each
  launch, on the server's stdin, and the token stays in the app's and the
  server's memory. Nothing on disk holds it, so a script can't read it. Script
  your tasks against the hosted API with an `mdb_` key, or start a server of
  your own from a terminal.
- **`cowork-server` from a terminal.** The server writes its token to
  `~/.cowork/.env` when it starts and finds none, or takes `COWORK_AUTH_TOKEN`
  from its environment. Only processes on your machine can reach the port.
  If you also run the desktop app, give this server its own home:
  `COWORK_HOME=~/.cowork-scripts cowork-server`. The desktop app's server can
  share `~/.cowork/.env`, and it removes the token line there each time it
  starts, so a script that reads the token from that file then gets nothing.
  A server with its own home keeps its own tasks, data and sign-in, and
  writes its token to `~/.cowork-scripts/.env`.
- **Self-hosted.** Sign the tester in the way the web app does: Keycloak's
  authorization-code flow with PKCE, against the client the web app uses.
  Send the access token it returns, and before each call trade the refresh
  token for a new one if the access token is about to expire. Password
  sign-in is off for testers, so a script can't trade an email and password
  for a token.

Keep keys and tokens out of shell history and logs. The examples read one from
a file.

## Create a task, run it, read its output

```sh
BASE=https://cowork.mindshub.ai/api/v1
AUTH="Authorization: Bearer $(cat ~/.config/cowork/api-key)"

# Create a daily task. Set next_run_at to the first run you want, in the future.
curl -sS -X POST "$BASE/schedules/" -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "title": "Daily pipeline summary",
  "prompt": "Summarize yesterday'\''s closed deals from the sales-db connection: count, total value and the three largest.",
  "cadence": "daily",
  "timezone": "UTC",
  "next_run_at": "2030-01-15T09:00:00+00:00"
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

Against a `cowork-server` you started from a terminal, use
`BASE=http://127.0.0.1:26866/api/v1` and read the token from `~/.cowork/.env`:

```sh
AUTH="Authorization: Bearer $(sed -n 's/^COWORK_AUTH_TOKEN=//p' ~/.cowork/.env)"
```

If you also run the desktop app, start that server with its own home,
`COWORK_HOME=~/.cowork-scripts cowork-server`, and read the token from there:

```sh
AUTH="Authorization: Bearer $(sed -n 's/^COWORK_AUTH_TOKEN=//p' ~/.cowork-scripts/.env)"
```

## Full schema

The server describes every route in OpenAPI.

- **`cowork-server` from a terminal:** `curl -H "$AUTH" http://127.0.0.1:26866/openapi.json`.
  The interactive page at `/docs` needs the same token, so a browser can't open
  it directly.
- **Hosted:** not published. The hosted address serves only `/api` from
  cowork-server, and `/openapi.json` there is the web app's page.

## What it doesn't do yet

- **No push.** There are no webhooks or callbacks: poll `/runs` to learn when a
  run finishes.
- **No cost on a run.** A run doesn't report its cost or tokens, and its output
  is only in its conversation.
- **No task version on a run.** A run doesn't say which version of the
  task's prompt and settings it ran, so an edit made after a run isn't
  visible from that run.
