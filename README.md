# Cowork Server

FastAPI backend for [MindsHub Cowork](https://github.com/mindsdb/cowork). Manages projects, conversations, files, scheduling, memory, and agent orchestration with a SQLite-backed data layer.

This repo is the **Python backend**. The **frontend** (Electron shell + React SPA) lives in a separate repo: [`mindsdb/cowork`](https://github.com/mindsdb/cowork). They are developed and released independently. At runtime, the frontend spawns `cowork-server` as a local sidecar and communicates over HTTP (`127.0.0.1:26866`).

## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
# Install and run

uv tool install cowork-server
cowork-server
```

The server starts on `http://127.0.0.1:26866`. Confirm with:

```sh
curl http://127.0.0.1:26866/api/v1/health/
```

## Development

```sh
# Run from source (auto-manages virtualenv + deps)
uv run cowork-server
```

When running alongside the Electron app in dev mode, the app spawns the server automatically — no manual start needed. The Electron app looks for a sibling `cowork-server/` directory by convention (override with `COWORK_SERVER_DIR`).

cowork-server imports anton's artifact folder constants (`anton.core.artifacts.internal_files`) at startup, with no fallback. A local editable anton older than the `anton-agent` floor in `pyproject.toml` makes the server fail to start with `ImportError`.

### Dev setup helper

```sh
uv run cowork-dev-setup
```

Initializes the database and validates configuration.

### Testing

```sh
uv run pytest
```

Tests use an isolated in-memory database and temporary directories — no side effects on your local `~/.cowork/` data.

Post-deploy integration runs use the target cluster's self-hosted runner. Dev
and staging obtain the fixed `cowork` test suite through auth's cluster-only
service URL. Production must not mutate that shared `@emailsink.dev` identity
while the fixed password remains committed. The production test path therefore
accepts a dedicated `COWORK_TEST_API_KEY` secret, a reviewed
`COWORK_TEST_USER_EMAIL` variable on the non-staff `@mindshub.ai` domain, and
the immutable dedicated organization id in `COWORK_TEST_ORG_ID`. The suite
resolves and matches that principal and organization through production auth,
and refuses an employee-classified or Hub-admin identity before testing. A
missing or mismatched identity fails the required prod run instead of falling
back to provisioning or reporting skipped tests. Standing-identity mode also
requires the test target to be exactly `https://cowork.mindshub.ai` before the
first network call, so a changed environment file or workflow cannot send the
production key to another origin.

Do not store or use `COWORK_TEST_API_KEY` yet, or configure its paired email and
organization id for a production run. The live `prod` GitHub Environment has no
protection rules or deployment-branch policy. Although `publish.yml` refuses to
enter its production build/deploy job from a non-main ref, a manually selected
branch runs that branch's workflow text and can remove the check. Workflow code
is therefore defense in depth, not the authority that protects an Environment
secret.

Before `COWORK_TEST_API_KEY` is stored or used, or any production evidence run
starts, the `prod` Environment must have a nonempty required-reviewer rule with
`prevent_self_review: true`, `can_admins_bypass: false`, and a selected-branches
deployment policy whose only entry is the `main` branch, created as an exact
Branch rule (no tag, wildcard, or second branch rule). Verify the live settings
without reading any secret value:

```sh
gh api repos/mindsdb/cowork-server/environments/prod \
  --jq '
    [.protection_rules[]?
      | select(.type == "required_reviewers")
      | {
          prevent_self_review,
          reviewers: [.reviewers[]?
            | {type, name: (.reviewer.login // .reviewer.slug)}]
        }] as $required_reviewers
    | {
        can_admins_bypass,
        required_reviewers: $required_reviewers,
        deployment_branch_policy
      }
  '
gh api 'repos/mindsdb/cowork-server/environments/prod/deployment-branch-policies?per_page=100' \
  --jq '[.branch_policies[] | {name, type}]'
```

The first command must show exactly one required-reviewer rule with at least one
named reviewer and `prevent_self_review: true`, plus `can_admins_bypass: false`,
`protected_branches: false`, and `custom_branch_policies: true`. The second must
print exactly `[{"name":"main","type":"branch"}]`. Only after both checks pass
may an operator store `COWORK_TEST_API_KEY`, `COWORK_TEST_USER_EMAIL`, and
`COWORK_TEST_ORG_ID` on that Environment. Evidence must come from a fresh
main-branch run started after the
protection was active, with `run_attempt: 1`; an eligible reviewer other than the
run's `actor` and `triggering_actor` must approve its `prod` Environment gate. A
rerun of an attempt that began before protection does not count. Keeping the
values at Environment scope prevents non-prod jobs from receiving them, but that
scope is safe only when these Environment controls are active.

#### Nightly production read-only smoke

The production nightly reads production on `mdb-prod` at `43 7 * * *` once its
schedule is live. It is held today; the Environment prerequisites below say what
turns it on. It uses the same guarded standing identity, but selects only
`tests/integration/test_production_read_only.py`. That selection is GET-only:
it reads health, conversations, schedules, files, and pins. It never
provisions an identity and does not create conversations, schedules, files,
artifacts, or model turns. The broad integration target excludes the
`production_read_only` marker, so release and staging callers cannot include
this production-only selection by accident.

Failures and the next recovery use the shared engineering-channel notifier.
The nightly is an alert, not a release gate. Review any new endpoint in
`READ_ONLY_ENDPOINTS` together with
`tests/test_production_read_only_workflow.py`, which rejects mutating HTTP calls
and pins the complete request list.

The unattended nightly cannot reference the existing `prod` Environment.
That Environment must retain its required-reviewer gate for production deploys,
and GitHub pauses every job that references such an Environment until a reviewer
approves it. The monitor instead uses a dedicated `prod-read-only` Environment
with unique `COWORK_PROD_READ_ONLY_API_KEY`,
`COWORK_PROD_READ_ONLY_USER_EMAIL`, and
`COWORK_PROD_READ_ONLY_ORG_ID` inputs. Unique names prevent a missing
Environment value from falling back to a repository or organization credential.

Before promoting this workflow to `main`, an operator must create
`prod-read-only` with no required-reviewer or wait-timer rule and a custom
deployment policy whose only entry is the exact `main` branch. The repository's
`main` protection must continue to require a pull-request approval, resolve
review conversations, and apply to administrators. These controls let the
scheduled job start without weakening the separately protected `prod`
Environment:

```bash
gh api --method PUT repos/mindsdb/cowork-server/environments/prod-read-only \
  --input - <<'JSON'
{
  "wait_timer": 0,
  "prevent_self_review": false,
  "reviewers": [],
  "deployment_branch_policy": {
    "protected_branches": false,
    "custom_branch_policies": true
  }
}
JSON
gh api --method POST \
  repos/mindsdb/cowork-server/environments/prod-read-only/deployment-branch-policies \
  -f name=main -f type=branch
```

Only after those controls exist may the operator copy the already reviewed
standing identity into the dedicated names. The commands prompt for the secret
and variable values and do not print them:

```bash
gh secret set COWORK_PROD_READ_ONLY_API_KEY \
  --repo mindsdb/cowork-server --env prod-read-only
gh variable set COWORK_PROD_READ_ONLY_USER_EMAIL \
  --repo mindsdb/cowork-server --env prod-read-only
gh variable set COWORK_PROD_READ_ONLY_ORG_ID \
  --repo mindsdb/cowork-server --env prod-read-only
```

Verify policy and names without reading credential values:

```bash
gh api repos/mindsdb/cowork-server/environments/prod-read-only \
  --jq '{protection_rules, deployment_branch_policy}'
gh api \
  'repos/mindsdb/cowork-server/environments/prod-read-only/deployment-branch-policies?per_page=100' \
  --jq '[.branch_policies[] | {name, type}]'
gh api repos/mindsdb/cowork-server/branches/main/protection \
  --jq '{required_pull_request_reviews, enforce_admins, required_conversation_resolution}'
gh secret list --repo mindsdb/cowork-server --env prod-read-only \
  | rg '^COWORK_PROD_READ_ONLY_API_KEY\b'
gh variable list --repo mindsdb/cowork-server --env prod-read-only \
  | rg '^COWORK_PROD_READ_ONLY_(USER_EMAIL|ORG_ID)\b'
```

The Environment response must have no `required_reviewers` or `wait_timer`
entry and must enable only custom branch policies. The branch-policy response
must be exactly `[{"name":"main","type":"branch"}]`. The branch-protection
response must show at least one required approval, administrator enforcement,
and required conversation resolution. Do not dispatch until every check passes
and the cowork-server#472 prerequisite has landed.

The nightly schedule is held out of the workflow until those checks pass.
GitHub arms a `schedule:` trigger as soon as the file reaches the default
branch, so shipping one before the Environment exists does not buy a monitor, it
buys a nightly page that reports a missing credential in the same shape as a
production outage. `workflow_dispatch` is the only trigger until an operator has
provisioned `prod-read-only` and watched one dispatched run go green. Restore the
`schedule:` block with `- cron: "43 7 * * *"` and delete this paragraph in the
same change; a contract test fails if only one of the two happens.

### Logging

Set `LOG_LEVEL` (default `WARNING`) to control verbosity. Enable file logging with `ENABLE_FILE_LOGGING=true` (writes to `LOG_DIR`, defaults to `~/.cowork/logs/`).

Database exception records omit SQL, parameter values, driver details and tracebacks, in the owned console and file handlers and in everything Uvicorn logs. `setup_logging()` attaches the database and provider filters to the `uvicorn`, `uvicorn.error`, `uvicorn.access` and `uvicorn.asgi` loggers, and to any handler a launcher already installed on them, so the policy holds under every launcher: the `cowork-server` CLI, `python -m uvicorn cowork.server:app`, or `uvicorn spa_wrapper:app` with Uvicorn's default log config. Uvicorn's levels stay as the launcher set them, so lifecycle lines such as `Finished server process` still print. The shared filter replaces each such record with `Database operation failed: error_type=<class> sqlstate=<code or unknown> site=<module>.<function>:<line>`, keeping only valid driver SQLSTATE codes. `site` names the logging call. When that call only relays the error, as Uvicorn's request runner and the server's lifespan guard do, `site` names the deepest Cowork frame the error passed through outside the session wrappers (`cowork.db.scoped`, `cowork.db.session` and `cowork.db.units`), such as the service line that ran a route's failing statement. Session logs retain error types and client HTTP status, including the 503 for a full pool. A 5xx `HTTPException` logs its status. Its own detail is not logged, but its cause's class, message and traceback are, unless the cause chain holds a database or provider error, which the filters replace. Any other error, including one that carries a `status_code` without being an `HTTPException`, logs only its type in the session line, then reaches Uvicorn, which logs it through the same filters. Exceptions and HTTP errors returned to callers stay unchanged.

A database error while the server starts or stops (a migration, the seed rows, or Postgres not yet reachable after a reboot) logs that safe line through the owned handlers. Uvicorn then prints only `RuntimeError: Database operation failed during server startup` (or `shutdown`), because Starlette hands it the failure as plain traceback text, which no filter can read. Other startup errors keep their traceback.

A call site passes the ids that locate a failure as `extra=log_context(...)` from `cowork.common.logger`, which sets record attributes: `request_id`, `project_id`, `schedule_id`, `conversation_id`, and `artifact_slug` for one artifact or `artifact_slugs` for several. The filters never change them, and the console and file formatters render them before the message, for example `[Req:<id>][Project:<id>][Schedule:<id>][Conversation:<id>][Artifact:'<slug>']` or `[Artifacts:'<slug>', '<slug>']`. Artifact slugs are folder names, so they render quoted. A site that carries only a request id passes `extra={"request_id": ...}` directly.

OpenAI and Anthropic SDK errors (any `OpenAIError` or `AnthropicError`) and Anton's typed provider errors are found, in the same handlers and Uvicorn loggers, through each exception's cause, then its context (even a suppressed one), then its exception group members. The record becomes `Provider operation failed: error_type=<the record's exception class> provider_error=<the first provider error's class> status=<that provider error's HTTP status or unknown>`; a wrapper's own status is never used. Cowork's request ID (`[Req:<id>]`) stays on the line; the provider's own request ID is not logged. Provider messages, bodies, requests and tracebacks are omitted. Routing failure warnings also omit body-derived type, code and parameter fields. Original exceptions, probe verdicts and client HTTP/SSE errors stay unchanged.

Uvicorn's access line keeps each request's method, path and status, and `setup_logging()` replaces its query with `?[redacted]`, because a query can carry an OAuth callback's code or a webhook's verify token.

SQLAlchemy query and connection diagnostics stay at `WARNING`, even when application logging is set to `DEBUG`. SDK request diagnostics and HTTP client logging stay at `ERROR`, including Anthropic, `httpx2` and the whole `httpcore` and `httpcore2` namespaces, because they can include request bodies, URLs or response headers such as `set-cookie`.

## Releasing

Releases are automatic on merge; there is no version to bump by hand (the
package version comes from the tag).

- Push to `main`: [`publish.yml`](.github/workflows/publish.yml) runs the unit
  tests, cuts a CalVer tag and GitHub release (`v0.<yy>.<m>.<d>.<seq>`), then
  builds and publishes to [PyPI](https://pypi.org/project/cowork-server/) via
  OIDC trusted publishing.
- Push to `staging`:
  [`publish-staging.yml`](.github/workflows/publish-staging.yml) does the same on
  the rc pre-release stream (`v0.<yy>.<m>.<d>.<seq>rc<n>`, GitHub and PyPI
  pre-release), pinning the matching `anton-agent` rc into the wheel so the pair
  installs exactly.

Both take their version, tag, and release from the shared `calver-release.yml`
reusable in [mindsdb/github-actions](https://github.com/mindsdb/github-actions)
(`prerelease: true` selects the rc stream). The publish jobs stay in these two
workflows: PyPI trusted publishing matches the OIDC claim on the workflow
filename and does not support reusable workflows.

### Nightly staging integration

The cowork-server maintainers own the deployed integration signal and its
staging prerequisites. [`nightly-staging-integration.yml`](.github/workflows/nightly-staging-integration.yml)
runs every day at 06:41 UTC and can also be dispatched by hand. It calls the
same `tests-integration.yml` reusable workflow as a deployment on `mdb-dev`,
then reports a failure or the first recovery through the shared
engineering-channel notifier. It is a standalone monitor and never gates a
publish, release, or deployment.

**The nightly runs staging's copy of the tests, not main's.** GitHub starts every scheduled run on the default branch, so the checkout would otherwise take `main`. `main` trails `staging` by every commit waiting for the weekly release. Any behavior change among those commits fails `main`'s tests against staging's pods, even though staging works as intended. So the job passes `ref: staging`, and `tests-integration.yml` hands that input to `actions/checkout`. The deploy callers leave `ref` empty and keep testing the commit they just deployed. [`test_nightly_integration_workflow.py`](tests/test_nightly_integration_workflow.py) fails if a scheduled caller's `ref` differs from its `deploy-env`.

The suite may create and delete test conversations, schedules, files, and agent
turns in staging. The fixed test tenant is reserved for the `cowork` suite, and
the workflow sets `COWORK_REQUIRE_INTEGRATION=true` for staging so a missing
target, identity source, replica, or port-forward fails instead of becoming a
green skip.

In the packaged Electron app, a background updater checks PyPI on every launch and upgrades automatically (with rollback on failure). See [`server-updater.ts`](https://github.com/mindsdb/cowork/blob/main/src/main/server-updater.ts) in the frontend repo.

## Architecture

```
cowork/
  api/v1/endpoints/   # FastAPI route handlers
  services/           # Business logic
  models/             # SQLModel / DB models
  schemas/            # Pydantic request/response schemas
  db/                 # Database session and migrations
  common/             # Shared utilities, settings
  harnesses/          # Agent adapters (Anton)
```

The server is designed to be **agent-agnostic** — core features (projects, conversations, files) are shared across agents, while agent-specific behavior lives in harness adapters. See [docs/DESIGN.md](docs/DESIGN.md) for the full architectural rationale.

### Harness system

A **harness** adapts an external agent library (Anton today) to the cowork-server interface. All harnesses implement the `HarnessProvider` protocol (`harnesses/base.py`), which exposes streaming responses, skill sync, and memory operations. The active harness is selected via the `harness` user setting. To add a new agent, implement the protocol and register it with the `@register` decorator.

**Each turn's scratchpads exit when the turn ends.** The Anton harness builds a fresh `ChatSession` for every turn, and each scratchpad that session starts is a child process of the server. When the turn completes, fails or is stopped, `close_session_scratchpads` (`cowork/common/chat_session.py`) kills those processes. A Stop or an opted-in client disconnect also interrupts a running cell and ends its turn; the harness does not start another model step. The next turn starts a new process, which reloads the variables the old one saved to the conversation's namespace snapshot after every cell. Only the scratchpads close: full-stack backends the agent launched keep running. Anything a cell starts in the scratchpad's process group, such as a default `subprocess.Popen` child or a thread, dies with it, so a long-lived service belongs in `launch_backend`. A child started with `start_new_session=True` or `setsid` escapes: it keeps running and holds two of the server's pipe descriptors until it exits. The connector probe closes its scratchpads the same way, and shutdown waits up to 5 seconds for closes still running. A scratchpad runs as `python <tmp>/anton_scratchpad_<random>.py`, so `pgrep -f anton_scratchpad_` counts the live ones; it should drop to 0 once no turn is running.

### Streaming & scheduling

Agent responses stream to clients via **Server-Sent Events** (SSE) on `POST /responses/`. The server tracks in-flight streams and supports cancellation (`/responses/cancel`) and late-join tailing (`/responses/tail`).

A streaming API question can set `"cancel_on_disconnect": true` to stop its own run when the
originating POST connection closes before the answer finishes. The option is a strict boolean,
defaults to `false`, and requires `"stream": true`. Browser questions leave it off, so a reload
still reconnects to the detached answer. Closing a `/responses/tail` observer never cancels a run.
If a client leaves during routing or setup, that request's cleanup prevents it from leaving a
live producer behind. Completion and server shutdown retain their existing outcomes.

The root of `response.created` carries a `run_id` UUID for that execution. An API client or proxy
can send `{"conversation_id": "...", "expected_run_id": "...", "reason": "client_gone"}` to
`POST /api/v1/responses/cancel`. The guard must match the authorized run; a stale or foreign ID
returns 404 and cannot stop a successor. Turn numbers can repeat, so they are not cancellation
guards. Existing Stop calls without `expected_run_id` still cancel the conversation's current
authorized run. A 200 response reports whether cancellation was requested; the stream's terminal
event and persisted answer determine the final outcome. Database or worker failures can prevent
successful finalization.

A proxy that keeps reading upstream after its client leaves must relay the disconnect explicitly.
The self-hosted recording proxy consumes the option, keeps its upstream connection detached for
audit records, and sends an exact-run cancel. Its final record comes from the answer's events.
See [Disconnect ownership and exact-run cancellation](docs/DESIGN.md#disconnect-ownership-and-exact-run-cancellation).

`tests/test_disconnect_transport.py` covers real sockets and a local executing scratchpad.
For the deployed remote-worker check, use the standard post-deploy identity and `COWORK_BASE_URL`,
then set `COWORK_RUN_DISCONNECT_WORKER_TEST=true` and `COWORK_DISCONNECT_NAMESPACE` before running
`uv run pytest -q tests/integration/test_disconnect_worker.py`.
`COWORK_DISCONNECT_CONTEXT` optionally selects the Kubernetes context. The runner needs read
access to pods and permission to exec the observer in the worker container. The check starts
its own bounded cell and observes process/container identity and heartbeat cessation; a
cancelled SSE event alone cannot pass it. It skips unless explicitly enabled.

`/responses/cancel` takes an optional `reason`. The UI sends `"stalled"` when it heard nothing for its idle window, and the turn then saves as a `response.failed` with code `stalled`, so a reload shows the stall card. Without a reason, or with any other value, the cancel is a Stop and keeps the partial answer with no error row. While anton waits on a model call that sends no output, the server (in cloud, the anton pod) writes a `model_wait` progress frame every 20 to 25 seconds so the turn's idle bounds do not end it. A model call that sends no output for 600 seconds ends the turn with code `model_timeout`; anton's `ANTON_MODEL_CALL_IDLE_TIMEOUT_S` sets that limit, and 0 or less turns it off. [Turn liveness and idle bounds](docs/DESIGN.md#turn-liveness-and-idle-bounds) lists every bound a quiet turn meets and what happens when each one fires.

`POST /responses/` reads what it needs (the turn's settings, its attachments, the conversation and the routing gate's history) as one short database unit in a worker thread (`run_db` in `cowork/db/units.py`). The request holds no pooled connection while the gate or the stream runs, and a wait for one never stalls the event loop. When no connection frees within `POOL_TIMEOUT` (5 s by default), the request is refused with `503`, a `Retry-After` header and `{"detail": "Cowork is busy. Try again in about 5 seconds.", "code": "server_busy"}`. An answer that meets the same wait after its stream started ends with one `response.failed` frame with code `server_busy` and the `retry_after` and `retry_at` fields `rate_limited` carries. The question sent again after that frame gets a stream of its own.

An answer produced in this process (`COWORK_TURN_BACKEND=inprocess`, the default) runs its own database work as units too. One at the start saves the question (pending) and loads the conversation. The harness reads the history, the attachments and each `recall_history` lookup in units of its own. One at the end saves the answer and clears the question's pending flag in one commit. Both hold a per-conversation lock that keeps a conversation's message writes in order. So such an answer holds no pooled connection while the model answers or the stream runs. A save that fails at the end of a finished or stopped answer ends the stream with `response.failed` (`server_busy` when no connection freed in time), never `response.completed`. An interrupted or failed answer whose save fails keeps its own `response.failed` frame, and the next boot's recovery seals an interrupted one into history. A Stop, a second Stop or a shutdown that arrives while a turn saves its question or its answer waits for that save, so the stream's last frame always matches what the database holds. The compacted history summary and new artifact index are database units too. The harness waits for its cleanup through `run_to_completion`, so Stop and shutdown cannot abandon these writes, and a full pool never blocks the event loop. When the pool stays full, a Stop waits up to `POOL_TIMEOUT` for each pending write (the answer, a compacted summary and a new artifact's row), so `/responses/cancel` can take up to three times `POOL_TIMEOUT`. A Stop during a running cell returns within 5 s only while the pool frees connections. An answer produced by the remote backend (`COWORK_TURN_BACKEND=remote`) works the same way. Staging the pod's workspace, saving the question, reading what the pod is seeded with, saving the memory and the compacted summary the pod reports, indexing the artifacts it created and saving the answer are each a unit, and the artifacts' owners are looked up in a unit before they are published. So it holds no pooled connection while the pod answers, and its settings come from the snapshot the request loaded. The turn indexes the artifacts the pod created once, however it ends. A Stop waits for that write and then for the answer's save, so with the pool full `/responses/cancel` can take up to twice `POOL_TIMEOUT`. A memory or summary save that is still waiting for a connection when a Stop lands is dropped.

The connector form's stream (`POST /api/v1/connectors/submissions/`) reads its conversation and the settings that build the probe's model client, and saves the probe's narrative, as units too. So it holds no connection while the credential probe waits on the model, and it never waits for one on the event loop. When no connection frees in time for the conversation read or the settings read, the form shows `Cowork is busy. Try again in about 5 seconds.` and the probe does not start. The stream ends at that refusal and saves nothing to the conversation. A save of the probe's narrative that fails still ends the stream with `response.completed`, because the form carries the probe's outcome.

A question sent into a conversation whose streamed turn is still answering is refused with `409` and `{"detail": "Another question is still being answered in this conversation. Wait for it to finish, then send yours again.", "code": "turn_in_progress"}`. It is refused before anything reads or saves it, and the running answer goes on. A turn whose stream has already ended counts as finished, so a question queued behind it is accepted. The check runs before any database read, and `RunRegistry.start` repeats it under a lock for that conversation before it opens the new turn's buffer. Two questions racing through the gate read the same history, so both compute the same turn number, which names the turn's buffer and which boot recovery reads as the question's position in the history. Once a turn saves its question, the registry refuses a later start at that turn's number or an earlier one, even after that turn has ended. So the second of two racing questions is refused with the same `409` whether it reaches the registry before or after the first turn ends, and two questions never share a buffer. A question that follows a turn that saved nothing (refused for a full pool, say) keeps that turn's number and is accepted. A queued follow-up waits for the preceding turn's internal terminal record once its final client frame is being written; slow buffer creation for one conversation does not hold up other conversations. It covers the turns running on the replica that receives the question: with several replicas, a duplicate send that reaches another replica is not refused there.

A turn that fails ends with one `response.failed` frame carrying a stable `code` and the user-facing `error` (`cowork/handlers/turn_errors.py`), and the conversation's saved events keep the same payload, so a reload shows the same card. For `included_allowance_exhausted` and `free_serving_paused` the frame also carries `reset_at`, the instant the free way forward comes back. An in-process turn passes the gate's `X-MindsHub-Reset-At` header through as sent. A hosted turn takes it from the anton worker's `turn_failed` reply, which scratchpad-controller forwards beside `error`. The producer keeps it only for those two codes and only when it parses as an ISO-8601 instant with a UTC offset. An older worker sends none, and its frame has no `reset_at`.

A turn that stops before it finishes, at a limit or another hand-back, still ends with `response.completed`, so every client sees the answer end and keeps its text. Its `response.status` is then `incomplete`, and `response.incomplete_details.reason` says why: `spend_ceiling`, `round_cap`, `handback_budget`, `handback_verifier_failure`, `handback_stuck` or `retry_exhausted` (anton's `ended_by`, described on `IncompleteDetails` in `cowork/schemas/responses.py`). A turn that finished has `status: "completed"` and `incomplete_details: null`. `stream: false` returns the same status and reason in its JSON body.

An answer whose text holds figures gets one `response.figure_sources` event just before `response.completed`. The event lists each figure: its place in the answer (UTF-16 offsets, the unit JavaScript strings index by), its text, its origin (`step`, `question`, `earlier_answer` or `none`) and the steps it came from. Each step it names is listed once, with its kind, a short label, and the project file or attached file, web page, SQL and requests it shows. A cloud pod sees only its conversation's folder and the project's artifacts, so a step there names a file in one of them or an attached file. The event also names the project the turn ran in (`project_id`), which a step's file path is relative to. The formatter (`cowork/harnesses/anton_harness/stream_formatter.py`) finds them in a worker thread once the answer is final, and `figure_sources.py` does the matching. It matches each figure only against this turn's steps: the scratchpad's cells, the results of the web tools anton runs itself (a provider's own web search never reaches the server, so a figure from it has no source), the files `read_text_file` read, other than the agent's memory files in `.anton/memory`, and the results of the connector tools that return the customer's records. Those memory reads, like every other tool, hand back the agent's own text or the user's (an earlier answer from `recall_history`, an artifact, a memory), so their results hold no figure. A cell that reads a memory file never names it as its file, and a cell that makes no request and whose only source would be the memory files it or an earlier cell read is not a step. That rule misses what the cell reader can't see: a cell that also computes a figure from its own numbers or loads data through a call the reader doesn't classify (a library's download, a pickle) is left out, and a memory path built from a name or found by listing the folder reads as any other file. A connector tool is an MCP tool anton registers as `{engine}__{tool}`, or `{engine}__{connection}__{tool}` when the turn has two connections of one engine. anton's `returns_records` (`anton.core.mcp.access`) says which ones return records, such as HubSpot's `search_crm_objects`. Such a step is a query labeled by its engine and tool (`hubspot: search_crm_objects`), with the connection when the name carries one (`hubspot (crm2): search_crm_objects`). Its result is read as the tool returned it, less the ISO date-times in its records (`2026-03-04T09:12:44.512Z`): a record's timestamps are not figures, and the minutes of the created and last-changed times every HubSpot record carries would source almost any small number. A connector call that carries a `sql` argument, as HubSpot's `query_crm_data` does, lists that argument as the step's SQL as the call sent it, through the credential sweep and the length cap (`MAX_QUERY_CHARS`) every step's SQL passes. It is not split into statements or filtered to read statements, as a cell's SQL is, so an argument that also holds a write statement is listed whole. A connector tool that returns docs, schema or account details (HubSpot's `tool_guidance` or `get_properties`), a write tool and a tool of an engine anton has not classified are not steps. Neither is any connector tool when the installed anton has no `returns_records`, as on a PyPI-channel desktop install, which resolves anton from the `anton-agent` floor in `pyproject.toml`. A call of a web tool, `read_text_file` or a connector tool is a step only when its `tool_done` event said `ok=True`, since the turn's tool rows hold a failed call's error as an ordinary result and that error can quote numbers. A cell that prints a figure also names the earlier cells that read the data it was computed from, such as the cell that loaded a file into a frame the printing cell sums. That cell may be an earlier turn's, since each turn's cells pass on the names and data they bound. The formatter reads the earlier turns only when the answer has figures and this turn ran a cell, and then only their last 50 cells. That never changes a figure's origin; past that, a figure can only lose the earlier cell that loaded its data, and its step the SQL or request of a function or string bound that far back. An earlier turn's step never holds a figure itself. A figure that no step of this turn holds and the question doesn't write gets the origin `earlier_answer` when one of the conversation's last 20 answers wrote it in the same form with an origin other than `none`. It names no source, and nothing of that answer is copied. The same form is the same value, exactly, with the same sign, currency sign, scale and percent, and the same unit written right after the digits (`3x`, `40kg`, `12h`). A unit after a space (`12 hours`) and an ISO currency code (`4,600 EUR`) are not part of it, so `12 hours` repeats `12 days`, while `12h` and `12hrs` differ. The turn reads those answers' saved events when it starts, in a database unit of its own (`read_earlier_answers` in `cowork/services/conversations.py`), and only for a conversation the caller owns. An answer with no event (one with no figures, a stopped or failed turn) still takes one of the 20 places. When the read fails, the server logs it and no figure is marked as repeated. The cowork client shows that origin's mark; a client that doesn't know an origin drops that figure, so it shows unmarked. The scratchpad's results arrive as stream events, and the other tools' arrive in the turn's tool rows. A call whose result anton marked `is_error` (one it refused to run, or one an error cut off) is not a step. A figure no step holds counts as the question's when the user typed it, in the question or in an answer to the agent's question during the turn. Context the client appends to the question for the agent (a connect form's state, picked Drive files, attached Notion pages) does not count. The event is saved with the answer's events, so a reload shows the same marks. A direct answer from the routing gate gets the event too, with no steps, so each figure in it is the question's, an earlier answer's or has no source. A non-streaming response carries the same steps and figures as `output[0].figure_sources`. A turn the user stops, a failed turn and a stalled turn get no event, and neither does an answer whose last text is anton's failure reply ("An unexpected error occurred: ..."). A turn that stops early at a limit or another hand-back (`status: "incomplete"`) gets the event as a finished turn does, since its answer is whole. A turn that compacts or hard-truncates its context mid-way still hands the formatter every tool row it wrote, the ones the compaction folded into its summary included, when its anton has `ChatSession.turn_messages()`. In process the server calls it only when the installed anton has it; a cloud pod with it sends those rows in its `history` frame. Its figures are then matched like any turn's. None of those rows are saved: the turn ran near or over the model's context window, so the next turn replays it as text only. A cloud pod or a desktop install whose anton predates `turn_messages` sends no rows after a compaction (PyPI-channel desktop installs resolve anton from the `anton-agent` floor in `pyproject.toml`). When such a turn called `read_text_file` on a file outside `.anton/memory`, a web tool anton runs itself or a connector tool that returns records, and that call's `tool_done` said it succeeded, a figure that no step holds, and that neither the question nor an earlier answer gives, is left out of the event, so it gets no mark rather than "no source". When finding the sources fails, the server logs it and the answer goes out without them.

A step's label, web page, SQL and requests pass through a best-effort credential sweep: `_redacted` in `cowork/harnesses/anton_harness/figure_sources.py`, whose docstring lists every key name and token shape it knows and the forms it still misses, and where a connector with its own secret fields adds them. A secret under a key name the sweep doesn't know, or passed by position, is sent as written, as the step's code in the scratchpad events already shows it. A cell's SQL lists only read statements, each on its own, so a statement after one in the same string is left out; a string that two databases' quoting, comment, bracket or dollar-quote rules would split differently lists no statement, and `cell_queries.py` names the forms the split still misses, such as a `;` inside a bracket before any quote, where Postgres ends a statement, a `'` inside Oracle's `q'[...]'` quoting, and a `;` between backticks, which Postgres reads as an operator character. The sweep masks the password a listed statement sets with `IDENTIFIED BY`, with a quoted value after a secret key's name such as `PASSWORD '...'`, `SECRET '...'` or `SESSION_TOKEN '...'`, or after `REPLACE`, as it does in a string literal that holds one. A request carries its method and URL template, never headers, bodies or query values.

A background **scheduler** loop polls the database every 30 seconds for due schedules, supporting `once`, `hourly`, `daily`, and `weekly` cadences. The poll and each run's reads and writes are database units, so a full pool refuses them after `POOL_TIMEOUT` rather than stalling the server. A run whose first write is refused does not start, and a cron slot then stays due for the next poll. A run's last two writes, its outcome and its finish, try again while the pool stays full, for up to 10 minutes, so a busy moment never leaves a run at `running` or its slot due. Each run creates a conversation and is tracked in `schedule_runs`. Deleting that conversation does not delete the run: the run keeps its status, timings, and error as audit history, and only its link to the conversation is released. A channel binding pinned to the conversation is released the same way, so the external chat stays bound to its project and the next inbound message starts a fresh conversation.

## Data Layer

Data lives in two places: a **SQLite database** for structured records and the **filesystem** for project files and agent workspaces. Understanding both is essential.

> The `~/.cowork` paths below are the default (prod) home. The desktop app runs one of several build channels, each with its own isolated home (`~/.cowork-dev`, `~/.cowork-stable`, etc.) selected via `COWORK_HOME`. See the [Cowork frontend README → Build Channels](https://github.com/mindsdb/cowork#build-channels) for the full mapping.

### SQLite database

- **Location**: `~/.cowork/cowork.db` (override with `DATABASE_URI`)
- **ORM**: [SQLModel](https://sqlmodel.tiangolo.com/) (SQLAlchemy + Pydantic)
- **Migrations**: Alembic (`cowork/db/alembic/versions/`). Startup runs `alembic upgrade head` (singular), so the graph must have exactly ONE head: if two branches each added a migration on the same parent, every fresh boot aborts with "Multiple head revisions". After merging or rebasing, check `alembic heads`; if it prints two revisions, add a no-op merge revision whose `down_revision` is the tuple of both heads (see `f4e2c1a9d3b7` for the pattern).

Key tables:

| Table | Purpose |
|-------|---------|
| `projects` | Project metadata and filesystem path |
| `conversations` | Conversation threads, linked to a project |
| `messages` | Individual messages with role, content (JSON), and harness tag |
| `message_events` | Streaming event payloads for a message |
| `files` | Metadata for uploaded files (path points to filesystem) |
| `schedules` / `schedule_runs` | Recurring prompts and their execution history |
| `settings` | Key-value user settings; sensitive values Fernet-encrypted |
| `pins` | User-pinned items (conversations, artifacts, etc.) |
| `channel_*` | Channel installations, bindings, sessions, and events |
| `shared_resource_attributions` | Creator and latest-editor user IDs for org-shared filesystem resources |
| `shared_resource_mutations` | Append-only actor trail for allowed org-shared mutations, including deletes |

All models use UUID primary keys with auto-tracked `created_at`/`modified_at` timestamps.

### Filesystem storage

```
~/.cowork/
├── cowork.db                       # SQLite database
├── .master_key                     # Fernet encryption key for settings
├── skills/                         # COWORK_SKILLS_DIR — canonical SKILL.md store
│   └── <slug>/SKILL.md             # one folder per skill (see docs/SKILLS.md)
├── projects/                       # COWORK_PROJECTS_DIR
│   ├── general/                    # Default project (always exists)
│   └── <project-name>/
│       ├── <user & agent files>    # Working directory visible to agents
│       ├── skills/                 # symlinks to skills enabled for this project
│       │   └── <slug> -> ~/.cowork/skills/<slug>
│       └── .anton/                 # Private agent workspace
│           ├── artifacts/          # Agent-produced outputs (HTML apps, docs, etc.)
│           │   └── <slug>/
│           │       ├── metadata.json
│           │       └── <files>
│           ├── memory/             # Persistent agent memory by category
│           └── context/            # Project context for agent runs
├── files/                          # COWORK_FILES_DIR — uploaded files
│   └── <file-id>/<filename>
└── data-vault/                     # COWORK_VAULT_DIR — encrypted connector creds
    └── <engine>/<connection-name>/
```

### How the two layers relate

The **database** holds structured metadata and relationships (which messages belong to which conversation, which conversation belongs to which project). The **filesystem** holds the actual content agents work with — project files, artifacts, memory entries, and uploaded documents. The `files` and `projects` DB tables store filesystem paths that point into the directory tree above.

This split is the result of an ongoing migration from a purely filesystem-based architecture. Structured data that benefits from querying and relationships, such as conversations, messages, settings, and schedules, lives in SQLite. Components that are inherently file-based, such as project working directories, agent artifacts, harness-managed memory, local-mode connector vault credentials, and skills, remain on the filesystem by design. Cloud OAuth credentials instead live per user in auth's Data Vault. Skills briefly lived in a DB table. They were moved back to canonical `SKILL.md` files so they can be edited, uploaded, and distributed per project; see [docs/SKILLS.md](docs/SKILLS.md). See [docs/SERVER_MIGRATION.md](docs/SERVER_MIGRATION.md) for the full migration story.

Agents (via their harness) have read/write access to their project's working directory and the private `.anton/` subdirectory. They do **not** access the SQLite database directly — all DB interaction flows through the service layer.

**Settings** use a hybrid approach: user preferences and API keys are stored in the `settings` DB table (with Fernet encryption for secrets), while desktop connector credentials live in the filesystem vault (`data-vault/`). Org-mode OAuth routes proxy the caller's private connection in auth's Data Vault instead.

**The MindsHub credential is the exception, and it is stored nowhere.** On the
desktop it is the user's own session token, which lives ten minutes, so the
Electron app hands it over at runtime through `PUT /api/v1/runtime-credential/minds`
and this process keeps it in memory. `SettingService._raw_data` overlays it onto
the stored rows, so every reader of `get_user_settings()` sees it without
knowing where it came from, and a value handed over beats any stored row.

Two properties follow. **Nothing survives a restart**, so the desktop app
re-pushes on every start of this process. And a key the user supplied by hand
travels the same way, which is what keeps a long-lived `mdb_` key out of both
`.env` and the settings table. The route is loopback-only and refuses in org
mode: an org deployment mints a per-turn credential in the turn producer and its
pods are never handed one.

**A live turn re-reads it per request, with one exception.** The overlay alone
was not enough: a turn copied the credential into its provider once, so a turn
running across a hand-over kept sending the token it started with and took a
gateway 401 that looked like a dead account. `build_llm_client` now passes an
`api_key_provider` into MindsHub-backed providers, so each outbound model call
reads the current in-memory value. Two limits are deliberate. Static
organization-mode and user-supplied keys keep their construction-time value,
because nothing rotates them. And the **scratchpad subprocess keeps the token it
was started with** for the rest of its turn: `export_connection_info()` hands it
a string once and it has no supplier, so a pad-side model call still runs on
that value. Refreshing it needs a pad IPC contract, which ENG-2116 scoped out.

`build_llm_client` capability-gates the kwarg on `inspect.signature`, so an
older Anton keeps the static key and logs the degradation rather than failing
every turn.

## API

All endpoints live under `/api/v1/`. Key resource groups:

| Path | Description |
|------|-------------|
| `/health` | Readiness probe |
| `/projects` | Project CRUD and working-folder management |
| `/conversations` | Conversation threads and message history |
| `/responses` | Streaming agent responses (SSE) |
| `/files` | OpenAI-compatible file uploads |
| `/schedules` | Recurring task scheduling |
| `/skills` | Agent skill definitions |
| `/memory` | Persistent agent memory |
| `/artifacts` | Agent-produced file previews |
| `/publish` | Publish artifacts to MindsHub (view.mindshub.ai in prod) |
| `/connectors` | Third-party service connections and OAuth |
| `/settings` | User preferences and API keys |
| `/runtime-credential` | Desktop hand-over of the MindsHub credential (write-only, loopback, local mode) |
| `/hub/workspaces` | Which MindsHub workspace this person is working in |
| `/hub/usage` | The caller's free Air allowance (as a proportion), balance, auto top up and credit spend, for the desktop's usage warnings. An organization with no free grant reads `freeTokens.limit` 0 and `freeTokens.resetsAt` null, since nothing refills |

### Declaring who may call a route

Every route declares its permission at its own definition, and `create_app()`
refuses to start if one does not (ENG-2094). Adding a route means picking the
class that names the credential the route actually takes:

| Declaration | The credential is |
|-------------|-------------------|
| `OpenByDesign` | nothing at all. Pinned route by route, with a written reason, in `tests/test_open_by_design_pin.py` |
| `LoopbackOnly` | the caller being on this machine (`require_local`) |
| `DesktopOnly` | not org mode (`require_local_tenancy` 403s the route there) |
| `LoopbackDesktopOnly` | both of the above |
| `PlatformSignature` | the calling platform's HMAC over the body (channel webhooks) |
| `Authenticated` | a verified `Principal` |
| `AuthenticatedInOrgMode` | a verified `Principal`, in org mode only; a no-op on desktop |
| `AuthenticatedOrgAdmin` | ... plus org-admin standing |

```python
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require

@router.get("/thing", dependencies=[Depends(require(AuthenticatedInOrgMode))])
```

Declare it on the `APIRouter` when every route on it takes the same
credential, with one exception: never declare `OpenByDesign` on a router.
FastAPI *adds* a route-level `dependencies=[...]` to its router's rather than
replacing it, so a router carrying the open marker hands it to every route
added later, and the walker cannot tell that apart from a deliberate choice.
Every other class refuses, so inheriting one is safe.

`COWORK_TENANCY_MODE=org python -m scripts.dump_routes` prints the current
surface with each route's declaration.

### The MindsHub workspace selector

`/api/v1/hub/workspaces` backs the workspace selector at the bottom of Cowork's
sidebar, which Cowork draws only when the listing carries two or more
workspaces: one workspace means nowhere to move to. A **MindsHub Workspace** is an org-internal container that owns hub
resources (API keys, artifacts, model entitlements) and lives in the auth
service. It has nothing to do with the filesystem directories this repo calls
workspaces, which is why the stored key is `hub_workspace_id`.

The selector used to be switched by auth's `authorization_ui` Statsig gate,
read out of the entitlements payload: no local Statsig client, one gate
governing the console and Cowork alike. It was retired once its surfaces had
been live in production long enough to trust (ENG-2043), so the selector is
unconditionally on now and this service no longer calls entitlements at all.

Four things about it are worth knowing before changing it.

**The sidecar makes the call, not the renderer.** Auth's ingress allows three
console origins per environment and no Cowork host, and a per-PR Cowork host
cannot be added to a static allow-list because its name carries the PR number.
The packaged Electron app would get away with a direct call because it runs with
`webSecurity: false`; the web SPA would not. Reading here works for both shells.
The outbound host is derived by `default_minds_auth_host()` from `ENV`, which the
desktop propagates when it spawns this process, and the credential is the
caller's own, never the stored provider key and never `minds_url`, both of which
an org admin can set.

**The credential arrives in its own header, `X-MindsHub-Authorization`.** It
cannot use `Authorization`: Electron's main process overwrites that on every
request to the loopback server with the server's own token, so the caller's
Keycloak JWT can never arrive under that name in the desktop shell.
`Authorization` is still the fallback **in org mode only**, where the ingress put
the caller's JWT there; on a desktop install that header holds this server's own
bearer, and forwarding it to auth would leak the one credential the main process
scopes to the loopback origin. `hub_credential` reads both, and it is
deliberately a different function from `caller_bearer` so a client cannot steer
the credential on the org model-catalog fetch by setting a header.

**The listing cache is keyed on the credential, not just the caller.** Auth
answers the listing per caller: an owner or admin sees every workspace in the
organization, a member only the ones they hold a grant on. So an
organization-keyed cache served one admin's menu to every member for the whole
TTL, and the grant check on `PUT /active` reads the same entry. The key is
`(auth host, organization, user, credential digest)`. The digest is not
belt-and-braces: `user_id` comes from the gateway-set principal and is `None` on
every desktop request, because `scope_from_principal` returns `LOCAL_SCOPE`
outside org mode, so identity alone collapses to one shared entry and a
sign-out/sign-in as another account would be served the previous one's
workspaces. A new session means a new token means a new entry. Entries are swept
on write, since nothing re-reads a departed caller's key and the dict would
otherwise grow for the process lifetime. The TTL follows whether **auth
answered**, not what it said: a reachable listing keeps the long TTL, and an
unreachable one the short failure budget, so a degraded auth does not add a
round trip to every menu open without also going stale for a long stretch.

**Two refusals on `PUT /active`, and neither may read the stored pick.** A
workspace missing from the caller's listing is a 403; one in the listing but
stamped archived is a 409, so the client can say retrying will not help instead
of offering a loop with no exit. The archived check reads the target row's own
`archived_at` rather than asking whether it is in the set the menu offered, and
that distinction is load-bearing. `hub_workspace_id` is an untagged
`UserSettings` field, so `PUT /api/v1/settings/hub_workspace_id` writes it with
no gate and no listing check, and `selectable` keeps the active row even when
archived. A refusal phrased against the offered set would therefore have been
talked into accepting an archived workspace by one call to the settings route.
Nothing that refuses a request may read a value any caller can write.

**Picking a workspace changes what the client shows, not what a turn is billed
to.** Neither turn credential carries a workspace: a desktop turn presents the
user's own session credential, whose organization comes from the token's
active-organization claim, and a cloud turn presents a minted key whose request
body has no workspace field. So nothing on the turn path reads
`hub_workspace_id`, and a test asserts that.

The pick is stored as an untagged `UserSettings` field, so it lands per `(org,
user)` in org mode and in the single global row on a desktop install. That is
interim: the shared per-user preference the console reads has no route in auth
yet, and a follow-up migrates this onto it.

### Who can read what in org mode

Local mode has one user, so none of this applies: every check below is inert on
the desktop.

In org mode a request acts as a pair, an organization and a user, and it never
carries less. The gateway authenticates the credential, asks the auth service
whether that user is still a member of that organization, and injects
`X-User-Id` and `X-Organization-Id`. cowork-server validates the shape of those
headers and then trusts them, so **the gateway being the only route to the pod is
what makes them trustworthy**, and that is a NetworkPolicy rather than
anything in this codebase. `deployment/cowork-server/templates/network-policy.yaml`
is that policy, on in staging and prod, and it admits only the nginx ingress
controller pods in the `infrastructure` namespace on port 9010. It is off in PR
environments, which take base values, so a PR environment does not enforce this
boundary and a forged-header caller inside one is served.
A request with no valid pair is answered 401 before any route runs, except on
`/api/v1/health/`, which the kubelet probes with no headers, and the channel
webhook paths, which third parties call.

### How the browser stays in one organization

Canonical Cowork web sends `X-Cowork-Expected-Organization-Id` with every
authenticated browser API request.
`TrustedHeaderMiddleware._organization_boundary_response` compares it with the
normalized `X-Organization-Id` supplied by the auth gateway before the route
runs. It applies only to Keycloak-shaped bearer JWTs. MindsDB API keys, opaque
service credentials, requests without a bearer, CORS preflights, health checks,
and channel webhooks keep their existing behavior.

The boundary always refuses. A missing header returns 426 and a malformed or
changed value returns 409. Both responses carry
`X-Cowork-Organization-Reload: required`, a JSON `code` and `detail`, and
`Cache-Control: no-store`. The browser reloads instead of letting an old
document continue under a new Keycloak organization. A request already inside a
route keeps the `Principal` created at its start, so a concurrent session change
cannot retarget that in-flight operation.

There is no setting that relaxes this. The boundary once had an `audit` position
that logged a violation and served the request anyway; it and the staged rollout
it existed for are gone. An environment that still carries the retired
`COWORK_ORGANIZATION_BOUNDARY_MODE` key boots normally and ignores it. Deleting
the field is what makes it unreadable: `AppSettings` sets no `env_prefix` and
gives every field an explicit `validation_alias`, so the environment source only
looks up names a field still claims. `extra="ignore"` only decides what happens
to an unknown key arriving in a `.env` file, which is not how the overlays
deliver this one. Both halves are pinned, the settings half by
`tests/test_app_settings.py::test_stale_organization_boundary_mode_env_var_is_inert`
and the request half by
`tests/test_principal.py::test_no_environment_variable_reopens_the_organization_boundary`.

`GET /api/v1/capabilities/organization-switch` is authenticated and returns
protocol version 1. It reports `expectedOrganizationEnforced: true` when org
tenancy and identity enforcement are both active. Identity enforcement is in
that answer because the picker should stay hidden while identity is only
audited, not because it gates the boundary: `TrustedHeaderMiddleware` runs the
boundary for every browser JWT request it builds a `Principal` for, under org
tenancy alone. It reports `enabled: true` when those hold and
`COWORK_ORGANIZATION_SWITCH_ENABLED=true`.

`COWORK_ORGANIZATION_SWITCH_ENABLED` is the product enable for the picker, not a
safety switch, and it is the lever to reach for to hide the picker without a
code change. `COWORK_IDENTITY_ENFORCE=audit` hides it too, by dropping
`expectedOrganizationEnforced`, but it reopens the no-principal path and leaves
the boundary refusing anyway.

Backing enforcement out needs the code and values from before cowork-server#524.
[deployment/cowork-server/README.md](deployment/cowork-server/README.md#back-out-enforcement)
records the procedure and why a Helm rollback cannot do it.

Inside one organization, two different rules apply, and which one you get
depends on the resource:

- **Shared with the organization:** projects, project files at the project root,
  skills, and project memory. Every member reads them.
- **Private to whoever created it:** conversations and their history, scheduled
  tasks, personal memory, uploaded files, and everything under a conversation's
  own workspace at `conversations/<conversation_id>/`. Live artifacts are in
  that last group, because the agent writes them into the conversation it is
  running in. Cloud OAuth connections are also user-private, held in auth's
  Data Vault rather than here. cowork-server stores none of it and relays the
  caller's own credential to auth for the whole lifecycle: the connect
  handshake and its status poll, the connection catalogue, one connection's
  detail, the Google Picker file grant, a short-lived picker access token, and
  disconnect. auth resolves the caller from that credential, so each route only
  ever reaches that user's own connections.

Shared visibility does not imply shared destructive access. Project creators
and organization admins can rename or delete a project; the General project is
immutable. Any member can create a skill, while only its creator or an admin can
edit, disable, or delete it. Packaged skills are immutable in org mode for every
role (desktop keeps its existing editable-copy behavior). The first member to
write a non-empty project-memory slot becomes its author, and only that author
or an admin can subsequently change it. Project instructions follow the project
creator/admin rule. Empty memory writes do not claim a slot, and ordinary
project files remain member-writeable.

The API derives these decisions from the trusted principal and returns typed
`attribution` and `capabilities` objects for the client. Stable user IDs drive
authorization, and they are the only identity these rows keep: no email address
is stored, and the API fills one in only when the actor is the viewer
themselves, so an attribution never discloses another member's address.
File-backed resource ownership lives in `shared_resource_attributions`, outside
the agent-writeable project tree, and every allowed protected mutation appends a
`shared_resource_mutations` row. A delete removes the current attribution but
keeps its mutation event. Unattributed legacy resources fail closed to
admin-only mutation.

The private rule is enforced by the service layer rather than by the routes:
`ConversationService._owned`, `FileService._owned_select` and
`ScheduleService._owned_select` each add `created_by == <the caller>` when a
request is org-scoped. Two places extend the same rule to the filesystem.
`_conversation_workspace_ok` in the project-file routes refuses a path under
another member's conversation directory, and `artifact_roots` drops another
member's conversation directories before the artifact list or delete ever sees
them, because those routes are addressed by project and slug and never receive a
conversation id.

One artifact at a time can leave the private group, and only its owner can put it
there. `POST /artifacts/workspace/{project}/{artifact}/comments-access` records a
grant in that artifact's `.revisions/draft-review.json` and mints the matching
rule in auth. A co-member then resolves that one artifact by id —
`artifact_scope.review_artifact_for_request` searches other members' workspaces
only after the caller's own, and only accepts a folder that carries the grant —
and gets the draft preview and comments, never the source, the edit routes or the
delete. Without a grant the answer is 404, so a private draft still cannot be
told apart from one that does not exist; with a grant the owner-only routes
answer 403 instead, because to a client already looking at the draft a 404 would
read as deleted. The artifacts list is unaffected either way: a co-member's
artifact never appears in it, so review starts from the link the owner shares.

Org-mode artifact authorization uses a separate server-owned identity. An agent
can edit `metadata.json`, so its local UUID cannot reserve a global comment key.
The `artifact_identities` SQL table binds each `(organization, local UUID)` to
its immutable owner and an auth-issued canonical key. Draft grants, publishing,
and policy deletion use that key; the comments REST/SSE proxy translates the
unchanged local UI key through the same table. Read requests never allocate.
Existing identities are adopted only when auth confirms a matching durable
owner and organization binding. New issuance uses a persisted random request ID
so concurrent calls and lost replies cannot allocate different identities.

This requires auth's internal `artifact-access/claim/` and `allocate/` endpoints
and the matching services publisher ownership checks. Apply auth's ownership
history migration before enabling new policy writes, then deploy these clients
with the coordinated ENG-2262 changes. Authorization failures preserve files
and stop sharing. SQL aliases survive artifact deletion and schema downgrades;
they are ownership history, not a cache that can be cleared during rollback.

Filesystem authorization decisions start with a resolved path. The route carries
that decision through opening the file: every
component below the project directory is opened `O_NOFOLLOW`, and a symlink
planted anywhere in the chain is refused. A pod mounts its own workspace
read-write, so without that a swapped directory component between the check and
the open reaches another member's tree, or another organization's.

**A refusal on a private resource answers 404, not 403**, with the same body a
genuine miss returns. Telling the two apart would confirm that another member's
file exists, which is most of what an attacker wants to know. Policy refusals
that reveal nothing personal answer 403 instead: the desktop-only routes say
`not available in org deployments`, and an organization-settings write without
the admin role says so plainly.

**The HTML preview hands out a bearer token in a URL** (`preview-mount-file`
returns one, `preview-asset` spends it), because an iframe cannot send an
`Authorization` header. The token is random, it is bound to the member and
organization that minted it, and it expires after 30 minutes, so a token that
escapes into a log or a screenshot is not a way in for anyone else.

Being the minter is not enough on its own, because **a mount grants a
directory**. The gate runs on the file the caller named, and an `.html` at the
project root sits in a directory every member's `conversations/<id>/` hangs off,
so a token minted on a shared file would otherwise read every workspace under it.
A mount therefore reaches only the workspace its own file lived in, and a mount
taken on a shared file reaches no workspace at all, not even the minter's own.
That check holds no session, deliberately: `preview-asset` serves every sub-asset
a page pulls, and a session there is a database connection per image.

### The default model is the one the free allowance covers

Every minds-cloud role defaults to `mindshub_air`, for all three roles: planning,
coding and router. **MindsHub's catalog declares it**, in the
`mindshub_model_policy_v1` config that already owns the alias registry, and the
declaration arrives as a `default_for` list on each `/v1/models` row. So moving a
default is a config edit plus an apply, not a release. MindsHub Air's usage draws the included allowance, so a user who has
picked no model can finish a whole turn without the wallet being charged for any
part of it.

The two roles a user never sees are why this is the default rather than a
premium model. Planning is the model in the picker, so a wrong choice there is
visible and fixable. Coding (the completion verifier and the scratchpad) and
router (respond-versus-delegate gating and history summarization) run unseen, so
a paid default there is denied on an empty wallet with nothing on screen to
explain why.

An explicitly stored model is never rewritten by this. Paying for a better model
is a pick in the Settings picker, and a funded wallet resolves to the same
default as an empty one until that pick is made.

### A stored model the wallet cannot pay for is swapped, not rewritten

`/v1/models` marks a model the org cannot currently pay for as `enabled: false`,
and the map is cached as `minds_model_enabled`. When a stored pin is flagged
that way, `_resolved_model` resolves the role to the first affordable model in
the map instead, for all three roles. The alternative is every turn failing on a
denial the user may not be able to see.

The stored row is left exactly as the user set it, which is the load-bearing
half: the moment the wallet can pay again, the next settings load flips the
alias back to `enabled: true` and the role resolves to the original pick with
nothing to re-select.

Two cases share that path and should not be confused. A pin the map flags
`false` is a real MindsHub model that is merely unaffordable right now. A pin
**absent** from a non-empty map is foreign or retired, so it would 404 on every
turn rather than 402, and it is healed the same way with no route back.

`enabled: false` has one more cause: an admin in the organization restricted the model. `/v1/models` names the lock on each disabled row in `disabled_reason` (`model_restricted`, `wallet_empty` or `included_allowance_exhausted`), and `GET /settings/recommended-models` relays it as `modelDisabledReasons`, so the picker shows a restricted model as restricted and offers no credits for it. The map comes from the MindsHub listing only. A custom openai-compatible endpoint's `disabled_reason` is dropped, because a BYO endpoint must not be able to claim an admin restricted a model. Resolution treats a restricted pin like any other `false` one and swaps it, for the members the rule restricts and no one else. A turn that still reaches the gateway on a restricted model fails with the `model_restricted` code: the gateway keeps the 403's `permission_denied` reason and names the rule on its `X-MindsHub-Deny-Detail` header and in the body's `error.deny_detail`, trusted by the same origin rules as the reason.

A restriction belongs to one member, so it is stored per member. Auth applies the model rules for each caller's org, workspace and team, so two members of one org get different listings, and `fetch_org_model_catalog` caches one listing per org and member. `minds_model_enabled` is an org row that every member's resolution reads. So in org mode `recommended_models` leaves the caller's `model_restricted` models out of it: each keeps the flag the org map already held for it, or `true` when it held none. The caller's restricted ids go to their own `minds_model_restricted` row instead, an untagged setting that `SettingService` files at the caller's user scope, and `UserSettings._minds_enabled_map` reads them as `false` over the org map. One member's restriction therefore never swaps another member off a model, and another member's load never lifts it. A member's restrictions reach resolution when that member's own settings load writes them. A listing that publishes no `enabled` flags leaves the stored list alone, the same evidence rule the org map follows. A desktop install has one member, so it keeps restrictions in `minds_model_enabled`.

The listing cache (`_minds_models_cache` in `cowork/services/providers.py`) drops expired entries on every write and holds at most `_MINDS_MODELS_CACHE_MAX` listings, evicting the least recently used, so keeping one entry per member does not grow it for the life of the pod. An entry with no organization in its key, which includes every desktop fetch, is keyed by the credential it was fetched with, held as a pydantic `SecretStr` so it prints as asterisks in any repr or log, so an account switch in the running app is never served the previous account's restricted or locked rows. Nothing is hashed from it.

The desktop closes the loop at the other end: a model the map locks is not
offered in either picker, so a swap only ever applies to a pin that was
affordable when it was made. Allowing the pick meant the turn silently ran a
different model from the one the picker named.

#### Where the answer comes from, in order

`MODEL_ROLE_DEFAULTS` in `cowork/common/settings/app_settings.py` is still a real
answer, not a legacy layer. Resolution is synchronous and does no network call in
the turn path, so the catalog's declaration has to be *persisted* before it can be
read, exactly as the availability map is.

| State | What resolves | When you are in it |
| :---- | :---- | :---- |
| A model is stored for the role | that model | the user picked one, or saved Settings once |
| `minds_role_defaults` names the role | the alias the catalog declares | any install that has loaded Settings since the catalog declared it |
| Nothing persisted | `MODEL_ROLE_DEFAULTS` | a fresh install sending its first message, and any install that has never reached the catalog |

The availability map still overrides the answer in every case where the wallet
cannot pay for it, so a declared default is not a grant: it says where to start,
never what may be called.

`GET /settings/recommended-models` writes the map, from the same fetch that
already refreshes `minds_model_enabled`, and it builds the `recommendedPair` it
serves the picker through `minds_role_start_models`, the same two steps resolution
takes: the declared default replaces the compiled one, then availability adjusts
it. Those two have to agree, because the picker shows the pair as the model each
role starts on and the desktop writes it back as an explicit pin when a save
repoints a role onto MindsHub. A pair built from the compiled table, or from a
declared default the wallet cannot pay for, would pin a model that turns never
run. The pair is rebuilt from whichever map resolution will read: the live one
when the gateway published defaults, the cached one when it did not.

### Provider probes always use a model any key can call

Why a probe sends a model at all: MindsHub bills per model, so a model the wallet
cannot pay for is denied, and that denial is indistinguishable from a bad key.
Probing a paid model tells an account with an empty wallet that its working key is
invalid. `MINDS_PROBE_MODEL` (`mindshub_air`) draws the included allowance
instead of the wallet, so the result reports reachability and key validity, which
is what these endpoints are for.

Two endpoints, and they do not behave identically.

`POST /settings/validate-provider` (onboarding, and the only caller is the
onboarding screen) probes a chat completion on every branch, and takes an optional
`model`:

- `provider: "minds"` always sends `MINDS_PROBE_MODEL` and **ignores** `model`.
- `provider: "openai-compatible"` sends `model` as asked, so validating one
  specific model never reports a pass earned by a different one. Omit it against a
  MindsHub base URL and it falls back to `MINDS_PROBE_MODEL`; omit it against any
  other host and the generic openai-compatible default applies.
- `provider: "anthropic"` sends `model` or `claude-sonnet-4-6`.

`POST /settings/test-providers` (the Settings health dot) probes **per provider
type**, and only the `minds-cloud` type is a chat completion, on
`MINDS_PROBE_MODEL`. The `openai-compatible` type is a `GET {baseUrl}/models`
listing probe, so a MindsHub host configured through that card is health-checked
against a route MindsHub does not deploy everywhere; those routes answer 404 or 401
even for a valid key, which is the reason the `minds-cloud` type does not use one.

An `apiKey` of `""` or `"***"` on that route means "use the stored one", which is
how the Settings UI re-tests a provider it only ever received masked. A stored key
only goes to a host this deployment saved: the provider cards in `providers_json`,
the scalar `openai_base_url`/`minds_url`, and the vendor host an omitted `mindsUrl`
already reaches. The comparison is on origin (`scheme://host[:port]`), not the whole
URL, because the question is which party receives the key; that also means a stored
URL carrying a path still matches a body sending the host alone. A URL the guard
cannot parse names no origin, so it is refused rather than raising. Omitting the URL
stays legal: a `minds-cloud` probe then goes to the vendor host, and an
`openai-compatible` one answers `missing base URL`, which is what it did before this
guard. A key the caller supplied may go anywhere, since nothing stored is at risk.

A refused provider comes back `fail` with its reason and is never pinged, rather
than failing the whole request. Callers send every configured provider in one call,
so refusing the request would blank the other providers' dots over one bad URL.

Each card in the body is a `ProviderProbeCard` (`cowork/schemas/settings.py`), which declares the four fields the ping reads, `type`, `apiKey`, `baseUrl` and `mindsUrl`, as strings. A card whose `type` or URL is not a string fails the whole request with a 422 before any key is resolved. The Settings UI does not hit this: it sends the cards it saved itself, with `type` normalized to a string by `backfillProviders` in cowork's `settingsTransform.js`. Every other field the UI holds on a card, such as `isDefault` or `name`, is ignored, so a renderer that adds one keeps getting its status dots.

When the MindsHub gateway refuses a `minds-cloud` probe with a named reason, the response carries it in `providerStatusReasons`, keyed by provider type: `{"minds-cloud": {"code": "free_air_daily_spend_fuse_exceeded", "resetAt": "<ISO instant or null>"}}`. The code is the gateway's own `X-MindsHub-Reason` word, one of `wallet_empty`, `included_allowance_exhausted`, `free_air_daily_spend_fuse_exceeded`, `rate_limited` and `policy_unavailable`, so the Settings notice can tell a velocity limit or the free-Air fuse from an empty wallet instead of matching on `HTTP 429` in the detail string. When a proxy strips that header, the probe reads the same word from the body `code`, which the gateway sets to match. `resetAt` carries the gate's `X-MindsHub-Reset-At`, which it sends on the allowance and fuse denials. The body `code` counts only when the probe's response came from the configured `minds_url` host. The header counts unless the response provably came from another host. These are the origin rules the turn-failure mapping applies (`gateway_denial` in `cowork/handlers/turn_errors.py`), so a probe aimed at any other host names no reason. A refused provider was never pinged, so it has no entry. `providerStatus` and `providerStatusDetails` keep their shape.

Every MindsHub-bound chat probe caps the completion at `max_tokens: 20`, not 1:
some models refuse a 1-token budget and fail the probe for a perfectly good key
(see `_chat_probe`). The cap is not sent to a non-MindsHub endpoint, because
OpenAI's reasoning models reject `max_tokens` and want `max_completion_tokens`.

The desktop app has a second copy of these validators in its Electron main process
(`cowork/src/main/provider-validation.ts`, called from the `settings:validate` IPC
handler in `cowork/src/main/index.ts`); the endpoints here serve the web build.
Both copies have to change together. One asymmetry worth knowing: the desktop
MindsHub onboarding path signs in through Keycloak rather than validating a pasted
key, so main's `validateMinds` has no live caller today, and it is the
openai-compatible and anthropic validators there that a packaged build actually
runs.

### Organization permission enforcement

Hosted turn admission checks `product.execute` through auth's internal
`POST /internal/permissions/authorize/` endpoint. This applies to free and paid
models, stored provider credentials, direct responses, remote queue submission
and scheduled runs. Schedules resolve the acting identity from the stored owner
and check it before creating a conversation. Queue submission rechecks current
access even when reusing a previously minted credential.

The request contains the server-resolved user and organization IDs and one
permission. It uses the existing `COWORK_TURN_AUTH_INTERNAL_BASE_URL` and
`COWORK_TURN_AUTH_INTERNAL_SECRET`; customer headers cannot choose that host or
credential. A confirmed denial returns `403` with `permission_denied`. Missing
configuration, malformed replies, transport errors and service authentication
failures return `503` with `permission_unavailable`. Authorization failures never
fall through to another model or delegated execution. Local desktop mode retains
its single-user behavior.

Artifact source edits, publishing/access changes, deletion, revision restoration
and repair management require `artifact.manage` as well as existing artifact
ownership. Starting an agent repair also requires `product.execute`. Capability
responses reflect these current grants. Generic project-file writes and deletes
apply the artifact grant to artifact storage paths too. Writes atomically replace
the selected directory entry, preserving the existing file mode where descriptor
chmod is available. This detaches a pre-existing hardlink so editing an ordinary
file cannot modify protected artifact bytes through the same inode. Failed writes
leave the original file intact and remove the temporary file.

Remote turns use the `anton_turn_v2` controller operation and declare their
workspace authority. The controller checks `artifact.manage` again when it
dequeues the turn. Without that grant, it mounts saved conversation files read
only and runs the agent in a temporary copy. All workspace edits in that turn
are temporary, including ordinary files; nothing is copied back or published.
The controller replaces warm workers when their storage authority is unsuitable.
Cowork requires the controller's verified workspace acknowledgement before
accepting worker output, and indexes artifacts only for persistent workspaces.
Deploy the companion scratchpad-controller change before enabling this producer;
older controllers fail these turns with `permission_unavailable`.

Publishing uses a separate `artifact_publish` mint purpose. It requires artifact
management without granting model execution. Execution mints retain their own
purpose and credential type; the publisher uses a fresh instance ID and never
hands its credential to an inference client. Deploy auth's decision/purpose APIs
and the publisher's artifact-only authentication path before this server change.
Keep these admission checks in place during rollback while restrictive custom
roles remain assigned. No new customer or staff permission grants are introduced.

### Datasource grants on cloud turns

`COWORK_TURN_DATASOURCE_ENABLED` (default `false`) lets a hosted turn query the
user's verified PostgreSQL and MySQL connections in the organization. When it
is on, the producer lists them from auth under the dedicated
producer service role (`COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID` and
`COWORK_TURN_DATASOURCE_PRODUCER_KEY`), bound to the turn key it has just
minted, registers one grant per connection before enqueueing, and puts only
connection ids and credential versions on the queue. No capability or
password crosses Redis; the pod reaches the gateway on the inference host the
turn's `llm` block already names. A listing or registration failure fails the
dispatch as `permission_unavailable` instead of silently dropping the
datasources.

Turn the flag on only after auth serves the datasource producer and resolver
endpoints, the inference deployment serves `/v1/datasources/`, the
scratchpad-controller passes the `datasource` block on, and the scratchpad image
includes the typed helper that reads the gateway from the turn's inference host,
with live pods from an older image recycled first. With the flag off nothing
in this path runs, so the flag is also the rollback.

### Sharing one skill with everyone

On a deployment that trusts identity headers and keeps `skills` in `COWORK_PERSONAL_RESOURCES`, a person shares one of their own skills with everyone, read-only, or hides it again.

- `PUT /api/v1/skills/{skill_id}/sharing` takes `{"shared": true}` or `{"shared": false}` and returns the skill. Both directions are idempotent.
- It answers 401 to a caller with no person: the desktop, or the operator view under `COWORK_IDENTITY_ENFORCE=audit`.
- It answers 409 "Sharing one skill at a time isn't available here." where sharing is unavailable: hosted org tenancy, and skills left off `COWORK_PERSONAL_RESOURCES`.
- It answers 403 "This skill is shared with you and can't be changed here." for a managed skill's name or a skill someone else shares with the caller, and 404 when the caller has no skill by that name.
- A share answers 409 with the reason for a packaged built-in, a name the agent already has from the harness's own skills or anton's packaged ones, a managed skill's name, a skill that can't be read in full, a skill holding a link, and a name someone else already shares a skill under.
- Any change answers 503 "Sharing is busy right now. Try again in a moment." while another change holds the list for 5 seconds, and while the filesystem refuses to read the list.

Each skill's `capabilities.canShare` is true for the caller's own skill where sharing is available, and false for a packaged built-in's copy, a managed skill and a skill shared with the caller. The skills list's top-level `canShare` says whether the caller may share here at all; the UI asks "who can use it" on create only when it is true. Both are false on the desktop, in hosted org tenancy, where skills are left shared and for the operator view.

`POST /api/v1/skills/` takes `"shared": true`, and `POST /api/v1/skills/upload` the multipart field `shared`, to share the new skill as it is created. Both default to false. A refused share answers 409 (503 when busy) and writes nothing; when the skill is written but its entry then can't be added, the new skill is removed and the error stands.

Deleting a shared skill stops sharing it. Renaming one answers 403 "Stop sharing this skill before you rename it.", because the list names a skill by its slug. A create, upload or rename into a name the person's list still has for a skill of theirs that is gone removes that entry first, so the new skill is shared only when they choose it. Once the list exists, every create, upload and rename into a new name takes the list's lock for that check: it answers 503 "Sharing is busy right now. Try again in a moment." while another change holds the list or while the filesystem refuses to read it, and writes nothing. A delete still deletes the skill then, and logs a warning naming the list it couldn't update.

Everyone else lists a shared skill after the packaged built-ins, the managed skills and their own, live from its creator's store, with `isBuiltin` false, `shared` true and no capability. A same-named skill earlier in that order hides it, and so does one the agent already has, and a listed skill that no longer exists is skipped; for a name two people share, the first entry wins. Their turns link it from the creator's store into a read-only skill root after the managed skills, with the turn's selection rule (enabled, and scoped to no project or to this project's name) applied to the skill's own metadata. Everyone else's edit, disable, rename or delete of it answers 403 "This skill is shared with you and can't be changed here."; disabling is refused too, because `enabled` lives in the skill's own `SKILL.md`. Everyone else's create, upload or rename into its name answers 409 "Someone else shares a skill named '<slug>'. Choose another name." Its creator sees it `shared` with every capability and edits, disables and deletes it as before, and everyone else sees each change.

The list is `<COWORK_SHARED_DIR>/shared/skills.json`, a JSON list of `{user_id, partition_id, skill}` in the order shared (`cowork.shared_skills`). Cowork writes it itself, under an `flock` on `skills.json.lock` beside it, replacing the file at once with mode 0644. A missing list shares nothing. A malformed one shares nothing too: every read logs a warning naming the file, the server still starts, and the next change that takes the list's lock replaces it, starting from empty: a share or hide, and a create, upload, rename or delete. A list larger than 1 MiB counts as malformed. A list the filesystem refuses to read (EIO, EMFILE, EACCES), or one that isn't a regular file (a FIFO, device, directory or symlink left in its place), is not malformed, so no change replaces it: while it lasts, readers share nothing, the server still starts, and every change logs an error naming the file, answers busy and writes nothing until an operator removes it. `python -m cowork.ops.skills share` and `unshare` change it on a person's behalf ([Operator commands](#operator-commands)).

## Configuration

Configuration is read from the database (`UserSettings` table) and can be managed through the Settings UI in the desktop app or via `PUT /api/v1/settings/`.

The `favicon` setting stores a browser tab icon as a data URI, separately from
the sidebar's `nav_logo`. It defaults to an empty string, which keeps the
page's original icon. Saving an empty string restores that default. Like the sidebar
logo, this is a personal preference scoped to each member within an
organization; local tenancy uses the instance's shared setting. `GET /api/v1/settings/`
returns the data URI without masking it.

The `show_figure_sources` setting (on by default) is read only by the UI, which marks each figure in an answer with the steps it came from while it is on. The server finds and saves every answer's `response.figure_sources` whatever the setting, so turning it back on shows the marks on earlier answers too. Like `favicon`, it is a personal preference in an organization, and local tenancy shares one setting for the instance.

Environment variables fall into these groups:

**Server-level** (`COWORK_*`) — control the cowork-server process itself. Each one also reaches the desktop sidecar through `<COWORK_HOME>/.env`, which the server reads below the process environment, so a value written there turns a setting on for the desktop too:

| Variable | Default | Description |
|----------|---------|-------------|
| `COWORK_LISTEN_PORT` | `26866` | Server port |
| `COWORK_SERVER_HOST` | `127.0.0.1` | Bind address |
| `COWORK_TENANCY_MODE` | `local` | `local` is the desktop sidecar: one user, no organization, no identity headers. `org` is the cloud deployment and turns on everything in "Who can read what in org mode" above. |
| `COWORK_IDENTITY_ENFORCE` | `enforce` | Org mode, and local tenancy with `COWORK_TRUST_IDENTITY_HEADERS` on. `enforce` answers 401 to a request carrying no valid identity headers. `audit` logs it and lets it through, which is the rollout mode the org cutover used; it now has to be asked for. |
| `COWORK_ORGANIZATION_SWITCH_ENABLED` | `false` | Shows the canonical-web organization picker. Requires org tenancy and identity enforcement. The product enable, not a safety switch: the expected-organization boundary refuses a mismatched tab whatever this says. Long-lived Helm environments set `true`. |
| `COWORK_TRUST_IDENTITY_HEADERS` | `false` | Local tenancy only. Reads `X-User-Id` and `X-Organization-Id` on a request that passed the bearer token, and gives each pair its own partition. The partition id is `uuid5(X-Organization-Id, X-User-Id)` (`cowork.principal.personal_partition_id`); new rows carry it as `org_id` and the user as `created_by`. Only for a deployment behind a proxy that verifies each person, strips client copies of those headers, and refuses a person's write (`PUT` or `DELETE /api/v1/settings/<key>`) of an operator-pinned setting: that write lands on the global row every person reads, and nothing here refuses it. Where `COWORK_PERSONAL_RESOURCES` leaves out `memory`, the proxy must also refuse a person's `PUT` or `DELETE` under `/api/v1/memory/`: that write lands on the store every person reads, and nothing here refuses it. `COWORK_IDENTITY_ENFORCE` applies: `enforce` answers 401 without a valid pair, and `audit` serves a request with no pair as the operator view, which lists every person's conversations, running answers, projects, files, schedules and pins, and reads skills, connections, artifacts and memory from the deployment's own stores; every artifact route given a person's `project_id` there (the listing, a delete by id, drafts, workspace and review) answers 404. A pair that is present but broken (one header alone, or an id that is not a UUID) answers 401 under either mode. Paths outside `/api/`, the web UI's page shell, name no person and log nothing. A path with a decoded `.` or `..` segment answers 400. Only a genuine CORS preflight (`Origin` with `Access-Control-Request-Method`) passes the bearer check without the token, so an `OPTIONS` request can't name a person either. With it on, every connection a save writes records its creator, the person's `sub`, in the bookkeeping field `_created_by`: a create stamps it, an update keeps the stored value whatever the request says, and the operator view stamps none. It never reaches the agent's environment or the connections API's fields. Every skill created through the server (create or upload) records its creator's `sub` in its `SKILL.md` metadata as `created_by`, never a value the request or the uploaded file carries; an edit, rename or project-rename rewrite keeps it, and the operator view, the packaged built-ins and the managed skills record none. The server refuses to start with it on in org tenancy, with `COWORK_REQUIRE_AUTH` off, with `COWORK_TURN_BACKEND=remote`, whose turns keep memory in each project's own folder rather than each person's or the deployment's tiers, with `COWORK_CODE_SANDBOX` off, and without `COWORK_AUTH_TOKEN` in its environment. With it on, the token acts as any person, so the server never generates it or copies it into `<COWORK_HOME>/.env`, and it removes a `COWORK_AUTH_TOKEN` line an earlier start wrote there. The desktop sidecar reads it from `<COWORK_HOME>/.env` like every setting here, and the desktop has no Landlock, so turning it on there stops the sidecar from starting. |
| `COWORK_PERSONAL_RESOURCES` | all nine | With `COWORK_TRUST_IDENTITY_HEADERS` on, the comma-separated resources each person keeps in their own partition: `conversations`, `projects`, `files`, `skills`, `connections`, `schedules`, `pins`, `memory`, `settings`. A resource left off stays shared across the deployment, as with the setting off. `memory` left off makes the deployment's own memory one store that every person's turns read and learn into, rules, lessons and profile included: its global tier is `COWORK_MEMORY_DIR`, and each project's tier sits under it at `projects/<project id>/`, never in the project's `.anton/memory`. Each turn reads the shared global tier and its current project's tier, never another project's. The server starts, and logs a warning at start that memory is shared (or, with `MEMORY_ENABLED` false, that no turn learns and the shared store is kept). Everyone's Memories listing (`GET /api/v1/memory/`) then holds the shared global tier and every project tier in that store, the caller's own projects included, each under its project id alone: a person sees the learned tier of a project they can't open without its name or files, and gets no capability on it. `projects` requires `conversations`, and `conversations` requires `files`, `pins` and `schedules`; the server refuses to start when the list breaks that, and an unknown name fails settings load. A personal schedule runs as its creator. With `memory` personal, a person's global tier is `<shared>/<partition>/memory/users/<sub>/` and each of their project tiers sits under it at `projects/<project id>/`, never in the project's `.anton/memory`; a turn with no person reads and writes no learned memory. With schedules shared (`memory` alone listed, for one), a scheduled run has no person, so it reads and learns no memory: a shared run that used its creator's memory would show it to everyone. Deleting a project removes its tiers: the deleter's where projects are personal, every person's where they are shared, and the project's tier in the deployment's store, whichever way memory is kept now, since a tier from before a switch stays on disk. Moving `memory` onto or off the list moves and merges nothing: each person's tiers and the shared store are separate folders, and each comes back as it was. With `settings` personal, the operator-pinned keys (the model, its endpoint and credentials, memory, the per-turn budgets, and how the agent acts: `act_first` and `proactive_dashboards`) still resolve from the global row, then the environment, and a person's own row for them is ignored. Whatever the list holds, a person's `PUT` or `DELETE` of one of those keys rewrites that global row for everyone, so the fronting proxy must refuse it. With `projects` personal, every artifact route resolves a path, a preview token, a project name or an artifact id only inside the caller's own projects, a preview counts a backend as running only when this server launched it for that project and artifact, the preview proxy forwards a request only to that live backend (503 otherwise), and publish history is per person. A person's open and reveal (`/artifacts/open`, `/artifacts/reveal`), and their PDF and Word export, are refused whatever the list holds; HTML export and publishing work. With `projects` shared, deleting a project answers 409 while another person still has conversations or schedules in it, the creator's delete included. With `projects` personal, a person's delete answers 409 too while the operator view has left a conversation or schedule in their project. With `projects` shared and `conversations` personal, a conversation's workspace (`conversations/<id>/` in the project folder) stays its owner's: anyone else's read, write, delete or listing answers 404. Choosing an existing folder as a project (adopting it) is refused on this deployment whatever the list holds. With `skills` personal, nothing fans skill links out of the shared store; each turn seeds the read-only builtins into the person's store and links that store's skills into its project, and the skills others share with the person ([Sharing one skill with everyone](#sharing-one-skill-with-everyone)) into a read-only root after the managed ones. With `skills` shared, everyone uses every skill in the one store, and a person changes, disables, renames or deletes only a skill they created: anyone else's change answers 403 "This skill is shared with you and can't be changed here.", and the skills API marks it `shared` and gives them no capability. The operator view changes any skill there, one with no recorded creator included (made before creators were recorded, by the operator view, or a packaged built-in's copy). Skills name their projects by name, and a project rename rewrites those names in the renamer's skill store, with one exception: with `projects` personal and `skills` shared, a person's rename leaves the shared store alone, because a shared skill scoped to a name applies to every person's project of that name. With `skills` personal and `projects` shared, a rename rewrites only the renamer's own skills. Local OAuth sign-in is refused (403) on this deployment whatever the list holds, the operator view included: the provider's redirect to the callback carries no identity. Connect the service with its credentials instead. With `connections` personal, a turn and an artifact's live backend read only the person's own vault and the connections others share with them (`COWORK_MANAGED_DIR`). With `connections` personal and `projects` shared, a live backend preview answers 409, and so does the preview proxy, for everyone: one backend serves every person who opens that artifact, with the secrets of whoever started it. With `connections` shared, everyone uses every connection, and a person changes or deletes only a connection they created: anyone else's change answers 403 "This connection is shared with you and can't be changed here.", and the connections API marks it `managed` and `shared`; one with no recorded creator stays `managed` only. Where two connections there have names that map to the same DS_ environment prefix (for example `sales-db` and `sales_db`), a person's listing and turn keep their own and hide the other, with a warning in the log. The operator view changes any connection there, one with no recorded creator included (saved before creators were recorded or by the operator view). A turn with no person, such as a scheduled run where schedules are shared, changes none: its agent can't relabel, overwrite or delete a saved connection, and one it saves records no creator. With `connections` personal or shared, a connection the agent saves from chat updates only one the person can change: one whose identity fields (for postgres, the database alone) match a connection shared with them, or someone else's, is saved as a new connection of their own. Ignored while `COWORK_TRUST_IDENTITY_HEADERS` is off. |
| `COWORK_CODE_SANDBOX` | `off` | How the agent's code runs (`cowork.code_sandbox`). `off`: every scratchpad cell, venv check, package install and artifact check starts as a child of the server, with the server's environment and files, as on the desktop. `landlock`: each one starts through anton's launcher, which locks it with Landlock, a Linux kernel feature, and hands it an allowlisted environment without `COWORK_AUTH_TOKEN`, `DATABASE_URI` or `MINDS_API_KEY`. A cell then reads, writes and runs code only in the conversation's project folder and the asker's own `HOME` and `TMPDIR` (`home/` and `tmp/` in `<shared>/<partition>/sandbox/`, each person's own whatever `COWORK_PERSONAL_RESOURCES` leaves shared, or `<COWORK_HOME>/sandbox/` for the operator view); reads and writes `<COWORK_HOME>/shared-files`, which the server creates at start and every person's agent shares, but runs no file there by its path, so nothing one person leaves there starts as a program in another's cell (an interpreter can still run a script it reads from there, such as `python script.py`); and only reads the turn's skill folders (the asker's own, the ones shared with them, and each skill in the managed folder's `skills/`), the conversation's attachment folders, and the system's read-only files. With `skills` and `projects` both left off `COWORK_PERSONAL_RESOURCES`, each project's `skills/` links into the one shared skill store, so every cell reads that whole store, read-only. Everything else is denied: the rest of the person's partition, every other person's, `<COWORK_HOME>`'s own files, the rest of the managed folder, `/home` and the shared `/tmp`. The launcher grants what a path resolves to, so the server opens each folder it grants inside `<COWORK_HOME>`, the shared root, or the projects, files or skills root from the outermost of those roots without following links; a root kept on another disk is set by its own path, not linked into `<COWORK_HOME>`. A link on the way, which agent code from before the sandbox may have left, fails the turn when it leads to the project, `HOME`, `TMPDIR` or `shared-files` (a link at `shared-files` itself fails the start), and leaves a skill or attachment folder out of the grants. A turn also refuses a project reached through a link before writing into it, makes a missing project folder again without following links, refuses an `ANTON_CONTEXT_DIR` outside the project, and makes its own folders in the project (`skills/` and the folders in `.anton/`) without following links. With `off`, those folders are made as on the desktop. The server itself isn't locked, so a link or a hard-linked file in a person's data vault, the memory tiers a turn hands anton (the deployment's shared ones where memory is shared), or the publish store fails that read or write (the turn, the connection's save or the publish), naming the path: agent code from before the sandbox may have left it there to send the server's writes into another person's space. Everyone else reading the connections that person shares goes on without them, and the server logs an error naming the vault and each connection left out. With `COWORK_TRUST_IDENTITY_HEADERS` on and `MEMORY_ENABLED` false, a turn hands anton no stored tier, so no tier is checked. The agent's prompt names `shared-files` and says every person's agent reads and changes it. A credential probe hands its candidate credentials to the cell as its `DS_*` variables instead of a file, both the flat names the probe's prompt gives and the namespaced ones anton's connection listing gives, and one with no conversation runs in `probes/` beside the asker's `HOME`. App backends are refused (a backend preview answers 409 "App backends are off on this deployment because agent code runs in a sandbox.", the preview proxy answers 403 and forwards nothing, and the agent is never offered one), and every Code mode route answers 403. Required with `COWORK_TRUST_IDENTITY_HEADERS`. The server refuses to start with `landlock` in org tenancy, with `COWORK_TURN_BACKEND=remote`, on an anton without the launcher (`anton.core.backends.sandbox`) or whose policy model refuses the policy this server builds, on a kernel below Landlock ABI 3 (Linux 6.2; Debian 12 needs the bookworm-backports kernel, or Debian 13), with a link at `<COWORK_HOME>`'s `.env`, `.master_key`, `streams`, `tmp` or `shared`, and when its startup self-test can't confirm that a path outside a grant is denied. With `landlock` and `projects` left off `COWORK_PERSONAL_RESOURCES`, it starts and logs a warning: every member of a shared project reads and writes its folder, venvs and saved cell state included. |
| `COWORK_MANAGED_DIR` | unset | The operator's managed folder, which Cowork only reads and every turn gets, whoever asked it. `memory/` holds `rules.md` and `lessons.md` (and an empty `profile.md`): anton's Cortex reads them as a read-only layer ahead of the turn's own memory and never writes them, a turn with no person included. `skills/<slug>/SKILL.md` are skills every person lists as built-in, with no capability; they follow the harness's own skills in anton's read-only skill roots, a person's same-named skill is never linked into a turn, and every change to a managed name (create, upload, rename into, edit, disable, delete, share) answers 403 "This skill is shared with you and can't be changed here." They are the skills the operator adds, which nobody created. A managed skill folder the server's user can't read after the start is left out of the listing and every turn, logged with the `chmod` fix, and keeps its name reserved. `data-vault/` holds the operator's own connections in the vault's own file format: every vault read lists them first, marked `managed` in the connections API, a same-named connection of the reader's own stays hidden, and so does one of the reader's own whose name maps to the same DS_ environment prefix, and every change to one answers 403 "This connection is shared with you and can't be changed here.", a delete before its OAuth revoke. `shared/connections.json` lists the connections people share, a JSON list of `{user_id, partition_id, engine, name}` that `python -m cowork.ops.connections share` and `unshare` write and the server reads live. Where connections are personal, everyone else reads each listed connection after their own, live from its creator's vault, marked `managed` and `shared` in the connections API. A same-named connection earlier in that order (the operator's, then the reader's own) hides it, and so does one whose name maps to the same DS_ environment prefix (for example `sales-db` and `sales_db`), which is logged as a warning; a listed connection that no longer exists is skipped. Everyone else's change to it (delete, access mode, picked files, token, label, or a save of that engine and name) answers 403 with the same sentence, before any OAuth revoke. Its creator sees it `shared` but not `managed` and changes, reconnects and deletes it as before, and everyone else sees each change, a refreshed token included. Once its creator deletes it, its entry shares nothing, and a new connection of theirs under that name answers 403 "This name is still shared for a connection of yours that is gone. Ask your operator to stop sharing it before you use the name again." until the operator unshares it. Blank means unset. The server refuses to start with it set in org tenancy or with `COWORK_TURN_BACKEND=remote`, whose workers receive none of it, when the folder or any of `memory/`, `skills/` and `data-vault/` is missing (it names each one), when the server's user can't read and search `memory/` (every turn would fail reading its rules), when it can't read and search `skills/` or `data-vault/`, or can't read a managed skill folder or its `SKILL.md` (each would crash every Skills or Connections page and every turn; it names the path and the `chmod` fix), when it can't read a managed connection file (the connection would vanish for everyone and free its name; it names the file and the `chown` fix), when `shared/connections.json` is malformed (it names the file), and when the installed anton's `Cortex` takes no `managed_hc`. A missing `shared/` folder or file shares nothing, and a registry broken after the start shares nothing and logs an error until it is fixed. The desktop sidecar reads it from `<COWORK_HOME>/.env` like every setting here. |
| `COWORK_SHARED_DIR` | `~/.cowork` | **Org mode and personal partitions only.** Root of the org-keyed tree: `<shared>/<org_id>/{skills,memory,projects,files,data-vault}` (and `publish` for a personal partition), where a personal partition's `<org_id>` is its partition id. Where skills are personal, it also holds `shared/skills.json`, the skills people share, which Cowork writes ([Sharing one skill with everyone](#sharing-one-skill-with-everyone)); its server user needs write access to `shared/`. In cloud, point it at the durable mount — on the default the data is ephemeral (boot warning). |
| `COWORK_PROJECTS_DIR` | `~/.cowork/projects` | Project storage root (local mode only) |
| `COWORK_FILES_DIR` | `~/.cowork/files` | Uploaded files root (local mode only) |
| `COWORK_SKILLS_DIR` | `~/.cowork/skills` | Skills store root (local mode only) |
| `COWORK_MEMORY_DIR` | `~/.cowork/memory` | Memory store root (local mode only) |
| `COWORK_VAULT_DIR` | `~/.cowork/data-vault` | Connector credential vault |
| `COWORK_OPENAI_COMPATIBLE_API` | `chat_completions` | The API an `openai_compatible` provider's planning and coding roles call. `responses` moves them to `{base}/responses` through anton's openai flavor, where OpenAI and Azure accept function tools together with a reasoning effort. It needs an anton that reports `RESPONSES_TRANSPORT_READY`; with an older anton both roles stay on chat completions, and Cowork logs one warning that says so. On the Responses path the agent loop runs without web tools, because OpenAI's hosted `web_search` reads the web from the provider's side, outside the deployment's egress controls. A Python cell's `web_search()` is separate and unchanged. The router and Gemini stay on chat completions. Read once at start. |

**Database pool**: these read their bare names, with no `DATABASE_` prefix. A SQLite engine is built without them and waits SQLAlchemy's 30 s for a connection outside a database unit; its database units run one at a time, and `POOL_TIMEOUT` still bounds a unit's waits.

| Variable | Default | Description |
|----------|---------|-------------|
| `POOL_SIZE` | `20` | Connections the pool keeps open. |
| `MAX_OVERFLOW` | `20` | Connections the pool may open beyond `POOL_SIZE` under load. Database units take one of `POOL_SIZE` plus `MAX_OVERFLOW` slots before they ask the pool. A negative value means no limit, for the pool and for units. |
| `POOL_TIMEOUT` | `5` | Seconds a database unit may wait in all, for a slot and then for a connection, before it is refused: a request with `503` and a `Retry-After` of the same number of seconds, a streamed answer with `server_busy`. On Postgres it is also the pool's own wait, so every route that reads the database answers `503` with `Retry-After` once it passes, `/api/v1/health/` included. `/api/v1/health/live` reads no database and keeps answering. It must be at least `1`; a lower value stops the server at startup. |

**Harness-level** (`ANTON_*`) — configure a specific agent harness. These are read by the harness adapter, not by cowork-server core. They use the harness prefix because the upstream agent library (anton) defines them:

| Variable | Harness | Description |
|----------|---------|-------------|
| `ANTON_PUBLISH_URL` | Anton | Artifact publish endpoint. Pointed anywhere but MindsHub's service, it takes HTML and Markdown only: sharing a full-stack app is refused before anything is uploaded, because that upload carries the app's connection credentials |
| `ANTON_SKILLS_ROOT_DIR` | Anton | Skill file storage |
| `ANTON_GLOBAL_MEMORY_ROOT_DIR` | Anton | Global memory files |

**By field name** (local tenancy only): every `UserSettings` field also reads the environment variable named after it, and a stored row wins over the variable. The router's reasoning effort is set this way, beside `PLANNING_REASONING_EFFORT` and `CODING_REASONING_EFFORT`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_ENABLED` | `true` | Whether the agent reads and learns memory. The key is operator-pinned, so on a deployment that trusts identity headers a stored global row wins over the variable for every person. There `false` turns learned memory off, whether memory is kept per person or shared: every turn gets two empty learned tiers and the operator's managed rules, runs with memory mode off and no background memory passes, and writes nothing; the Memories listing is empty, and a change or delete of a slot under `/api/v1/memory/` answers 409; and stored memory stays on disk, unread, until memory goes back on. Elsewhere (the desktop, hosted org mode) `false` only turns anton's memory mode off, and stored memory is still read. |
| `ROUTER_REASONING_EFFORT` | unset | The router model's reasoning effort. The router's provider serves anton's history summaries and the route gate, so the effort reaches both. On `openai_compatible` the gate runs `ROUTER_MODEL`. On other providers the gate runs the provider's default router model, so the effort is sent only while `ROUTER_MODEL` is that default. Unset or empty sends no effort, which a model that doesn't reason needs, because it refuses any effort. On `openai_compatible`, a router model that reasons by default, such as `gpt-5.6-luna`, needs `none` to take the gate's function tool on chat completions. The composer's per-task effort never applies to the router. |

In Docker/Lightsail deployments, the container also receives `ANTON_MINDS_API_KEY`, `ANTON_OPENAI_API_KEY`, etc. — these are consumed by the Anton agent library directly (not by cowork-server settings), and are injected by the provisioning lambda via cloud-init user-data.

### Operator commands

On a deployment that trusts identity headers, an operator with a shell where cowork-server runs acts on one person's space with `python -m cowork.ops.<module> <action>`. Every action takes `--organization-id`, the deployment's id the proxy sends as `X-Organization-Id`, and `--user-id`, the person's `sub` the proxy sends as `X-User-Id`, except the memory actions where memory is shared, which act on the deployment's one store and refuse both. cowork-server works out the space from them with `cowork.principal.personal_partition_id`, as it does for the person's requests. Results print on stdout as plain text, and errors and log lines on stderr; a command that can't do what it was asked exits with status 1. Every connection `share` and `unshare` takes `--registry-stdout`: it then writes nothing, prints the new registry on stdout (exactly the bytes it would have written) and its message on stderr, so a caller can check the registry and write it itself. A skill `share` and `unshare` take no such flag and no `--managed-dir`: they change the shared-skill list as the person's own Skills page does, so they run inside Cowork's container as the user Cowork runs as, and refuse root, whose `shared/` folder or lock file would stop Cowork's own writes.

| Command | What it does |
|---------|--------------|
| `python -m cowork.ops.memory print ...` | Where memory is kept per person (`memory` in `COWORK_PERSONAL_RESOURCES`), prints what the person's agent learned, as their Memories page lists it: the global tier, then each project tier, with the folder each is in, and refuses a call without both `--organization-id` and `--user-id`. Where memory is shared, takes neither (either one is refused) and prints the deployment's one store: the global tier, then each project tier, titled `Project <project id>`. It ends with `Nothing learned.` when every tier is empty. While `MEMORY_ENABLED` is false it prints only that memory is off and stored memory is kept, unread, until memory goes back on. |
| `python -m cowork.ops.memory clear ...` | Where memory is kept per person, removes everything the person's agent learned, in every project of theirs (`cowork.services.memory.reset_person_memory`). Where memory is shared, takes neither person argument (either one is refused, so a clear meant for one person never clears everyone's) and removes the shared store's global slots and every project tier in it (`cowork.services.memory.clear_deployment_memory`), and no person's tier or managed file. A link where the store's `projects/` folder should be stops print and clear with an error before anything is read or removed. It can't be undone. While `MEMORY_ENABLED` is false it refuses and removes nothing, so stored memory stays as it is. The person arguments are checked first in every case. |
| `python -m cowork.ops.skills share ... --skill <slug>` | Shares the person's skill `<slug>` with everyone, read-only, on their behalf: adds it to `<COWORK_SHARED_DIR>/shared/skills.json`. Nothing is copied: everyone else reads the skill live from the person's store, and the person still edits, disables and deletes it, but can't rename it until it is hidden. They can hide it again from their Skills page. It runs the checks and prints the refusal texts of a share from the Skills page ([Sharing one skill with everyone](#sharing-one-skill-with-everyone)), plus "User `<sub>` has no skill named `<slug>`." for a skill the person doesn't have and "Skill `<slug>` of user `<sub>` is already shared." for one already shared. |
| `python -m cowork.ops.skills unshare ... --skill <slug>` | Removes the person's entry for `<slug>` from `<COWORK_SHARED_DIR>/shared/skills.json`, whether or not the skill still exists, so it also clears the entry of an account the identity provider no longer holds. The person's skill stays theirs, in their store. Refused when the person shares no skill by that name. |
| `python -m cowork.ops.skills snapshot ... [--project <name>]` | Lists the skills the person's agent sees in a project (`general` unless named), the skills others share with them included, with where each comes from (`own`, `host`, `managed`, `shared` or `anton`), and prints one `sha256`: the sha256 of the lines `<sha256 of the file>  <slug>/<path in the skill>`, one per file of each skill, sorted by slug and path, as `sha256sum` prints them. Every file the agent can read counts, helper scripts and reference files included; a skill's `stats.json`, hidden files and links don't. A shared skill's lines hash the files in its creator's store, so the digest changes when its creator edits it. |
| `python -m cowork.ops.connections share ... --name <name> [--engine <engine>] --managed-dir <folder>` | Shares the person's saved connection named `<name>` with everyone, read-only: adds it to `<folder>/shared/connections.json`, where `<folder>` is the managed folder, writable by the caller. Nothing is copied: everyone else reads the connection live from the person's vault, and the person still changes it. Names are unique per engine only: one Google account connected to Gmail and to Calendar saves two connections under one name, and `--engine` picks one. Refused where `COWORK_PERSONAL_RESOURCES` leaves connections shared, when the person has no connection by that name, or more than one and no `--engine` (the message names `--engine`), when it is already shared, when someone else already shares one of that engine and name, when the managed `data-vault/` holds one, and when the registry is malformed, which the command never rewrites. |
| `python -m cowork.ops.connections unshare ... --name <name> [--engine <engine>] --managed-dir <folder>` | Removes the person's connection named `<name>` from `<folder>/shared/connections.json`. The person's connection stays theirs, in their vault. Refused when the person shares none by that name, and when they share it for more than one engine and no `--engine` is given. |

## Docs

- [docs/DESIGN.md](docs/DESIGN.md) — Architectural overview and design decisions
- [docs/MIGRATION.md](docs/MIGRATION.md) — Migration guide from the legacy server
- [docs/MIGRATION_PROGRESS.md](docs/MIGRATION_PROGRESS.md) — Migration status tracker

## License

See [LICENSE](LICENSE).
