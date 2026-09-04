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

### Logging

Set `LOG_LEVEL` (default `INFO`) to control verbosity. Enable file logging with `ENABLE_FILE_LOGGING=true` (writes to `LOG_DIR`, defaults to `~/.cowork/logs/`).

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
  harnesses/          # Agent adapters (Anton, Hermes, etc.)
```

The server is designed to be **agent-agnostic** — core features (projects, conversations, files) are shared across agents, while agent-specific behavior lives in harness adapters. See [docs/DESIGN.md](docs/DESIGN.md) for the full architectural rationale.

### Harness system

A **harness** adapts an external agent library (Anton, Hermes, etc.) to the cowork-server interface. All harnesses implement the `HarnessProvider` protocol (`harnesses/base.py`), which exposes streaming responses, skill sync, and memory operations. The active harness is selected via the `harness` user setting. To add a new agent, implement the protocol and register it with the `@register` decorator.

### Streaming & scheduling

Agent responses stream to clients via **Server-Sent Events** (SSE) on `POST /responses/`. The server tracks in-flight streams and supports cancellation (`/responses/cancel`) and late-join tailing (`/responses/tail`).

A background **scheduler** loop polls the database every 30 seconds for due schedules, supporting `once`, `hourly`, `daily`, and `weekly` cadences. Each run creates a conversation and is tracked in `schedule_runs`. Deleting that conversation does not delete the run: the run keeps its status, timings, and error as audit history, and only its link to the conversation is released. A channel binding pinned to the conversation is released the same way, so the external chat stays bound to its project and the next inbound message starts a fresh conversation.

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

This split is the result of an ongoing migration from a purely filesystem-based architecture. Structured data that benefits from querying and relationships — conversations, messages, settings, schedules — lives in SQLite. Components that are inherently file-based — project working directories, agent artifacts, harness-managed memory, connector vault credentials, and skills — remain on the filesystem by design. (Skills briefly lived in a DB table; they were moved back to canonical `SKILL.md` files so they can be edited, uploaded, and distributed per project — see [docs/SKILLS.md](docs/SKILLS.md).) See [docs/SERVER_MIGRATION.md](docs/SERVER_MIGRATION.md) for the full migration story.

Agents (via their harness) have read/write access to their project's working directory and the private `.anton/` subdirectory. They do **not** access the SQLite database directly — all DB interaction flows through the service layer.

**Settings** use a hybrid approach: user preferences and API keys are stored in the `settings` DB table (with Fernet encryption for secrets), while connector credentials live in the filesystem vault (`data-vault/`).

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
was started with** — `export_connection_info()` hands it a string once and it
has no supplier, so a pad-side model call still runs on that value. Refreshing
it needs a pad IPC contract, which ENG-2116 scoped out.

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
| `/publish` | Publish HTML artifacts to 4nton.ai |
| `/connectors` | Third-party service connections and OAuth |
| `/settings` | User preferences and API keys |
| `/runtime-credential` | Desktop hand-over of the MindsHub credential (write-only, loopback, local mode) |
| `/hub/workspaces` | Which MindsHub workspace this person is working in |

### The MindsHub workspace selector

`/api/v1/hub/workspaces` backs the workspace selector at the top of the desktop
app's sidebar. A **MindsHub Workspace** is an org-internal container that owns hub
resources (API keys, artifacts, model entitlements) and lives in the auth
service. It has nothing to do with the filesystem directories this repo calls
workspaces, which is why the stored key is `hub_workspace_id`.

Five things about it are worth knowing before changing it.

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

**The switch is auth's Statsig gate, not a local setting.** Auth declares
`authorization_ui` in its `configs/statsig_gates.json`, evaluates it with its
server SDK, and reports the verdict in the entitlements payload; this service
reads it from there. One gate governs the console and Cowork rather than two that
can disagree, and Cowork holds no Statsig client and no SDK key. Every answer
short of a definite yes reads as off: no bearer, auth unreachable, a version of
auth with no gates field, or the gate off. `COWORK_HUB_WORKSPACES_FORCE_ON` is an
ON-only development override for walking the surface where no rule targets you;
it cannot switch the surface off, so it cannot escape the kill switch.

**Both caches are keyed on the credential, not just the caller.** Auth answers
the listing and the gate per caller: an owner or admin sees every workspace in
the organization, a member only the ones they hold a grant on, and
`authorization_ui` declares `idType: userID`. So an organization-keyed cache
served one admin's menu to every member for the whole TTL, and the grant check on
`PUT /active` reads the same entry. The key is
`(auth host, organization, user, credential digest)`. The digest is not
belt-and-braces: `user_id` comes from the gateway-set principal and is `None` on
every desktop request, because `scope_from_principal` returns `LOCAL_SCOPE`
outside org mode, so identity alone collapses to one shared entry and a
sign-out/sign-in as another account would be served the previous one's
workspaces. A new session means a new token means a new entry. Entries are swept
on write, since nothing re-reads a departed caller's key and the dicts would
otherwise grow for the process lifetime. The TTL follows whether **auth
answered**, not what it said: a gate auth evaluated as off is a real answer and
keeps the long TTL, which matters because off is the state this ships in.

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

`COWORK_ORGANIZATION_BOUNDARY_MODE=audit` logs a missing, malformed, or changed
expected organization and lets the request continue. In `enforce` mode, a
missing header returns 426 and a malformed or changed value returns 409. Both
responses carry `X-Cowork-Organization-Reload: required`, a JSON `code` and
`detail`, and `Cache-Control: no-store`. The browser reloads instead of letting
an old document continue under a new Keycloak organization. A request already
inside a route keeps the `Principal` created at its start, so a concurrent
session change cannot retarget that in-flight operation.

`GET /api/v1/capabilities/organization-switch` is authenticated and returns
protocol version 1. It reports `expectedOrganizationEnforced: true` only when
both identity and expected-organization enforcement are active. It reports
`enabled: true` only when those boundaries are active and
`COWORK_ORGANIZATION_SWITCH_ENABLED=true`.

Roll this out in four separate steps:

1. Deploy the capability-aware Cowork client. The picker stays hidden.
2. Deploy cowork-server with the organization boundary in `audit` and switching
   disabled.
3. Set the boundary to `enforce` on every replica while switching remains
   disabled, then verify the capability still reports `enabled: false`.
4. Enable switching separately and verify the capability reports all three
   required values.

Inside one organization, two different rules apply, and which one you get
depends on the resource:

- **Shared with the organization:** projects, project files at the project root,
  skills, project memory, connected apps. Every member reads them.
- **Private to whoever created it:** conversations and their history, scheduled
  tasks, personal memory, uploaded files, and everything under a conversation's
  own workspace at `conversations/<conversation_id>/`. Live artifacts are in
  that last group, because the agent writes them into the conversation it is
  running in.

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

Both of those decide from a resolved path and the route then opens that path, so
the decision is carried to the open rather than trusted afterwards: every
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
default is a config edit plus an apply, not a release. MindsHub Air's usage draws the monthly included allowance, so a user who has
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
invalid. `MINDS_PROBE_MODEL` (`mindshub_air`) draws the monthly included allowance
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

## Configuration

Configuration is read from the database (`UserSettings` table) and can be managed through the Settings UI in the desktop app or via `PUT /api/v1/settings/`.

Environment variables fall into two namespaces:

**Server-level** (`COWORK_*`) — control the cowork-server process itself:

| Variable | Default | Description |
|----------|---------|-------------|
| `COWORK_LISTEN_PORT` | `26866` | Server port |
| `COWORK_SERVER_HOST` | `127.0.0.1` | Bind address |
| `COWORK_TENANCY_MODE` | `local` | `local` is the desktop sidecar: one user, no organization, no identity headers. `org` is the cloud deployment and turns on everything in "Who can read what in org mode" above. |
| `COWORK_IDENTITY_ENFORCE` | `enforce` | Org mode only. `enforce` answers 401 to a request carrying no valid identity headers. `audit` logs it and lets it through, which is the rollout mode the org cutover used; it now has to be asked for. |
| `COWORK_ORGANIZATION_BOUNDARY_MODE` | `enforce` | Canonical web only. `enforce` requires a browser JWT request to name the trusted organization it expects. `audit` logs violations and accepts them for a staged rollout. Long-lived Helm environments explicitly start in `audit`. |
| `COWORK_ORGANIZATION_SWITCH_ENABLED` | `false` | Enables the version 1 organization-switch capability only while identity and expected-organization enforcement are both active. |
| `COWORK_SHARED_DIR` | `~/.cowork` | **Org mode only.** Root of the org-keyed tree: `<shared>/<org_id>/{skills,memory,projects,files}`. In cloud, point it at the durable mount — on the default the data is ephemeral (boot warning). |
| `COWORK_PROJECTS_DIR` | `~/.cowork/projects` | Project storage root (local mode only) |
| `COWORK_FILES_DIR` | `~/.cowork/files` | Uploaded files root (local mode only) |
| `COWORK_SKILLS_DIR` | `~/.cowork/skills` | Skills store root (local mode only) |
| `COWORK_MEMORY_DIR` | `~/.cowork/memory` | Memory store root (local mode only) |
| `COWORK_VAULT_DIR` | `~/.cowork/data-vault` | Connector credential vault |
| `COWORK_HUB_WORKSPACES_FORCE_ON` | `false` | Development override that turns the MindsHub workspace surfaces on where no Statsig rule targets you. ON only, so it can never switch them off and never escape the kill switch. The switch itself is auth's `authorization_ui` gate; see "The MindsHub workspace selector" above. Never set in a deployed environment. |

**Harness-level** (`ANTON_*`, `HERMES_*`) — configure a specific agent harness. These are read by the harness adapter, not by cowork-server core. They use the harness prefix because the upstream agent libraries (anton, hermes-agent) define them:

| Variable | Harness | Description |
|----------|---------|-------------|
| `ANTON_PUBLISH_URL` | Anton | Artifact publish endpoint |
| `ANTON_SKILLS_ROOT_DIR` | Anton | Skill file storage |
| `ANTON_GLOBAL_MEMORY_ROOT_DIR` | Anton | Global memory files |
| `HERMES_HOME` / `HERMES_ROOT_DIR` | Hermes | Hermes data root |

In Docker/Lightsail deployments, the container also receives `ANTON_MINDS_API_KEY`, `ANTON_OPENAI_API_KEY`, etc. — these are consumed by the Anton agent library directly (not by cowork-server settings), and are injected by the provisioning lambda via cloud-init user-data.

## Docs

- [docs/DESIGN.md](docs/DESIGN.md) — Architectural overview and design decisions
- [docs/MIGRATION.md](docs/MIGRATION.md) — Migration guide from the legacy server
- [docs/MIGRATION_PROGRESS.md](docs/MIGRATION_PROGRESS.md) — Migration status tracker

## License

See [LICENSE](LICENSE).
