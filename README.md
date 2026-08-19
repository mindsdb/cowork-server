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

A background **scheduler** loop polls the database every 30 seconds for due schedules, supporting `once`, `hourly`, `daily`, and `weekly` cadences. Each run creates a conversation and is tracked in `schedule_runs`.

## Data Layer

Data lives in two places: a **SQLite database** for structured records and the **filesystem** for project files and agent workspaces. Understanding both is essential.

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

### The default model is the one the free allowance covers

Every minds-cloud role defaults to `mindshub_air` (`MODEL_ROLE_DEFAULTS` in
`cowork/common/settings/app_settings.py`), for all three roles: planning, coding
and router. Its usage draws the monthly included allowance, so a user who has
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
| `COWORK_SHARED_DIR` | `~/.cowork` | **Org mode only.** Root of the org-keyed tree: `<shared>/<org_id>/{skills,memory,projects,files}`. In cloud, point it at the durable mount — on the default the data is ephemeral (boot warning). |
| `COWORK_PROJECTS_DIR` | `~/.cowork/projects` | Project storage root (local mode only) |
| `COWORK_FILES_DIR` | `~/.cowork/files` | Uploaded files root (local mode only) |
| `COWORK_SKILLS_DIR` | `~/.cowork/skills` | Skills store root (local mode only) |
| `COWORK_MEMORY_DIR` | `~/.cowork/memory` | Memory store root (local mode only) |
| `COWORK_VAULT_DIR` | `~/.cowork/data-vault` | Connector credential vault |

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
