# Cowork Server

FastAPI backend for [Minds Cowork](https://github.com/mindsdb/cowork). Manages projects, conversations, files, scheduling, memory, and agent orchestration with a SQLite-backed data layer.

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

## Releasing

1. Bump `version` in `pyproject.toml`
2. Create a GitHub release with a matching tag (e.g. `v0.1.5`)
3. The [`publish.yml`](.github/workflows/publish.yml) workflow builds and publishes to [PyPI](https://pypi.org/project/cowork-server/) via OIDC trusted publishing

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

## API

All endpoints live under `/api/v1/`. Key resource groups:

| Path | Description |
|------|-------------|
| `/health` | Readiness probe (includes `server_version` and `anton_version`) |
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

## Configuration

Configuration is read from the database (`UserSettings` table) and can be managed through the Settings UI in the desktop app or via `PUT /api/v1/settings/`.

Environment variables fall into two namespaces:

**Server-level** (`COWORK_*`) — control the cowork-server process itself:

| Variable | Default | Description |
|----------|---------|-------------|
| `COWORK_SERVER_PORT` | `26866` | Server port |
| `COWORK_SERVER_HOST` | `127.0.0.1` | Bind address |
| `COWORK_PROJECTS_DIR` | `~/.cowork/projects` | Project storage root |
| `COWORK_FILES_DIR` | `~/.cowork/files` | Uploaded files root |
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
