# Cowork Server Migration

Reference for major decisions and risks in the migration from the original cowork server (embedded in the Electron app) to the standalone cowork-server (FastAPI + SQLite).

See also: the [Cowork Server API for Agents](https://docs.google.com/document/d/...) design doc by Minura Punchihewa (May 12, 2026) which defined the original intent.

## Architecture

The core idea: decouple the agent from the application. The server defines a harness-agnostic API contract, and each agent (Anton, Hermes, etc.) is a concrete `HarnessProvider` registered via a decorator. The server handles conversations, projects, settings, schedules, and pins as **app components**. The harness is responsible only for streaming responses, memory, and skills.

`HarnessProvider` (`harnesses/base.py`) is a Protocol defining `stream_response()`, `sync_skills()`, `overwrite_memory()`, `retrieve_memory()`, `delete_memory()`, and `list_memory()`. Harnesses register with `@register` and are instantiated by name via `get_harness()`.

## Build and Release

The Electron app (`cowork/`) manages the server lifecycle:

- **Installer** (`src/main/installer.ts`): On first launch, installs `uv` (Python package manager), then runs `uv tool install cowork-server` from the pinned git tag. The version is pinned in `COWORK_SERVER_VERSION`.
- **Server process** (`src/main/server-process.ts`): In dev mode, runs `uv run cowork-server` from the sibling source directory. In production, runs the installed `cowork-server` binary from `~/.local/bin/`. Polls `/health` to confirm startup.
- **Dependency check** (`src/main/server-deps.ts`): Verifies Python dependencies (uvicorn, etc.) are available before attempting server start.

## API Changes

The server API was restructured from the original for clarity, but the client wasn't updated in lockstep. To bridge the gap, a **compat layer** was introduced:

- **CamelCase translation**: The original server used camelCase in JSON responses (matching the JS client). The new server uses snake_case internally. `CamelResponse`/`CamelRequest` base classes (`schemas/base.py`) apply `alias_generator = to_camel` so the wire format stays camelCase. Tagged `SHIM:client-compat` for future removal.
- **Compat stubs** (`api/v1/endpoints/compat/stubs.py`): Placeholder endpoints for features not yet ported (integrations/OAuth, browse status). Return safe empty responses so the client doesn't 404.
- **Attachments bridge**: The design doc replaced attachments with a Files API + input_file content blocks in the Responses request. The compat layer maps old attachment endpoints to the new `FileService` so the current client still works.

Key structural API changes from the design doc:
- `/projects/active` removed — active state is a field on the project, toggled via PATCH
- `/conversations/{id}/turns/{index}` became `/conversations/{id}/items` for message retrieval
- Connectors, data sources, integrations, and data vault consolidated under `/connectors/`
- Settings moved from `.env` file read/write to DB-backed CRUD at `/settings/{key}`
- Files API (`/files/`) introduced as the canonical upload path

## What Was Ported vs. Rethought

**Ported largely as-is:**
- Artifacts service — still entirely filesystem-based (`<project>/.anton/artifacts/<slug>/`), reading `metadata.json` from disk. No DB table.
- Publish service — same flow, same `state.json` on disk.
- Scratchpad — only the cancel stub exists; the full scratchpad is still managed by the harness (Anton core).
- Connectors/data vault — `LocalDataVault` from Anton core, filesystem-based encrypted credentials.

**Rethought:**
- **Conversations and messages** — previously stored as episodes on the filesystem by Anton's `HistoryStore`. Now first-class DB entities. Messages are persisted during streaming with a companion `message_events` table that stores the raw SSE event log for replay.
- **Settings** — previously `~/.anton/.env` parsed by both Electron and the Python server. Now a DB `settings` table with encryption for sensitive values. A one-time migration reads the old `.env` and seeds the DB (read-only, non-destructive).
- **Memory** — the design doc noted memory is harness-specific. `MemoryService` is a pass-through that delegates to whichever `HarnessProvider` is active. Anton uses its `Cortex`/`Hippocampus` system with files on disk; Hermes has its own implementation. No shared DB table.
- **Skills** — moved to a DB `skills` table, synced to the active harness via `sync_skills()`.
- **Projects** — previously implicit from the filesystem. Now a DB `projects` table with a "General" project auto-seeded at startup.

## Data Migration

The migration from filesystem to SQLite is **additive and non-destructive** — nothing in `~/.anton/` is ever deleted or modified.

- **Settings**: Auto-migrated from `~/.anton/.env` to the `settings` DB table on first access (only if the table is empty). The `.env` file is left untouched.
- **Conversations**: Fresh start. Old episodes in `<project>/.anton/episodes/` are not imported. The `EpisodicMemory`/`HistoryStore` code is commented out with a TODO questioning whether it's needed.
- **Everything else** (artifacts, memory, vault, project files): Remains on the filesystem by design. The DB tracks metadata only where needed (projects, files, pins).

Recovery: `~/.cowork/cowork.db` can be deleted for a clean slate; settings will re-seed from `.env` on next startup. All original data remains at `~/.anton/`.

## Path Rename: `.anton` to `.cowork`

The `.anton` → `.cowork` path rename has **not** been started. Both the server and the Electron client still reference `~/.anton/.env` for settings, and per-project `.anton/` directories (artifacts, memory, instructions) are used throughout both repos. When this rename is undertaken, it should be coordinated across both repositories to avoid client/server mismatch. Any rename must be non-destructive — read from the new location, fall back to the old, never delete the original.

## Open Items

- **Episodic history import**: No migration tool exists for old `<project>/.anton/episodes/` data. Decide whether this is needed.
- **OAuth connectors**: Ported but not yet end-to-end tested with the new frontend integration.
- **Connector submissions**: Currently always routes through Anton harness regardless of active harness setting.
- **Client-side turn data**: Rich UI state (thinking steps, scratchpad cells) lives in browser localStorage, not the server. Cross-device sync would require promoting this to a server-side sidecar.
