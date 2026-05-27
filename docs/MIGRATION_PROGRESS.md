# Migration Progress

Tracks the status of migrating compat/shim endpoints to canonical APIs,
per the spec in `docs/MIGRATION.md`.

## Completed Migrations

These compat routes have been fully migrated — the client now calls
canonical endpoints directly.

| Old compat route | Canonical endpoint | Notes |
|---|---|---|
| `GET /datasources` | `GET /connectors/connections` | Client wraps bare array |
| `DELETE /datasources/{e}/{n}` | `DELETE /connectors/connections/{e}/{n}` | Direct path change |
| `GET /datasources/{e}/{n}` | `GET /connectors/connections/{e}/{n}` | Direct path change |
| `POST /datasources` | Client-side stub | Was always a no-op stub |
| `POST /datasources/validate` | Client-side stub | Was always a no-op stub |
| `GET /connectors` (flat) | `GET /connectors/specs` | Response shape: bare array |
| `GET /connectors/{id}` (flat) | `GET /connectors/specs/{id}` | Direct path change |
| `POST /connectors/match` (flat) | `POST /connectors/specs/match` | Direct path change |
| `POST /connectors/{id}/save` | `POST /connectors/submissions` | connector_id moved to body |
| `POST /datavault/submissions` | `POST /connectors/submissions` | Direct path change |
| `GET /conversations/{id}/messages` | `GET /conversations/{id}/items` | Response: bare array |
| `GET /responses/in-flight-list` | (now canonical on `/responses`) | Moved to responses.py |
| `GET /responses/in-flight` | (now canonical on `/responses`) | Moved to responses.py |
| `POST /responses/cancel` | (now canonical on `/responses`) | Moved to responses.py |
| `GET /responses/tail` | (now canonical on `/responses`) | Moved to responses.py |
| `GET /projects/active` | Derived client-side from project list | `is_active` field on projects |
| `DELETE /pins/{type}/{id}` | `DELETE /pins/{id}?item_type=` | Client already used canonical |
| `GET /artifacts` (stub) | `GET /artifacts` (real impl) | Ported from old server |
| `GET /artifacts/preview` (stub) | `GET /artifacts/preview` (real impl) | Ported from old server |
| `POST /artifacts/preview-mount` (stub) | `POST /artifacts/preview-mount` (real impl) | Ported from old server |
| `POST /artifacts/open` (stub) | `POST /artifacts/open` (real impl) | Ported from old server |
| `POST /artifacts/reveal` (stub) | `POST /artifacts/reveal` (real impl) | Ported from old server |
| `GET /publish` (stub) | `GET /publish` (real impl) | Ported from old server |
| `POST /publish` (stub) | `POST /publish` (real impl) | Ported from old server |
| `GET /{name}/instructions` (stub) | `GET /projects/{name}/instructions` (canonical) | Ported from old server |
| `GET /{name}/files` (stub) | `GET /projects/{name}/files` (canonical) | Ported from old server |
| `GET /{name}/files/{path}` (stub) | `GET /projects/{name}/files/{path}` (canonical) | Ported from old server |
| `PUT /{name}/files/{path}` | `PUT /projects/{name}/files/{path}` (canonical) | Ported from old server |
| `POST /preview-mount-file` (stub) | `POST /projects/preview-mount-file` (canonical) | Ported from old server |
| `POST /{name}/files/upload` | `POST /projects/{name}/files/upload` (canonical) | Ported from old server |
| `DELETE /{name}/files/{path}` | `DELETE /projects/{name}/files/{path}` (canonical) | Ported from old server |
| `GET /{name}/files-raw/{path}` | `GET /projects/{name}/files-raw/{path}` (canonical) | Ported from old server |
| `GET /settings` (compat blob) | `GET /settings/` (per-key list) | Client transforms SettingResponse[] to flat blob |
| `PUT /settings` (compat bulk patch) | `PUT /settings/{key}` (per-key upsert) | Client writes each changed key individually |
| `GET/POST /settings/raw` (compat) | `GET/POST /settings/raw` (canonical) | Legacy .env access for Onboarding |
| `POST /settings/test-providers` | `POST /settings/test-providers` | Moved to canonical settings.py |
| `POST /settings/validate` | `POST /settings/validate` | Moved to canonical settings.py |
| `GET /settings/configured` | `GET /settings/configured` | Moved to canonical settings.py |
| `GET /settings/install-status` | `GET /settings/install-status` | Moved to canonical settings.py |
| `GET /settings/reveal-key/{name}` | `GET /settings/reveal-key/{name}` | Moved to canonical settings.py |
| `POST /settings/validate-provider` | `POST /settings/validate-provider` | Moved to canonical settings.py |
| (state.json providers list) | `providers_json` DB setting | Client saves as JSON-encoded string |
| (state.json providerStatus) | Client-side only | Ephemeral; from test-providers response |

## Remaining Compat Routes (3)

### 1. Integrations (`integrations_router`)
- **Routes:** `GET /`, `POST /{service}/oauth/start`
- **Why:** OAuth flows partially ported to `/connectors/oauth` but the client still calls `/integrations/` paths.
- **To migrate:** Update client to use `/connectors/oauth/{engine}/start` paths. The integrations list can be derived from connector specs that have OAuth methods.

### 2. Attachments (`attachments_router`)
- **Routes:** `GET /{project}/{session}`, `POST /{project}/{session}/upload`, `DELETE /{id}`, `DELETE /{project}/{session}/{id}`
- **Why:** Client uses attachment-specific upload/list/delete. Stubs proxy to FileService.
- **To migrate:** Rewrite client to use `POST /files/`, `GET /files/`, `DELETE /files/{id}`. Also need `attachmentRawUrl()` equivalent (file content download) and to remove `moveAttachmentToProject()` (no server equivalent).

### 3. Scratchpad + Browse (minimal stubs)
- **Routes:** `POST /scratchpad/cancel`, `GET /browse/status`
- **Why:** Scratchpad cancel is a no-op stub; browse status returns `{available: false}`.
- **To migrate:** Low priority. Not actively used. Cancel should ideally happen via stream disconnect.

## Other Compat Mechanisms (not route-level)

- **CamelResponse / CamelRequest** (`cowork/schemas/base.py`): Auto camelCase serialization. Remove `alias_generator` when client accepts snake_case.
- **`# SHIM:client-compat` markers**: Grep for these across the codebase to find all compat code.
- **Memory endpoint** (`memory.py`): Has inline format bridge (flat items to grouped sections). Marked with SHIM comment.
