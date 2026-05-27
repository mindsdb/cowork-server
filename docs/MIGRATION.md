# Cowork Server API for Agents

**Status:** In Progress  
**Review:** In Progress  
**Date:** May 12, 2026  
**Author:** Minura Punchihewa  
**Repository:** [mindsdb/cowork-server](https://github.com/mindsdb/cowork-server)

## Overview

This document presents the proposed specification for the Cowork server API and how individual agents can be onboarded.

The API should be defined independently of any particular agent while still aligning with the implemented Cowork UI. This specification also introduces a base abstraction for agents (harnesses) mapped to API endpoints, so onboarding a new agent is primarily an abstraction implementation task.

A key architectural concern is separating application-owned components from harness-owned components.

## Architecture

This section provides a high-level architecture inspired by the current implementation work.

### Agent (Harness) Provider Abstraction

```python
class HarnessProvider(Protocol):
    async def stream_response(
        self,
        *,
        messages: list[dict],
        project: str | None,
        conversation_id: str | None,
        model: str | None,
    ) -> AsyncIterator[str]:
        ...

    ...


# Factory: instantiate provider by name
class HarnessProviderFactory:
    _providers: dict[str, type[HarnessProvider]] = {
        "openai": OpenAIHarnessProvider,
        "anthropic": AnthropicHarnessProvider,
    }

    @classmethod
    def create(cls, provider: str) -> HarnessProvider:
        try:
            provider_cls = cls._providers[provider]
        except KeyError:
            raise ValueError(f"Unsupported provider: {provider}")
        return provider_cls()
```

> Note: This is a minimal version of the abstraction, shown for simplicity.

### Concrete Agent Provider (Example)

```python
import anton


class AntonHarness(Protocol):
    async def stream_response(
        self,
        *,
        messages: list[dict],
        project: str | None,
        conversation_id: str | None,
        model: str | None,
    ) -> AsyncIterator[str]:
        # Example: will not run as-is
        anton.turn_stream(messages, project, conversation_id, model)

    ...
```

### Cowork Server API Mapping

```python
from uuid import UUID

import HarnessProviderFactory
import settings

# The agent is instantiated from configured settings.
# This can also be controlled at the endpoint level.
agent = HarnessProviderFactory.create(settings.agent)


@router.post("/responses")
async def create_response(req: ResponsesRequest):
    return agent.stream_response(
        req.messages, req.project, req.conversation_id, req.model
    )
```

Each abstraction method maps to a corresponding API endpoint.

The agent should act as a task executor, while concerns like projects and conversations remain application-level. In the current implementation, many of these are still tightly coupled with the harness.

---

## Current API Specification

The following captures the current Cowork server contract.

### Projects

- `GET|POST /v1/projects` - List/create (`CreateProjectRequest`).
- `GET|PUT /v1/projects/active` - Read/set active project (`SetActiveRequest`).
- `PATCH /v1/projects/{name}` - Rename (`RenameProjectRequest`).
- `DELETE /v1/projects/{name}` - Delete project.
- `GET /v1/projects/{name}/instructions` - `anton.md` metadata only (synthetic row if missing).
- `GET /v1/projects/{name}/files` - Recursive file listing (includes synthetic `anton.md`).
- `GET /v1/projects/{name}/files/{path:path}` - Read UTF-8 text (size cap).
- `PUT /v1/projects/{name}/files/{path:path}` - Write/replace (`FileWriteRequest`).
- `POST /v1/projects/{name}/files/upload` - Multipart files to project root.
- `DELETE /v1/projects/{name}/files/{path:path}` - Delete file.

### Conversations

- `GET /v1/conversations` - List conversations (limit, optional project; default from store, or `project=all`).
- `GET /v1/projects/{name}/conversations` - Same list scoped to one project.
- `GET /v1/conversations/{conversation_id}` - Conversation metadata.
- `PATCH /v1/conversations/{conversation_id}` - `ConversationPatch` (title, project move, etc.).
- `GET /v1/conversations/{conversation_id}/messages` - Messages with display merging + per-turn events.
- `DELETE /v1/conversations/{conversation_id}/turns/{turn_index}` - Remove one displayable user->assistant cycle.
- `DELETE /v1/conversations/{conversation_id}` - Delete conversation.

### Chat (Responses API)

- `POST /v1/responses` - OpenAI-style Responses API.

### Attachments

- `GET /v1/attachments/{project_name}/{session_id}` - List uploads (optional IDs filter).
- `POST /v1/attachments/{project_name}/{session_id}/upload` - Multipart upload.
- `DELETE /v1/attachments/{attachment_id}` - Remove from state and delete files when known.

### Artifacts

- `GET /v1/artifacts` - List artifact cards (optional `project_path`, capped).
- `GET /v1/artifacts/preview` - Path query text preview JSON (truncated content).
- `POST /v1/artifacts/preview-mount` - Register HTML parent for iframe; returns token and `relUrl`.
- `GET /v1/artifacts/preview-asset/{token}/{rel_path:path}` - Serve asset under mount.
- `POST /v1/artifacts/open` - Open file in OS default app.
- `POST /v1/artifacts/reveal` - Reveal in file manager.

### Scratchpads

- `POST /v1/scratchpad/start` - Create/start pad (`ScratchpadStartRequest`).
- `POST /v1/scratchpad/execute` - Run code; returns cell dict.
- `POST /v1/scratchpad/execute-stream` - SSE stream with progress/cell/error.
- `POST /v1/scratchpad/install` - Install packages on pad.
- `POST /v1/scratchpad/reset` - Reset pad.
- `POST /v1/scratchpad/cancel` - Cancel execution.
- `GET /v1/scratchpad/view` - Query by name (default `default`).
- `GET /v1/scratchpad/notebook` - Rendered notebook.
- `GET /v1/scratchpad/cells` - All cells as dicts.
- `POST /v1/scratchpad/close` - Close and remove from registry.
- `POST /v1/scratchpad/cleanup` - Cleanup and remove.
- `GET /v1/scratchpad/list` - List pad names.

### Pins

- `GET /v1/pins` - All pins.
- `POST /v1/pins` - Pin (`PinRequest`).
- `DELETE /v1/pins/{item_id}` - Unpin (`item_type` query, default `task`).
- `PUT /v1/pins/reorder` - Reorder (`ReorderRequest.item_ids`).
- `POST /v1/pins/{task_id}/visit` - Visit count; optional auto-pin after 3 visits.

### Data Sources

- `GET /v1/datasources` - Vault connections + engine registry.
- `GET /v1/datasources/{engine}/{name}` - Read connection for edit (secrets as keep sentinel).
- `POST /v1/datasources/validate` - Field validation / missing fields.
- `POST /v1/datasources` - Create/update (`DatasourceSaveRequest`).
- `DELETE /v1/datasources/{engine}/{name}` - Delete (may revoke Google OAuth).

### Connectors

- `GET /v1/connectors` - List connector summaries.
- `GET /v1/connectors/{connector_id}` - Full connector JSON spec.
- `POST /v1/connectors/match` - NL/id to ranked candidates (`MatchRequest`).
- `POST /v1/connectors/{connector_id}/save` - Save to vault (`SaveConnectorRequest`).

### Integrations

- `GET /v1/integrations` - Catalogue (Drive/Calendar/Gmail) + OAuth status.
- `POST /v1/integrations/google-drive/oauth/start` - Start Drive OAuth (PKCE).
- `GET /v1/integrations/google-drive/oauth/callback` - Drive callback (HTML).
- `POST /v1/integrations/google-calendar/oauth/start` - Start Calendar OAuth.
- `GET /v1/integrations/google-calendar/oauth/callback` - Calendar callback.
- `POST /v1/integrations/gmail/oauth/start` - Start Gmail OAuth.
- `GET /v1/integrations/gmail/oauth/callback` - Gmail callback.

### Data Vault

- `POST /v1/datavault/submissions` - Stage form values; SSE from agent (`SubmitFormRequest`).
- `GET /v1/datavault/submissions/{submission_id}` - Metadata + field names (values redacted).

### Settings

- `GET /v1/settings` - Merged env + UI prefs (API keys masked as `***`).
- `PUT /v1/settings` - Partial update (`SettingsPatch`) into `~/.anton/.env` + preferences.
- `POST /v1/settings/validate` - Returns `configReady/configError/provider/model`.

### Search

- `GET /v1/search` - Local Cowork search across resources (projects, artifacts, etc.).

### Browse

- `GET /v1/browse/status` - Probe Anton package for broad browse/search vs URL-only context.

### Schedules

- `GET /v1/schedules` - Schedules + `runs_index`.
- `POST /v1/schedules` - Create (`ScheduleRequest`).
- `PUT /v1/schedules/{schedule_id}` - Partial update.
- `DELETE /v1/schedules/{schedule_id}` - Delete.
- `POST /v1/schedules/{schedule_id}/pause` - Disable.
- `POST /v1/schedules/{schedule_id}/resume` - Enable.
- `POST /v1/schedules/{schedule_id}/run-now` - Manual run returns `{ schedule, session }`.
- `GET /v1/schedules/{schedule_id}/runs` - Run history (limit).

### Published Artifacts

- `GET /v1/publish` - Publishable HTML + readiness + history.
- `POST /v1/publish` - Publish HTML via Minds (`PublishRequest.path`).
- `DELETE /v1/publish` - Unpublish (path query).

### Memory

- `GET /v1/memory` - List markdown memory by scope (optional `project_path`).
- `POST /v1/memory` - Save memory file (`MemorySaveRequest`).
- `DELETE /v1/memory` - Delete by scope, `relative_path`, optional `project_path`.

### Skills

- `GET /v1/skills` - List skills (`Anton SkillStore`).
- `POST /v1/skills` - Create/update skill (`SkillSaveRequest`).
- `DELETE /v1/skills/{label}` - Delete skill.

---

## Proposed API Specification

Below is a more structured specification that preserves current functionality while clarifying ownership and endpoint design.

### Projects (App component)

- `GET|POST /v1/projects` - List/create projects.
- `PATCH /v1/projects/{project_id}` - Update project.
- `DELETE /v1/projects/{project_id}` - Delete project.

**What's changed**

- Removed `/v1/projects/active`; active project should be project metadata.
- Removed file-related project endpoints; project context should live in metadata.
- Uploading conversation-specific files is moved out of project files.

### Conversations (App component)

- `GET /v1/conversations` - List conversations (limit, optional project).
- `GET /v1/conversations/{conversation_id}` - Get conversation.
- `PATCH /v1/conversations/{conversation_id}` - Update conversation.
- `GET /v1/conversations/{conversation_id}/items` - Messages + per-turn events.
- `DELETE /v1/conversations/{conversation_id}/items/{message_id}` - Remove one user->assistant cycle.
- `DELETE /v1/conversations/{conversation_id}` - Delete conversation.

**What's changed**

- Removed `/v1/projects/{name}/conversations`; use `project` query param on `/v1/conversations`.
- Replaced turns index route with message ID route for clarity.

### Chat (Responses API) (App component with harness task execution)

- `POST /v1/responses` - OpenAI-style Responses API.

**What's changed**

This endpoint now accepts OpenAI-style file inputs. Files are uploaded first via Files API, then referenced by `file_id`.

```json
{
  "conversation": "44f07ad22d9c42dfb50f8e993180fc40",
  "model": "claude-opus-4-7",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "What is in this file?"
        },
        {
          "type": "input_file",
          "file_id": "d4aa798c-c145-46f1-bcf3-dd475bcc8709"
        }
      ]
    }
  ],
  "stream": "true"
}
```

Notes:

- File inputs become part of conversation history and can power chat file bubbles.
- Since Attachments are removed, conversation files for context cards should be derived from conversation history.

### Files (App component, replacement for Attachments)

- `POST /v1/files/` - Upload file (`multipart`: `file` + `purpose` fields).
- `GET /v1/files/` - List files (optional `purpose` filter).
- `GET /v1/files/{file_id}` - Retrieve file metadata.
- `DELETE /v1/files/{file_id}` - Delete file (`204` success, `404` not found).
- `GET /v1/files/{file_id}/content` - Download raw file content.

### Attachments

No longer needed. Uploads are handled through Files + Responses.

### Artifacts (Published Artifacts included) (App component)

TBD.

### Scratchpads

Currently, only `/v1/scratchpad/cancel` is actively used (stop button behavior while request is processing).  
Ideally, scratchpad cancellation should happen automatically when the UI disconnects the stream (already how Anton-backed Minds behaves).

These endpoints may become relevant again if scratchpad-as-a-service returns.

### Pins (App component)

- `GET /v1/pins` - All pins.
- `POST /v1/pins` - Pin (`PinRequest`: `item_type` in `task|project|artifact|schedule`, `item_id`, optional `title`).
- `DELETE /v1/pins/{item_id}` - Unpin (`item_type` query, default `task`).

**What's changed**

- Removed `/v1/pins/reorder` (not used).
- Removed `/v1/pins/{task_id}/visit` (auto-pin not actively used).

### Data Sources, Connectors, Integrations, and Data Vault (App component, Anton data vault aligned)

- `GET /v1/connectors/specs/` - List connector specs (metadata).
- `GET /v1/connectors/specs/{connector_id}` - Full spec (forms, fields, methods).
- `POST /v1/connectors/specs/match` - Match NL query to candidates (`MatchRequest`).
- `POST /v1/connectors/submissions/` - Submit form; probe + save credentials to vault (SSE).
- `GET /v1/connectors/connections/` - List saved connections enriched from specs.
- `GET /v1/connectors/connections/{engine}/{name}` - Read saved connection (secrets as `ANTON_VAULT_KEEP`).
- `DELETE /v1/connectors/connections/{engine}/{name}` - Delete saved connection.
- `POST /v1/connectors/oauth/{engine}/start` - Start PKCE OAuth; returns `auth_url`, `redirect_uri`, `started_at`.
- `GET /v1/connectors/oauth/{engine}/callback` - OAuth callback; exchange code, save tokens, return HTML.

**What's changed**

- Consolidated connector-related endpoints under `connectors`.
- Converted legacy `/connectors` routes into `/connectors/specs`.
- Converted `/datavault/submissions` to `/connectors/submissions`.
- Converted `/datasources` connection endpoints to `/v1/connectors/connections`.
- Standardized OAuth under `/connectors/oauth/{engine}`.

### Settings (App component)

Settings are now database-backed rather than `.env`-backed. Sensitive values (API keys, etc.) are encrypted at rest and not exposed in endpoint responses.

- `GET /v1/settings` - List all settings.
- `PUT /v1/settings/{key}` - Create/update setting by key (`value` in body).
- `DELETE /v1/settings/{key}` - Delete setting by key.

**What's changed**

- Settings moved from `.env` to DB to support sharing across harnesses.
- Sensitive values are hidden; use `is_set=true` with `value=null`.

### Search (App component)

Retained if cross-resource search remains a product requirement.

- `GET /v1/search` - Local Cowork search across resources.

### Browse

Currently not in use.

### Schedules (App component)

No changes.

- `GET /v1/schedules` - Schedules + `runs_index`.
- `POST /v1/schedules` - Create (`ScheduleRequest`).
- `PUT /v1/schedules/{schedule_id}` - Partial update.
- `DELETE /v1/schedules/{schedule_id}` - Delete.
- `POST /v1/schedules/{schedule_id}/pause` - Disable.
- `POST /v1/schedules/{schedule_id}/resume` - Enable.
- `POST /v1/schedules/{schedule_id}/run-now` - Manual run returns `{ schedule, session }`.
- `GET /v1/schedules/{schedule_id}/runs` - Run history (limit).

### Memory (Harness component)

Memory is not currently shared across harnesses. Each harness manages its own memory model/scopes (for example, Hermes does not support project-level memory).

The response model changes to:

```python
class MemoryResponse(BaseModel):
    scope: MemoryScope
    category: str
    content: str
    project_id: UUID | None = None

    @model_validator(mode="before")
    def validate_project_id(cls, values):
        return validate_project_id(values)
```

Note: filename is not included, but can be added if needed.

- `GET /v1/memory/` - List all memory across scopes/projects.
- `PUT /v1/memory/` - Overwrite memory by scope/category/optional project.
- `DELETE /v1/memory` - Delete memory by scope/category/optional project.

### Skills (Harness component)

- `GET /v1/skills` - List all skills.
- `POST /v1/skills` - Create a new skill.
- `GET /v1/skills/{skill_id}` - Get skill by ID.
- `PUT /v1/skills/{skill_id}` - Update skill by ID.
- `DELETE /v1/skills/{skill_id}` - Delete skill by ID.

**What's changed**

- Added `GET /v1/skills/{skill_id}` and `DELETE /v1/skills/{skill_id}` for full CRUD.
- Operations now use IDs instead of labels.

---

## Remaining Work / Improvements

### To Be Done

#### Front-end Integration

- Integrate front-end with the new server version.
- Current owner: Paul Newsam.
- PRs:
  - [mindsdb/cowork-server#1](https://github.com/mindsdb/cowork-server/pull/1)
  - [mindsdb/cowork#102](https://github.com/mindsdb/cowork/pull/102)
- Timeline: next couple of days.

#### Artifacts

- Port Artifacts in a harness-agnostic way.
- Suggested owners: Max Stepanov or Jorge Torres.
- Without this, artifacts are not available in Cowork app on this implementation.
- Need Cowork-level decisions (publishing + GUI surfacing).

#### OAuth Connectors

- Test OAuth connectors end-to-end.
- Should be heavily tested once front-end integration lands.
- Potential regression risk due to substantial recent updates.
- Mentioned owner/context: Martyna Slawinska.

#### Harness-specific Submission Flows for Connections

- `/connectors/submissions` currently runs through Anton regardless of configured harness.
- Functionally okay (credentials still land in data vault), but potentially misleading UX.
- Known limitation to address.

### Improvements

#### Channels

- Port Channels to Cowork server in a harness-agnostic way.
- Suggested owner: Konstantin Sivakov.

#### Shared Memory for Harnesses

- Memory is intentionally harness-local for now.
- Could be revisited in future.

#### DB-backed Data Vault

- LocalDataVault currently stores credentials in files.
- A DB-backed vault would be more robust and improve encryption/security support.
- SQLite is available but not yet used for Data Vault.

#### Understanding Hermes Code Execution

- Better understand Hermes execution behavior:
  - Environment creation/management
  - Session persistence across executions
  - Credential access patterns for data sources/apps
- Status: TBD (knowledge and architecture follow-up).

