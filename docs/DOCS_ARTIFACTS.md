# Docs: structured document artifacts

A doc is a living document the agent and the person write together. It opens beside the chat. The agent writes it section by section while the person watches, and the person types in it, picks statuses in its tables, and selects a passage to ask the agent to change. It is modelled on Claude's Docs artifacts.

The code is in `cowork/docs/`, the agent tools are in `cowork/harnesses/anton_harness/doc_tools.py`, and the HTTP API is in `cowork/api/v1/endpoints/docs.py`. The renderer side lives in the cowork repo under `src/renderer/cowork/components/doc/`.

The modules depend in one direction, each on those above it:

| Module | What it holds |
| --- | --- |
| `model` | The doc shape, validation and limits |
| `markdown` | The Markdown dialect the agent writes, both ways |
| `ops` | The atomic edit operations |
| `render` | The agent's outline and full views, the page (`doc.html`) and export markup |
| `store` | The doc file in its artifact folder: the lock, atomic writes, one kept version per editing session |
| `hub` | In-process fan-out of changes to the editors and chats watching |
| `sharing` | Which docs are shared |
| `service` | Create and apply (rendering the page under the doc's lock), and `announce` |
| `republish` | A shared page following its edits |
| `sync` | What follows a write (announce, republish), and catching shared docs up at startup |
| `prompt`, `export` | What the agent is told; downloads |

The package uses the services; they never import its behaviour, and know a doc only by its files (`store`'s names).

## The format

```json
{"schema": "cowork.doc/1", "id": "<artifact id>", "title": "…", "rev": 7,
 "tabs": [{"id": "t_…", "title": "…", "blocks": [{"id": "b_…", "type": "paragraph", "v": 7, "by": "agent", "text": "…"}]}]}
```

Block types:

| Group | Types |
| --- | --- |
| Text | `paragraph`, `heading` (`level` 1–3), `bullet`, `numbered`, `task` (`done`), `quote`, `callout` (`tone`, `title`) |
| Code | `code` |
| Data | `table`, `chart`, `metrics` |
| Other | `image`, `divider`, `pending` |

- `table`: columns of kind `text`, `number`, `status` or `date`, and rows whose cells are keyed by column id.
- `chart`: `bar`, `hbar`, `stacked`, `line`, `area` or `donut`, drawn from a table `source` or from inline `x`/`series`.
- `pending`: a placeholder the agent fills later.
- Inline text is the Markdown subset the agent writes, plus the chips `{{date:YYYY-MM-DD}}` and `{{status:Label}}`.

Each block's `v` is the rev that last changed it, and `by` who changed it.

## Edit operations

`ops.apply_ops` is atomic: every op lands or none does. A refusal names the op by index and says how to fix it.

| Ops | Applies to |
| --- | --- |
| `insert`, `replace`, `update`, `edit_text`, `delete`, `move` | Blocks |
| `update_cell`, `add_rows`, `delete_rows`, `add_column`, `update_column`, `delete_column`, `sort_rows` | Tables |
| `add_tab`, `rename_tab`, `delete_tab`, `move_tab`, `rename_doc` | Tabs and the doc |

Positions are `start`, `end`, `{after}`, `{before}` and `{end_of_section: heading id}`.

An edit that passes `if_rev` is refused as `stale` when a block it touches moved on after that rev, unless the change was made earlier in the same call. That rule lets an agent edit and a person's typing coexist without silent overwrites.

## Storage

A doc is an ordinary anton artifact folder:

```
<project>/.anton/artifacts/<slug>/
  metadata.json            type "document", primary "doc.cdoc.json"
  doc.cdoc.json            the doc (schema cowork.doc/1)
  doc.html                 its page, rendered after every write (what sharing publishes)
  .revisions/doc-history/  kept versions, index.json, and open.json (the session writing now)
```

The artifacts library, cards, provenance and delete therefore work unchanged. `card_for_folder` marks the card `kind: "Doc"` and `format: "doc"`, which is how the renderer knows to open the doc editor.

## One write path

Both the agent and the editor apply the same ops through `cowork.docs.service.apply`. That function:

- serialises writers of one doc on its lock, which holds across processes too (the artifact's `.revisions/.lock`), so two replicas serving one shared tree never write the same rev;
- keeps one version per editing session: when a session ends, the doc as it left it is kept. A session is one reply of the agent's (`turn`, which the doc tools name per turn), or a stretch of a person's typing that ends after a five-minute pause, after thirty minutes, or when the agent writes. The newest 100 are kept, and only as many as fit in 50 MB, always at least one;
- re-renders `doc.html`, the page a doc is shared as, still under the lock.

Then `cowork.docs.sync.after_write` announces the change on the in-process hub (`cowork.docs.hub`) to the editors and chats watching, and for a shared doc schedules its republish.

Every write lands at once, like a Google Doc's: there is no save step and no versions to manage.

## HTTP API

Everything is under `/api/v1/docs` and addressed like the artifact workspace routes. Reads need review access; writes need `artifact.manage` and ownership.

| Method | Path | What |
| --- | --- | --- |
| POST | `/` | Create a doc the person starts: `{projectId, title, markdown?}` |
| GET | `/{project}/{artifact}` | `{doc, card, capabilities, originConversationId}`; the card (with the share's access) for the owner only |
| POST | `/{project}/{artifact}/ops` | `{ops, tab?, ifRev?, clientId?}` → `{doc, rev, results, touched, removed}`; 409 for `stale` |
| GET | `/{project}/{artifact}/events` | SSE `doc.updated` with the doc. Also catches writes from other processes by watching the file. |
| GET | `/{project}/{artifact}/export?format=md\|html\|docx\|pdf` | Download. PDF and Word are refused where several people share the server: their converters fetch the doc's web images from it. |
| GET | `/events?conversation_id=` | SSE `doc.created` / `doc.updated` for docs the agent touches in that conversation |

`POST /api/v1/responses` takes three optional fields:

- `docs: true`: the client shows docs, so the agent may write them. Without it (or one of the two below, which imply it) the turn has no doc tools.
- `doc_context`: the doc, tab and selection open beside the chat, plus an "Edit with agent" `request`. It is handed to the agent as a `<doc-view-context>` block ahead of the message.
- `output: "doc"`: the composer's "Write it as a doc" choice.

## Agent tools

`create_doc`, `read_doc`, `edit_doc` and `list_docs` are registered for in-process turns (desktop and local) whose client shows docs (`docs: true`). `create_doc` carries the system-prompt section (`cowork/docs/prompt.py`: `DOC_PROMPT`), which covers:

- when to write a doc;
- the outline-then-fill workflow with `{{pending: …}}` blocks;
- the Markdown dialect: tables, `chart` and `metrics` fences, callouts and chips;
- how to shape content;
- how to edit safely with `if_rev` and the doc view context.

Each handler streams one progress line, for example "Writing “Risks”", so the call shows as a step in the chat. Results are plain text built for a model: an outline of block ids, what was written, and the pending sections still to fill.

Turns that run on a remote worker (org mode) do not get these tools yet, so they are not handed `doc_context` or the Output: Doc note either: such a turn is answered like any other. Moving the engine and tools into anton and adding them to the cloud allowlist is the follow-up that brings docs there.

## A shared doc follows its edits

Sharing a doc publishes its page (`doc.html`) through the normal publish flow and access. From then on `cowork.docs.republish` republishes it four seconds after the last write, keeping its URL and access, so there is no "update the shared copy" step; a burst of writes (a person typing, the agent filling sections) is one republish. That is not done in org mode, where autopublish owns what a shared artifact shows. At startup (`sync.start`) each shared doc's page is re-rendered and republished if it differs from what went out, which catches an edit made just before the app quit.

## Working on it

Tests, in `tests/`:

- `test_docs_format.py`: the model, Markdown and rendering.
- `test_docs_ops.py`: the edit operations and the stale guard.
- `test_docs_store.py`: the write path, editing sessions and their versions, and the hub.
- `test_docs_endpoints.py`: the HTTP API.
- `test_docs_tools.py`: the agent's tools.
- `test_docs_sharing.py`: which docs are shared, catching them up at startup, and the debouncer.

By hand: run the app against a server with a throwaway `COWORK_HOME` (never your own `~/.cowork`). The editor's tests sit beside it in the cowork repo, under `src/renderer/cowork/components/doc/`.

Model-in-the-loop check: give a model only `DOC_PROMPT` and the tool schemas, and let it call the real handlers.
