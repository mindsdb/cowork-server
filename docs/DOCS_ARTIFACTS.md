# Docs: structured document artifacts

A doc is a living document the agent and the person write together: tabs of typed blocks with stable ids, modelled on Claude's Docs artifacts.

The code is in `cowork/docs/`.

The modules depend in one direction, each on those above it:

| Module | What it holds |
| --- | --- |
| `model` | The doc shape, validation and limits |
| `markdown` | The Markdown dialect the agent writes, both ways |
| `ops` | The atomic edit operations |
| `undo` | What puts back one of the person's writes |
| `render` | The agent's outline and full views, the page (`doc.html`) and export markup |
| `store` | The doc file in its artifact folder: the lock, atomic writes, one kept version per editing session |
| `hub` | In-process fan-out of changes to the editors and chats watching |
| `service` | Create and apply (rendering the page under the doc's lock), and `announce` |

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

- `table`: columns of kind `text`, `number`, `status` or `date`, and rows whose cells are keyed by column id. A date column's cells are ISO (`YYYY-MM-DD`): a whole date written out ("November 4, 2026") is stored as ISO wherever it comes from (`model.to_iso_date`, the editor's `toIsoDate`), so the column sorts by date and the editor's date picker reads it.
- `chart`: `bar`, `hbar`, `stacked`, `line`, `area` or `donut`, drawn from a table `source` or from inline `x`/`series`.
- `image`: `src` is a web address, a `data:` image or a file beside the doc (`images/…`), never a path elsewhere on the machine.
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

The person's ops (`by: "user"`) are taken as sent: an empty text empties a block. The agent's are sifted for the blanks models fill into every field they are offered. A chart written with its own copy of a table's numbers is pointed at that table (`ops.link_inline_charts`), so the numbers live in one place.

`undo.undo_ops` gives the ops that put back what one write changed, from the doc just before and after it, and nothing else: another writer's change in between is never part of them. An undo is a write like any other, so its own undo is the redo.

## Storage

A doc is an ordinary anton artifact folder:

```
<project>/.anton/artifacts/<slug>/
  metadata.json            type "document", primary "doc.cdoc.json"
  doc.cdoc.json            the doc (schema cowork.doc/1)
  doc.html                 its page, rendered after every write
  .revisions/doc-history/  kept versions, index.json, and open.json (the session writing now)
```

The artifacts library, provenance and delete therefore work unchanged.

## One write path

Both the agent and the editor apply the same ops through `cowork.docs.service.apply`. That function:

- serialises writers of one doc on its lock, which holds across processes too (the artifact's `.revisions/.lock`), so two replicas serving one shared tree never write the same rev;
- keeps one version per editing session: when a session ends, the doc as it left it is kept. A session is one reply of the agent's (`turn`, which the doc tools name per turn), or a stretch of a person's typing that ends after a five-minute pause, after thirty minutes, or when the agent writes. The newest 100 are kept, and only as many as fit in 50 MB, always at least one;
- re-renders `doc.html`, the page a doc is shared as, still under the lock;
- for the person's writes, answers `undo`, the ops that put that write back, worked out under the same lock.

Callers then announce the change (`service.announce`) on the in-process hub (`cowork.docs.hub`) to the editors and chats watching.

Every write lands at once, like a Google Doc's: there is no save step and no versions to manage.

## Working on it

Tests, in `tests/`:

- `test_docs_format.py`: the model, Markdown and rendering.
- `test_docs_ops.py`: the edit operations and the stale guard.
- `test_docs_undo.py`: undoing and redoing each kind of edit.
- `test_docs_store.py`: the write path, editing sessions and their versions, and the hub.
