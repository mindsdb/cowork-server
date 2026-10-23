# Docs: structured document artifacts

A doc is a living document the agent and the person write together: tabs of typed blocks with stable ids, modelled on Claude's Docs artifacts.

The code is in `cowork/docs/`.

The modules depend in one direction, each on those above it:

| Module | What it holds |
| --- | --- |
| `model` | The doc shape, validation and limits |
| `markdown` | The Markdown dialect the agent writes, both ways |
| `ops` | The atomic edit operations |
| `render` | The agent's outline and full views, the page (`doc.html`) and export markup |

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

## Working on it

Tests, in `tests/`:

- `test_docs_format.py`: the model, Markdown and rendering.
- `test_docs_ops.py`: the edit operations and the stale guard.
