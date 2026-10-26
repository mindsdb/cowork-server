# Docs: structured document artifacts

A doc is a living document the agent and the person write together: tabs of typed blocks with stable ids, modelled on Claude's Docs artifacts.

The code is in `cowork/docs/`.

The modules depend in one direction, each on those above it:

| Module | What it holds |
| --- | --- |
| `model` | The doc shape, validation and limits |
| `markdown` | The Markdown dialect the agent writes, both ways |
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

- `table`: columns of kind `text`, `number`, `status` or `date`, and rows whose cells are keyed by column id. A date column's cells are ISO (`YYYY-MM-DD`): a whole date written out ("November 4, 2026") is stored as ISO wherever it comes from (`model.to_iso_date`, the editor's `toIsoDate`), so the column sorts by date and the editor's date picker reads it.
- `chart`: `bar`, `hbar`, `stacked`, `line`, `area` or `donut`, drawn from a table `source` or from inline `x`/`series`.
- `image`: `src` is a web address, a `data:` image or a file beside the doc (`images/…`), never a path elsewhere on the machine.
- `pending`: a placeholder the agent fills later.
- Inline text is the Markdown subset the agent writes, plus the chips `{{date:YYYY-MM-DD}}` and `{{status:Label}}`.

Each block's `v` is the rev that last changed it, and `by` who changed it.

## Working on it

Tests, in `tests/`:

- `test_docs_format.py`: the model, Markdown and rendering.
