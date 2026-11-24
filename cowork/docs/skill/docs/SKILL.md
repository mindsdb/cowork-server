---
name: docs
description: 'MANDATORY before creating, reading or changing a doc. Loads the doc tools and the rules for writing and changing a doc.'
metadata:
  display_name: Write and edit docs
  provenance: host
---
How to write a new doc:
1. Call `create_doc` first, before any research, with the title and an outline:
   - the lead sentence, if you already know the answer, and any short line that belongs under it (an as-of date, who was there);
   - then one pending line per section, e.g. `{{pending: Risks — what could slip and the fallback}}`. The words are yours; only `{{pending: …}}` is syntax.
   The user sees the plan at once.
2. Fill one section per `edit_doc` call, in reading order. Each call is a `replace` whose target is that section's pending block id, with Markdown that starts `## <Heading>`. Research what a section needs just before you write it, not all up front.
3. In chat, end with one or two lines that name the doc and say what the user can do with it. Never repeat the doc's content in chat.

Doc Markdown, used in `create_doc`, `insert` and `replace`:
- Text: `#`/`##`/`###` headings, paragraphs, `-` bullets, `1.` numbered items, `- [ ]` / `- [x]` tasks. Indent 2 spaces to nest.
- Inline: `**bold**`, `*italic*`, `` `code` ``, `[text](https://…)` links, and `{{date:2026-10-21}}` for a date someone may act on (a deadline, a due date, a next step). It works in any text: paragraphs, list items, tasks, callouts and table cells. In a date column, write plain `YYYY-MM-DD` instead; the column draws it as a date.
- `> [!NOTE]` / `> [!TIP]` / `> [!WARNING]` / `> [!CAUTION]` callouts, with an optional title after the marker and the body on the next `>` lines. A `> plain quote` is a quote. `---` is a divider.
- Fenced code with a language. Images are `![alt](url "caption")` on their own line: a web address, or the full path of an image the user attached or one in the project, which is copied into the doc.
- Pipe tables. A column of numbers right-aligns. A column of `YYYY-MM-DD` values is a date column. A column of short labels named Status, Stage, Priority, Risk, Decision, Health, Result or Phase becomes a dropdown; its options are the values used. To offer more options than the rows use (say, every pipeline stage), follow up with `update_column {options: [...]}`. Tables hold values, not formulas: compute totals yourself and write the numbers.
- Charts are a fenced `chart` block holding JSON:
  - drawn from a table above it: `{"type":"bar","title":"Revenue by quarter","unit":"$M","source":{"table":"previous","x":"Quarter","y":["Revenue","Cost"]}}`. `"previous"` is the nearest table above the chart; you can also give a table block id.
  - or with inline data: `{"type":"line","title":"…","x":["Jan","Feb"],"series":[{"name":"Signups","values":[120,180]}]}`.
  - Types: bar, hbar (long labels), stacked (parts that sum to a whole), line (a level over time), area, donut (parts of one whole, 6 or fewer).
  - Inline data holds up to 8 series.
  - Prefer drawing a chart from its table, so the numbers live in one place and the user's edits flow through.
- Headline numbers are a fenced `metrics` block holding a JSON list (up to 8 tiles): `[{"label":"ARR","value":"$4.2M","delta":"+12% QoQ","tone":"good"}]`. `tone` is `good` or `bad` and colours the delta. Leave it out for a neutral delta.

Shaping the content:
- The first sentence of the doc and of each section states its point: the decision, the result, or the answer with its number or date. Never write "This document outlines…".
- Items compared on several attributes (options, vendors, people, line items, tasks with owners and dates) go in a table: one row per item, units in the header, rows sorted by what readers scan for.
- Numbers whose shape is the point (a trend, a ranking, parts of a whole), usually five or more, go in a chart, with the numbers in its source table. Four or fewer read fine as a table or a metrics row.
- Data the user hands you (an attached file, a pasted table) that the doc is about goes in the doc as a table of its own rows, every one of them when there are up to about 30, so readers can check each figure. A table of totals, averages or a quarter-by-quarter rollup is not a substitute: add it beside the rows if it helps. Draw a chart from the rows when the numbers have a shape, and write the summary above them. Past about 30 rows, put the full table in its own tab and keep the rows the summary relies on beside it.
- Steps go in a numbered list. A risk or caveat that must not be missed goes in a callout.
- Action items go in tasks people can tick: `- [ ] Marco books the security review — due {{date:2026-10-24}}`. When there are many of them, or people need to sort them by owner or status, use a tracker table instead (Task, Owner, Due, Status).
- A fact the user's material does not give is an open question. Say it in one line where it matters, or ask in chat. Never invent it.
- Write sentences under 25 words and paragraphs of three sentences or fewer. Use specifics (numbers with units, names, dates), not adjectives. No emoji. No "key takeaways" or recap sections. Size the doc to the ask.
- Facts you looked up are cited in the doc with `[source](url)` links, or in a Sources section at the end.
- Give material with its own bulk (long raw data, sources, appendices) its own tab with `add_tab`. Otherwise one tab is right.

Changing a doc:
- The person may have edited the doc since you last saw it. Read what you will change first, unless your own last call returned its current rev: `read_doc` with the `ids` of those blocks, or the whole tab when the doc is short. View "outline" lists the block ids. Pass that rev as `if_rev` to `edit_doc`. If the result is `stale`, the person changed a block you targeted: read it again, keep their change, and resend.
- Change only what was asked; every other word stays. If your change makes another figure in the doc wrong (a total, the lead's number, a summary tile), fix that too and name it in your chat reply.
  - For a word or figure, use `edit_text` with `find` quoted exactly as the doc has it.
  - For a whole block or section, `replace` its block ids. Replacing one block with one block of the same kind keeps its id, so a chart drawn from a rewritten table still points at it.
  - For a new block, `insert` with a `position`.
  - For one table cell, use `update_cell`. For rows and columns, use `add_rows` / `delete_rows` / `add_column` / `update_column` / `sort_rows`.
  - To tick a task, `update` it with `set: {"done": true}`.
