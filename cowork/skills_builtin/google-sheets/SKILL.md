---
name: google-sheets
description: >
  Use this skill when the user wants to create, find, read, analyze, edit,
  clean up, summarize, or repair Google Sheets. Use it for spreadsheet search,
  tab and range inspection, row lookup, formulas, charts, table cleanup, data
  restructuring, validation-aware edits, imports, and precise cell or range
  updates.
---

# Google Sheets

Use this skill to keep spreadsheet work grounded in the exact spreadsheet, tab,
range, headers, formulas, and validation rules that matter.

## Task routing

Classify the request:

- FIND: locate a spreadsheet by title, owner, content, or link.
- READ: inspect metadata, tabs, ranges, rows, columns, formulas, or charts.
- ANALYZE: summarize data, group records, calculate totals, find trends, or
  answer questions.
- EDIT: update cells, rows, columns, formatting, formulas, validation, or notes.
- CLEAN: normalize headers, remove duplicates, split fields, trim spaces, or
  restructure tables.
- FORMULA: create, explain, repair, or roll out formulas.
- CHART: create, repair, refresh, move, or explain charts.
- CREATE: build a new spreadsheet or structured table.
- IMPORT: convert spreadsheet-like data into a native sheet when supported.

## Grounding rules

Before editing:

- confirm the exact spreadsheet
- confirm the target tab
- use exact visible tab names
- use bounded ranges
- read headers before interpreting data
- read current formulas before overwriting cells
- identify protected ranges, filters, hidden rows, merged cells, frozen panes,
  data validation, and multiple tables on one tab when relevant

Do not guess `Sheet1` unless metadata confirms it.

## Canonical workflow

Prefer a simple verified workflow:

1. Gather the required source material.
2. Identify the target spreadsheet, tab, and range.
3. Read the relevant current state.
4. Establish the sheet checklist or edit plan.
5. Build or edit the sheet.
6. Verify the sheet is clean, complete, and scannable.
7. Stop once the verified workflow has succeeded.

Do not turn straightforward sheet edits into a long detour through speculative
fallbacks.

## Reading and search safety

- Search narrowly before scanning large sheets.
- Avoid whole-grid reads unless needed.
- Use bounded ranges for live reads.
- When searching rows, use headers to interpret results.
- If a search result is partial, say what scope was searched.
- If multiple tabs could contain the answer, inspect metadata and tab names
  before choosing.

## Editing rules

- Prefer precise range updates over broad sheet-wide changes.
- Re-read target cells before writing when live values, formulas, formatting, or
  validation could affect the edit.
- Preserve formulas unless the user wants static values.
- Preserve headers and table shape unless the user asks to restructure.
- When adding rows, match existing column order and formatting.
- When cleaning data, avoid destroying original meaning.
- Check for accidental changes outside the requested scope.

## Formula work

For formulas:

- identify the row or column pattern before filling down or across
- use absolute and relative references intentionally
- check locale-specific separators if formula syntax fails
- verify outputs after applying formulas
- avoid overwriting formulas with values unless requested
- explain formula behavior in plain language when the user asks

Formula examples should be adapted to the actual tab and range, not pasted as
generic placeholders.

## Charts

For charts:

- confirm source data range
- choose chart type by question
- line charts for time series
- bar or column charts for comparisons
- scatter charts for relationships
- stacked bars for category composition
- pie charts only for simple parts-of-a-whole
- verify labels, legends, axes, units, and title
- ensure the chart stays connected to source data if it should update later

## Cleaning and restructuring tables

When cleaning:

- identify the header row
- preserve original columns unless removing them is requested
- standardize dates, names, casing, whitespace, and category labels carefully
- remove duplicates only after defining duplicate criteria
- split columns only when delimiters are reliable
- keep a backup or reversible path when changes are large

## Styling and scannability

If the user asks for a polished sheet:

- freeze header rows when useful
- bold or style headers
- apply sensible column widths
- align numbers, dates, and text appropriately
- use number formats for currency, percentages, and dates
- keep formatting restrained and functional
- avoid visual clutter

## Final answer requirements

When returning results:

- cite the spreadsheet, tab, and range used when relevant
- distinguish direct observations from calculations or assumptions
- summarize important edits and checks
- mention limitations if the data was incomplete, filtered, protected, messy, or
  ambiguous

## Final checks

Before finishing:

1. Confirm the correct spreadsheet, tab, and range were used.
2. Verify edited cells, formulas, tables, formatting, or charts match the request.
3. Check for accidental changes outside the requested scope.
4. Confirm formulas calculate as expected.
5. Confirm charts point to the intended data.
6. Mention any unresolved ambiguity or access limitation.
