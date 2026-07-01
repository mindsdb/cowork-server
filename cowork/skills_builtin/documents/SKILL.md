---
name: documents
description: >
  Use this skill when the user wants to create, edit, review, redline, comment
  on, polish, or verify structured document deliverables. Use it for Word or
  Google Docs-style reports, memos, briefs, proposals, letters, handbooks,
  forms, templates, redlines, comment-based revisions, accessibility cleanup,
  metadata cleanup, document merging, style normalization, watermarks,
  footnotes, tables of contents, captions, cross-references, and final layout
  QA where readability and formatting matter.
---

# Documents Skill: Read, Create, Edit, Redline, Comment

Use this skill for serious document work where the final result must be
readable, structured, professionally formatted, and faithful to the user's
request.

This skill covers document artifacts such as Word documents, Google
Docs-targeted documents, formal reports, memos, proposals, SOPs, briefs,
forms, questionnaires, handbooks, templates, and review drafts.

## Core contract

- Deliver only the requested final document unless the user asks for drafts,
  previews, QA images, or intermediate files.
- Preserve the original document when editing an existing file. Work from a
  copy if the app supports it.
- Do not invent facts or make factual or technical errors just to make the
  document look polished.
- Treat layout QA as part of the job, not an optional finishing touch.
- If a visual or structural check cannot be completed because the app lacks
  the needed capability, say that clearly.

## Start by classifying the task

First identify the user's real task:

- READ: summarize, review, extract, critique, or answer questions.
- CREATE: make a new document from source material or instructions.
- EDIT: revise an existing document while preserving its structure.
- REDLINE: propose tracked-style edits or marked changes.
- COMMENT: add review feedback or margin-style notes.
- FINALIZE: remove comments, accept/reject changes, clean metadata, or prepare
  a clean copy.
- MERGE: combine multiple documents while preserving order and structure.
- TEMPLATE: create or fill a reusable document with fields and placeholders.
- FORM: create or edit a questionnaire, intake form, checklist, or controlled
  response document.
- ACCESSIBILITY: improve headings, alt text, links, tables, and reading order.
- PRIVACY: scrub personal metadata, hidden comments, identifiers, or sensitive
  text.

When the request contains multiple tasks, do them in a sensible order:
read/inspect -> plan -> edit/create -> verify -> final cleanup.

## Visual verification gate

You do not know a document is satisfactory until you inspect how it renders.
Text extraction and structural reads can miss layout defects.

For any new document, major rewrite, table-heavy document, form, proposal,
report, or layout-sensitive edit:

1. Render or preview the document pages if the app provides that capability.
2. Inspect every page, not only the first page.
3. Look for clipping, overlap, missing glyphs, broken tables, awkward spacing,
   orphan headings, bad page breaks, and header/footer mistakes.
4. Fix defects and inspect again.
5. Deliver only after the latest inspected version is clean.

If page rendering is unavailable, do a structural QA pass instead:

- check headings and outline order
- check table widths and row readability
- check lists and numbering
- check images and captions
- check headers, footers, page numbers, and section breaks
- disclose that visual page rendering was not available

## Design preset contract

For new documents and major rewrites, choose a document design preset before
drafting. For existing-document edits, preserve the source document and apply
minimal local changes unless the user asks for a redesign.

Choose one:

- GOOGLE DOCS DEFAULT: native-feeling, simple, Arial-like typography, black
  hierarchy, minimal decoration, clean title treatment.
- STANDARD BUSINESS BRIEF: formal memos, RFI responses, decision memos, board
  briefs, executive notes.
- COMPACT REFERENCE GUIDE: launch guides, negotiation briefs, checklists,
  operator guides, dense reference material.
- NARRATIVE PROPOSAL: grants, proposals, persuasive documents, longer prose,
  stakeholder-facing materials.
- FORM OR QUESTIONNAIRE: response fields, choices, scales, check targets, and
  clear completion flow.
- HANDBOOK OR MANUAL: repeatable sections, procedures, examples, tables,
  callouts, and navigation.

Resolve the preset into concrete choices before drafting:

- page size and margins
- title style
- heading ladder
- body text size and line spacing
- paragraph spacing
- list indentation
- table width and padding
- callout treatment
- header/footer behavior
- accent color use

Do not mix visual systems. Once a style is chosen, apply it consistently.

## Google Docs-targeted documents

When the intended destination is a native Google Doc:

- Use a simple native-feeling document style.
- Prefer black headings and clean spacing over decorative Word-style effects.
- Avoid title underlines, horizontal rules, ornate borders, and heavy first-page
  furniture unless the user explicitly wants a designed document.
- Keep tables simple and editable.
- Keep styles easy for a user to continue editing inside Google Docs.
- Do not rely on Word-only visual effects that may import badly.

## Form factor selection

Choose content form factors deliberately. Start from the information type, then
pick the lightest readable structure that helps the reader understand, compare,
act, or fill in information.

Use:

- PROSE SECTION for narrative, background, explanation, or rationale.
- LEAD CALLOUT for decisions, recommendations, key takeaways, or executive
  emphasis.
- NUMBERED STEPS for sequence, workflow, SOPs, procedures, or instructions.
- GROUPED BULLETS for loose factors, requirements, pros and cons, or
  considerations.
- CHECKLIST for actions, acceptance checks, review criteria, or completion
  tasks.
- NOTE BOX for warnings, caveats, constraints, or important reminders.
- DEFINITION LIST for terms, metadata, key facts, roles, or short labels.
- TABLE for repeated comparable records with shared fields.
- FORM LAYOUT for questionnaires, intake forms, evaluations, and response
  fields.
- SOURCE LIST for citations, evidence, references, appendices, and source
  material.

Do not use a table just to package ordinary prose. If cells become
mini-paragraphs, use prose sections, bullets, steps, callouts, or appendices.

## Table gate

Use a table only when the content is truly row and column data:

- repeated items
- shared fields
- status grids
- budgets
- schedules
- requirements matrices
- compliance matrices
- comparison tables
- form response grids

Before finalizing tables:

- Check whether the table is actually easier to read than prose.
- Avoid adjacent tables unless there is a clear reason.
- Convert paragraph-heavy cells to prose, bullets, or labeled sections.
- Keep captions visually paired with their tables.
- Keep table headers clear and repeated on continuation pages when possible.

## Table quality rules

Invest real care in tables. Bad tables are one of the easiest ways to make a
document feel broken.

Use these rules:

- Set deliberate column widths; do not default to equal-width columns.
- Compact short fields such as number, date, status, score, checkbox, result,
  year, or owner.
- Reserve wider columns for narrative or multi-line content.
- Avoid overly wide tables when a narrower table reads better.
- Use generous cell padding.
- Allow rows to expand; do not use fixed row heights that can clip text.
- Use vertical alignment intentionally, usually middle alignment for compact
  values.
- Use horizontal alignment by column type: centered for short values, left for
  narrative text.
- Keep line spacing inside cells comfortable.
- Prefer wrapping and column-width adjustments before shrinking text.
- Keep spacing before and after tables so they do not feel stuck to surrounding
  paragraphs.
- Check for text pressed against cell borders or pinned to the upper-left.

If a table spans pages, keep column order consistent and repeat headers when
possible.

## Forms and questionnaires

Design forms as usable documents, not as dense spreadsheets.

For forms:

- Use clear section headings.
- Make response targets obvious.
- Provide enough space for answers.
- Keep checkboxes, scales, and options readable.
- Avoid cramped full-grid borders.
- Size fields and columns based on the content they hold.
- Use subtle structure and spacing instead of heavy lines everywhere.
- Make instructions short and located near the fields they explain.

## Document design standards

Before creating a new document, think through the high-level design:

- What kind of document is this: memo, report, SOP, workflow, form, proposal,
  letter, handbook, or manual?
- Who is the reader?
- What does the reader need to do after reading?
- What belongs on the first page?
- What belongs in front matter, body, appendices, or source notes?
- What is the heading hierarchy?
- Where should tables, checklists, callouts, or forms appear?
- How dense should each page be?
- How should page breaks behave around tables, figures, and headings?

Professional documents should feel natural, polished, and appropriate to their
purpose. Formal documents should earn polish through typography, spacing,
hierarchy, and consistency instead of decorative excess.

## Density and readability

- Avoid long walls of plain text unless the genre demands it.
- Use headings and short paragraphs to make the document skimmable.
- Use bullets for scan-friendly grouped points.
- Use numbered lists for order-dependent steps.
- Use callouts sparingly for decisions, warnings, or important context.
- Prefer clarity over ornament.
- Do not cram too much into narrow areas.
- When space is tight, adjust layout, split content, or shorten wording before
  shrinking text too far.

## Typography and color

- Use professional, easy-to-read fonts.
- Use a clear type scale for title, subtitle, headings, body, captions, and
  footnotes.
- Use bold, italics, and underlines with restraint.
- Use color intentionally for hierarchy or emphasis.
- In formal documents, keep color restrained.
- Keep heading colors, table fills, callout styling, and accent treatments
  consistent.

## Spacing and page flow

Check rigorously for spacing issues:

- clear spacing between sections
- enough space after headings
- no awkward gaps before tables or figures
- no orphan headings at the bottom of a page
- captions kept near their figures or tables
- no large blank areas caused by an oversized table or image
- balanced side margins
- consistent indentation

If a table or visual causes a large blank gap:

- split it cleanly across pages
- reduce visual size modestly
- simplify labels
- move supporting detail to an appendix
- preserve readability rather than forcing everything onto one page

## Images, figures, and visual components

Use visuals only when they improve comprehension, navigation, or usability.

For images and figures:

- Keep them near the text they support.
- Preserve aspect ratio.
- Use captions when the figure needs explanation or citation.
- Add alt text when supported.
- Avoid low-resolution images.
- Make sure images do not overlap text or break the page layout.

For diagrams:

- Use them only when relationships, process, or structure need to be explained.
- Keep labels readable.
- Avoid decorative diagrams that do not carry meaning.

## Text boxes and callouts

Text boxes and callouts must have breathing room:

- generous internal padding
- intentional alignment
- sufficient line spacing
- clear spacing around the box
- no clipped or crowded text
- restrained emphasis

Use callouts for decisions, risks, warnings, definitions, or key takeaways, not
for every paragraph.

## Editing existing documents

When the user asks to edit an existing document, preserve the original and make
minimal local changes unless they request a rewrite.

- Prefer inline edits over rewriting whole paragraphs.
- Keep the original structure unless there is a strong reason to change it.
- If restructuring is needed, do it surgically.
- Avoid heavy blanket deletions.
- Do not silently change legal, financial, technical, or factual meaning.
- Keep feedback close to the point of change.
- Maintain headings, tables, lists, comments, footnotes, bookmarks,
  cross-references, links, headers, and footers where possible.

The goal is trackable improvement, not a fresh draft, unless the user asks for
a fresh draft.

## Review, redline, and comment behavior

For review tasks:

- Separate factual issues from style issues.
- Identify unclear claims, missing evidence, contradictions, duplicated points,
  weak transitions, and inconsistent terminology.
- Suggest wording that preserves the author's intent.
- Put comments close to the relevant text when comments are requested.
- Use end summaries only for overall themes, not as a replacement for local
  feedback.
- If the user asks for a clean final copy, remove review artifacts only after
  confirming that is the requested deliverable.

For redlines:

- Make changes narrow and purposeful.
- Avoid deleting whole sections unless the user requested major restructuring.
- Preserve enough surrounding context for the user to understand each change.

## Accessibility checks

When accessibility matters, check:

- heading order is logical
- tables have clear header rows
- images have useful alt text where supported
- links have meaningful text
- reading order makes sense
- color is not the only way information is conveyed
- form fields have labels or nearby instructions
- captions identify figures and tables clearly

## Privacy and cleanup

Before final delivery when the document is sensitive:

- Remove unintended comments or hidden review notes.
- Remove personal metadata if requested.
- Check for hidden tracked changes.
- Check headers, footers, footnotes, endnotes, and comments for sensitive text.
- For redaction, remove the underlying text, not only the visible appearance.
- Search for redacted terms afterward when the app supports searching.

## Captions, cross-references, and navigation

For long documents:

- Use a table of contents when it improves navigation.
- Use consistent heading levels so navigation works.
- Caption tables and figures when they are referenced from the text.
- Keep references accurate after edits.
- Avoid claiming page numbers or cross-references are correct unless checked.

## Merging multiple documents

When combining documents:

- Confirm source order.
- Preserve section breaks intentionally.
- Normalize conflicting styles only when needed.
- Keep page numbering consistent.
- Preserve important headers, footers, footnotes, images, tables, and captions.
- Check for duplicate title pages, repeated front matter, or conflicting
  numbering.

## Final QA checklist

Before finishing any substantial document task:

1. The document matches the requested purpose and audience.
2. The title, headings, and section order make sense.
3. The requested content is present and accurate.
4. Tables fit, have readable padding, and do not clip text.
5. Lists use consistent structure and indentation.
6. Images, captions, and callouts are placed correctly.
7. Headers, footers, page numbers, and section breaks are correct.
8. There are no unintended overlaps, missing glyphs, awkward page breaks, or
   large unexplained gaps.
9. Comments, tracked changes, metadata, and redactions are handled according to
   the user's request.
10. The final deliverable is the format the user asked for.

If any check fails, fix the document and check again.
