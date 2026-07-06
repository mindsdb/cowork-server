---
name: docx
description: >
  Use this skill whenever the user wants to create, read, edit, or manipulate
  Word documents (.docx files). Triggers include "Word doc", "word document",
  ".docx", reports, memos, letters, templates, headings, page numbers, tables
  of contents, letterheads, images, comments, tracked changes, find-and-replace,
  or converting content into a polished Word document.
---

# DOCX Creation, Editing, and Analysis

A `.docx` file is a ZIP archive containing XML files. Treat Word documents as
structured files, not just plain text.

## Quick reference

| Task | Approach |
| --- | --- |
| Read or analyze content | Extract text and inspect structure |
| Create a new document | Generate structured DOCX content, then validate |
| Edit an existing document | Preserve existing structure and edit targeted parts |
| Handle tracked changes | Read, preserve, accept, or represent changes intentionally |
| Inspect layout | Convert or preview pages visually when possible |

## Reading content

When reading a `.docx`:

- Preserve heading order and section hierarchy.
- Treat tables, images, captions, footnotes, comments, and tracked changes as
  separate content types.
- For review tasks, distinguish body text from comments and suggested changes.
- If exact wording matters, avoid over-summarizing.
- If the document has tracked changes, include inserted and deleted text only
  according to the user's request.

## Creating new documents

When creating a new `.docx`, decide the document type before drafting:

- memo
- report
- proposal
- letter
- template
- handbook
- guide
- checklist
- form

Use a consistent style system:

- explicit page size and margins
- clear title and heading hierarchy
- readable body text
- real list structure
- consistent table geometry
- page numbers for multi-page documents
- tables of contents only when useful

## Page size and layout

Set page size explicitly when the tool allows it.

Common page sizes:

| Paper | Width DXA | Height DXA | Content width with 1 inch margins |
| --- | ---: | ---: | ---: |
| US Letter | 12240 | 15840 | 9360 |
| A4 | 11906 | 16838 | 9026 |

Rules:

- Use 1 inch margins unless the user or template says otherwise.
- For landscape pages, verify that width, height, and orientation are all correct.
- Check mixed portrait and landscape sections carefully.
- Avoid large blank gaps caused by badly placed page breaks.

## Styles and headings

Use real heading styles rather than manually bolded paragraphs.

For reliable navigation and tables of contents:

- Heading 1 should represent top-level sections.
- Heading 2 and Heading 3 should follow the document hierarchy.
- Headings should include outline levels when the tool supports them.
- Do not fake headings with direct formatting only.
- Keep title, subtitle, headings, body, captions, and footnotes visually distinct.

Use common fonts when compatibility matters. Arial, Calibri, Aptos, and Times
New Roman are safer than obscure fonts unless the user provides a brand font.

## Lists

Use real list structure. Do not fake lists by typing bullet characters, hyphens,
or manual numbers into ordinary paragraphs.

Rules:

- Use real bullets for unordered lists.
- Use real numbering for ordered steps.
- Keep wrapped lines aligned under the list text.
- Keep nested lists shallow.
- Restart numbering only when intended.
- Do not place multiple list items inside one paragraph separated by newlines.

## Tables

Tables need explicit geometry to render consistently.

Rules:

- Use tables only for true row and column data.
- Set table width deliberately.
- Set column widths deliberately.
- Make cell widths match the corresponding columns when possible.
- Use internal cell margins for readable padding.
- Avoid percentage widths when compatibility is uncertain.
- Avoid fixed row heights that can clip text.
- Avoid using tables as decorative dividers.
- Keep table width within the page content width.

Width rule of thumb:

- US Letter with 1 inch margins has 9360 DXA content width.
- Full-width tables should not exceed the content width.
- Column widths should add up to the table width.

For long tables:

- repeat header rows when possible
- keep column order stable
- split cleanly across pages
- keep captions close to the table

## Images

When inserting images:

- specify the image type when the tool requires it
- preserve aspect ratio
- set dimensions deliberately
- include alt text when supported
- keep images near the text they support
- verify placement after insertion

Do not let images overlap text, cover headings, or push content into awkward
page breaks.

## Page breaks and sections

Use real page breaks and section breaks.

Rules:

- Use a real page break for a new page.
- Use section breaks for orientation, columns, or header/footer changes.
- Do not simulate page breaks with many blank paragraphs.
- Check that headings do not land alone at the bottom of a page.

## Hyperlinks, bookmarks, and references

For links:

- Use external hyperlinks for URLs.
- Use bookmarks and internal links for document navigation.
- Check that link text is meaningful.
- Avoid raw pasted URLs when cleaner link text is better.

For cross-references:

- Keep captions and referenced objects in sync.
- Re-check after editing or moving sections.

## Footnotes and endnotes

Use real footnotes or endnotes when the document needs citations, source notes,
or explanatory notes.

Rules:

- Keep notes concise.
- Do not mix citation styles without reason.
- Verify note numbering after edits.

## Headers and footers

Use headers and footers for repeated document furniture:

- page numbers
- document title
- date
- confidentiality notice
- company name
- section title

Check first-page, odd/even, and section-specific headers separately when they
exist.

## Tables of contents

Only add a table of contents when the document is long enough to benefit from
navigation.

Rules:

- Use real heading levels.
- Include only the needed heading depth.
- Update or verify the table of contents after edits.
- Do not manually type a fake TOC unless the output format cannot support a
  live one.

## Editing existing documents

Follow this order:

1. Inspect the document structure.
2. Identify the exact parts that need changing.
3. Preserve existing styles, numbering, tables, images, links, comments,
   tracked changes, footnotes, headers, and footers unless the user asks to
   change them.
4. Make targeted edits.
5. Validate the output and inspect layout.

Prefer small replacements over rewriting whole sections. If a major rewrite is
needed, explain the restructure and preserve meaning.

## Tracked changes and comments

For tracked changes:

- Preserve them when the user asks for review mode.
- Accept or reject them only when the user requests a clean document.
- Make clear whether the final file includes visible markup.

For comments:

- Keep comments anchored near the relevant text.
- Do not move all feedback to the end unless the user asks.
- Remove comments only when producing a final clean version.

## Google Docs compatibility

When the `.docx` is intended for Google Docs:

- Use simple, compatible typography.
- Avoid complex Word-only layout features.
- Avoid decorative title borders and title rules unless specifically requested.
- Use tables conservatively.
- Use standard headings and lists.
- Verify the imported or converted result if possible.

## Validation and visual QA

After creating or editing:

- confirm the file opens
- confirm page count if layout matters
- inspect rendered pages or previews when available
- verify no clipped text, broken tables, missing glyphs, or overlap
- check headers and footers
- check tables, images, lists, links, comments, and tracked changes

If visual rendering is unavailable, do a structural inspection and disclose
that visual QA was not available.

## Final checks

Before finishing:

1. Confirm the final output is a `.docx` when requested.
2. Confirm requested edits or generated sections are present.
3. Verify headings, lists, tables, images, links, page breaks, page numbers,
   comments, tracked changes, and footnotes as relevant.
4. Confirm the document preserves the user's intended structure and meaning.
5. Mention any compatibility, formatting, or preservation caveats.
