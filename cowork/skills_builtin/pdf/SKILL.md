---
name: pdf
description: >
  Use this skill whenever the user wants to do anything with PDF files. This
  includes reading or extracting text or tables from PDFs, combining or merging
  PDFs, splitting PDFs apart, rotating pages, adding watermarks, creating new
  PDFs, filling PDF forms, encrypting or decrypting PDFs, extracting images,
  OCR on scanned PDFs, making PDFs searchable, or producing a final .pdf file.
---

# PDF Processing Guide

Use this skill for PDF reading, extraction, transformation, creation, forms,
security, scanned documents, and final verification.

## Exact reading workflow

PDF text extraction can lie. Do not rely on a single method when exactness
matters.

For careful reads:

1. Inspect rendered pages or previews to understand visible structure.
2. Extract text page by page.
3. Extract tables separately from body text.
4. Compare extraction against the rendered page for important values.
5. Preserve page numbers in summaries and citations when useful.

Be especially careful with:

- multi-column layouts
- tables
- footnotes
- headers and footers
- scanned pages
- rotated pages
- forms
- text layered behind watermarks
- PDFs generated from slides or images

## Task routing

Classify the job first:

- READ: summarize, extract text, answer questions, inspect metadata.
- TABLES: extract table rows, preserve headers, export structured data.
- MERGE: combine multiple PDFs into one file.
- SPLIT: split by page, range, section, bookmark, or one file per page.
- ROTATE: rotate selected pages or normalize page orientation.
- WATERMARK: add, audit, or remove visible stamps/watermarks.
- CREATE: generate a new PDF deliverable.
- FORM: inspect, fill, flatten, or verify PDF forms.
- SECURITY: encrypt, decrypt with a user-provided password, or preserve permissions.
- IMAGES: extract embedded images or convert pages to images.
- OCR: read scanned pages with the model (see Scanned PDFs and OCR).

## Common tool patterns

Use these Python libraries in the scratchpad, not command-line tools such as
`pdftotext`, `qpdf` or `pdfimages`: Cowork on the web has none of them, and a
person's computer often doesn't either. `pypdf`, `pdfplumber` and `pypdfium2`
are installed on Cowork on the web. Put every library a cell imports in that
cell's `packages` array: one already installed is skipped, and a missing one is
installed before the cell runs.

### Read page count and text

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
print(f"Pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages, start=1):
    print(f"--- Page {i} ---")
    print(page.extract_text() or "")
```

### Merge PDFs

```python
from pypdf import PdfReader, PdfWriter

writer = PdfWriter()
for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        writer.add_page(page)

with open("merged.pdf", "wb") as output:
    writer.write(output)
```

### Split PDF into one file per page

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    writer = PdfWriter()
    writer.add_page(page)
    with open(f"page_{i + 1}.pdf", "wb") as output:
        writer.write(output)
```

### Rotate pages

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader("input.pdf")
writer = PdfWriter()

for index, page in enumerate(reader.pages, start=1):
    if index == 1:
        page.rotate(90)
    writer.add_page(page)

with open("rotated.pdf", "wb") as output:
    writer.write(output)
```

### Extract metadata

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
meta = reader.metadata
print(meta)
```

### Extract text with layout

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        print(f"--- Page {i} ---")
        print(page.extract_text(layout=True) or "")
```

### Extract tables

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        for j, table in enumerate(tables, start=1):
            print(f"Table {j} on page {i}")
            for row in table:
                print(row)
```

### Render a page to an image

Use this to look at a page, check a layout, or read a scanned page.

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("document.pdf")
image = pdf[0].render(scale=2).to_pil()  # first page, as a PIL image
image.save("page_1.png")
```

### Create a PDF

Cowork on the web doesn't have `reportlab` installed: put `reportlab` in the
cell's `packages` array (it is skipped where it is already installed).

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("output.pdf", pagesize=letter)
width, height = letter
c.drawString(72, height - 72, "Report Title")
c.drawString(72, height - 96, "This PDF was generated from source content.")
c.save()
```

### Add a watermark

```python
from pypdf import PdfReader, PdfWriter

watermark = PdfReader("watermark.pdf").pages[0]
reader = PdfReader("document.pdf")
writer = PdfWriter()

for page in reader.pages:
    page.merge_page(watermark)
    writer.add_page(page)

with open("watermarked.pdf", "wb") as output:
    writer.write(output)
```

### Encrypt a PDF

```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader("input.pdf")
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

writer.encrypt("userpassword", "ownerpassword")

with open("encrypted.pdf", "wb") as output:
    writer.write(output)
```

## Tables

When extracting tables:

- identify the page and table boundaries
- preserve column headers
- preserve blank cells when meaningful
- avoid splitting one multi-line cell into separate records
- check totals, dates, currencies, and signs
- compare extracted rows against the rendered page

If extraction is messy, use the rendered page as the source of truth and rebuild
the table carefully.

## Scanned PDFs and OCR

For scanned PDFs:

- Treat the PDF as image-based until text extraction proves otherwise.
- There is no OCR engine such as tesseract. Read scanned pages with the model.
  If `reading-files` is listed in your skills, recall it: it renders each page
  with `pypdfium2`, sends it to `get_llm()` as an image, saves the text page by
  page, and asks before reading more than 20 pages. Otherwise render each page
  as above (list `pypdfium2` in `packages`) and send it to `get_llm()`.
- Read each page separately so page numbers remain traceable.
- Mention uncertainty when the scan is low-resolution, skewed, handwritten, or noisy.
- After OCR, search for important terms and spot-check against the page image.

## Forms

For PDF forms:

- Inspect available fields before filling.
- Match values to visible labels.
- Preserve checkboxes, radio buttons, dates, signature areas, and multiline fields.
- Verify both stored values and visible rendered output.
- Flatten only if the user wants a final non-editable version.

## Subscripts and superscripts in generated PDFs

When generating PDFs with common built-in fonts, avoid Unicode subscript and
superscript characters because they may render as boxes. Prefer markup or
manual baseline adjustment when the PDF tool supports it.

Examples:

- use `H<sub>2</sub>O` instead of Unicode subscript 2 in rich paragraph tools
- use `x<super>2</super>` instead of Unicode superscript 2 in rich paragraph tools

## Editing and coordinates

PDF coordinates are easy to get wrong.

For coordinate-based edits:

- render or preview the target page first
- identify the page coordinate system
- place content conservatively
- verify the edited page visually
- adjust coordinates and verify again

Never assume added text, stamps, or overlays fit correctly without checking the
rendered output.

## Redaction

True redaction must remove the underlying text or image content. Drawing a
black rectangle over text is not enough.

After redaction:

- search for the removed text
- check copied text output
- inspect the rendered page
- preserve a non-redacted source copy unless the user asked otherwise

## Final verification

Before finishing:

1. Confirm the correct input PDFs were used.
2. Confirm page count and page order.
3. Confirm page ranges for split, extract, rotate, or merge tasks.
4. Inspect important output pages visually.
5. Verify extracted text, tables, form values, images, or watermarks as relevant.
6. For OCR or scanned documents, mention quality limitations.
7. For password-protected files, confirm the user provided authorization.
8. Deliver the requested final PDF or extracted artifact only.
