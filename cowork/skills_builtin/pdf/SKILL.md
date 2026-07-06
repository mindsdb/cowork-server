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
- OCR: process scanned PDFs and make them searchable when supported.

## Common tool patterns

Use whatever PDF tools the app provides. When code execution is available,
these are reliable patterns:

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

### Create a PDF

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
- Convert pages to images if OCR is needed and supported.
- OCR each page separately so page numbers remain traceable.
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

## Command-line equivalents when available

Useful operations:

```bash
# Extract text
pdftotext input.pdf output.txt
pdftotext -layout input.pdf output.txt
pdftotext -f 1 -l 5 input.pdf output.txt

# Merge
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf

# Split ranges
qpdf input.pdf --pages . 1-5 -- pages1-5.pdf
qpdf input.pdf --pages . 6-10 -- pages6-10.pdf

# Rotate page 1
qpdf input.pdf output.pdf --rotate=+90:1

# Decrypt with a user-provided password
qpdf --password=mypassword --decrypt encrypted.pdf decrypted.pdf

# Extract images
pdfimages -j input.pdf output_prefix
```

Use command-line tools only when available in the app environment.

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
