"""A published page for a file that isn't a web page (ENG-4687).

Publishing a PDF, Word document, spreadsheet, deck, CSV or text file shares a
read-only page: what the file says, as HTML, with a link to download the file
itself. ``publish._render_file_to_html`` writes the page the way it writes a
Markdown one (the same template, the same cache), and this module builds its
body. It runs in the export's conversion process
(``cowork.services.export_child``), which the server kills past its time or
memory limit, so a hostile file can't hold the server.

Each kind is bounded: a long file shows its start and says the download has
the rest. Everything taken from the file is escaped; the page carries no
script. A PDF shows in the browser's own viewer, from the copy published beside the
page, with its text below in case the host won't show it. A file a reader can't open raises ``FilePageRefused``, which the
server answers as not publishable (413), like a Markdown file it can't render.
"""
from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from xml.etree import ElementTree

# The kinds a person can share as a page of their own. HTML and Markdown are
# ``artifacts.PUBLISHABLE_STATIC_SUFFIXES``.
PUBLISHABLE_FILE_SUFFIXES = (".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".txt")

# How much of each kind the page shows.
MAX_PDF_PAGES = 200
MAX_TABLE_ROWS = 1000
MAX_TABLE_COLUMNS = 50
MAX_SHEETS = 20
MAX_SLIDES = 200
MAX_TEXT_CHARS = 2 * 1024 * 1024
MAX_CELL_CHARS = 2000


# The page's own look, over the published page template's: a bar that stays at
# the top with the file's name and a Download button, then the file. Its
# colours are the template's variables.
_PAGE_STYLE = """<style>
  .doc { max-width: 1100px; padding: 0 24px 48px; }
  .file-bar { position: sticky; top: 0; z-index: 1; display: flex; align-items: center;
              justify-content: space-between; gap: 16px; padding: 12px 0; margin: 0 0 16px;
              background: var(--bg); border-bottom: 1px solid var(--border); }
  .file-bar h1 { margin: 0; padding: 0; border: 0; font-size: 1.05em; font-weight: 600;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-download { flex-shrink: 0; display: inline-flex; align-items: center; gap: 8px;
                   padding: 8px 16px; border-radius: 8px; background: var(--accent); color: #0d1117;
                   font-weight: 600; font-size: 14px; line-height: 1; }
  .file-download:hover { text-decoration: none; filter: brightness(1.1); }
  .file-download svg { width: 16px; height: 16px; }
</style>"""
# A PDF takes the whole width and the rest of the window's height.
_PDF_STYLE = """<style>
  .doc { max-width: none; padding: 0 16px 16px; }
  .file-pdf { display: block; width: 100%; height: calc(100vh - 74px); border: 0; border-radius: 8px; }
</style>"""
_DOWNLOAD_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>'
)


class FilePageRefused(Exception):
    """The file can't be read as its kind: it fails the same way every time."""


def published_file_name(name: str) -> str:
    """The name the file is published under beside its page, for the download link.

    The publisher bundles what the page links to by a relative name without
    ``:``, ``#`` or ``?``, so the link holds only safe characters. The link's
    ``download`` attribute keeps the file's own name for whoever saves it.
    """
    stem, dot, suffix = name.rpartition(".")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem if dot else name).strip("._") or "file"
    return f"{safe[:120]}.{suffix.lower()}" if dot else safe[:120]


def file_page_body(*, data: bytes, suffix: str, name: str) -> str:
    """The body of the page that shows *data*, a file of *suffix* named *name*."""
    render = _RENDERERS.get(suffix)
    if render is None:
        raise FilePageRefused(f"{suffix} files aren't published as a page")
    link = published_file_name(name)
    bar = (
        f'<header class="file-bar"><h1>{html.escape(name)}</h1>'
        f'<a class="file-download" href="{html.escape(link)}" download="{html.escape(name)}">'
        f"{_DOWNLOAD_ICON}Download</a></header>"
    )
    style = _PAGE_STYLE + (_PDF_STYLE if suffix == ".pdf" else "")
    body = render(data, link) if suffix == ".pdf" else render(data)
    return f"{style}\n{bar}\n{body}"


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS] + "…"
    return html.escape(text)


def _more(shown: str) -> str:
    """Says the page shows only part of the file: *shown* is "the first 1000 rows"."""
    return f'<p class="file-note"><em>Showing {shown}. Download the file for the rest.</em></p>'


def _table(rows: list[list[object]], *, header: bool) -> str:
    if not rows:
        return '<p class="file-note"><em>Empty.</em></p>'
    out = ['<div style="overflow-x:auto"><table>']
    for index, row in enumerate(rows):
        cell = "th" if header and index == 0 else "td"
        out.append("<tr>" + "".join(f"<{cell}>{_esc(value)}</{cell}>" for value in row) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def _pdf(data: bytes, link: str) -> str:
    """The PDF itself, in the browser's own viewer, and its text below as the fallback.

    The page embeds the copy published beside it (the download link's file).
    A host that won't show a PDF in a page still leaves its text readable.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        total = len(reader.pages)
        texts = []
        for index in range(min(total, MAX_PDF_PAGES)):
            texts.append((index + 1, (reader.pages[index].extract_text() or "").strip()))
    except (PyPdfError, ValueError, KeyError, TypeError) as exc:
        raise FilePageRefused("not a PDF that can be read") from exc
    # Opens fit to the width, without the page thumbnails, where the browser's viewer takes the hint.
    viewer = (
        f'<object class="file-pdf" data="{html.escape(link)}#view=FitH&amp;navpanes=0" type="application/pdf">'
        "<p><em>This browser can't show the PDF here. Download it, or read its text below.</em></p>"
        "</object>"
    )
    pages = "\n".join(
        f"<h3>Page {number}</h3>\n<pre>{html.escape(text[:MAX_TEXT_CHARS], quote=False)}</pre>"
        for number, text in texts
        if text
    )
    if total > MAX_PDF_PAGES:
        pages += _more(f"the text of the first {MAX_PDF_PAGES} of {total} pages")
    text_part = f"<details><summary>Text of the PDF</summary>\n{pages}\n</details>" if pages else ""
    return f"{viewer}\n{text_part}"


def _csv(data: bytes) -> str:
    text = data[: MAX_TEXT_CHARS * 4].decode("utf-8-sig", errors="replace")
    rows: list[list[object]] = []
    truncated = False
    try:
        for row in csv.reader(io.StringIO(text)):
            if len(rows) >= MAX_TABLE_ROWS:
                truncated = True
                break
            rows.append(row[:MAX_TABLE_COLUMNS])
    except csv.Error as exc:
        raise FilePageRefused("not a CSV that can be read") from exc
    body = _table(rows, header=True)
    return body + (_more(f"the first {MAX_TABLE_ROWS} rows") if truncated else "")


def _txt(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    body = f"<pre>{html.escape(text[:MAX_TEXT_CHARS], quote=False)}</pre>"
    return body + (_more("the start of the text") if len(text) > MAX_TEXT_CHARS else "")


def _xlsx(data: bytes) -> str:
    """Each sheet as a table of its values.

    A cell shows the value its file saved. A workbook written by code (as
    Anton writes them, with openpyxl) saves no results for its formulas, so a
    formula cell with no saved value shows its formula instead of a blank.
    """
    from openpyxl import load_workbook

    try:
        values = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        formulas = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
    except Exception as exc:  # openpyxl raises zip, XML and key errors for a broken file
        raise FilePageRefused("not a spreadsheet that can be read") from exc
    try:
        parts = []
        for count, (sheet, formula_sheet) in enumerate(zip(values.worksheets, formulas.worksheets)):
            if count >= MAX_SHEETS:
                parts.append(_more(f"the first {MAX_SHEETS} sheets"))
                break
            rows: list[list[object]] = []
            truncated = False
            pairs = zip(sheet.iter_rows(values_only=True), formula_sheet.iter_rows(values_only=True))
            for row, formula_row in pairs:
                if len(rows) >= MAX_TABLE_ROWS:
                    truncated = True
                    break
                rows.append([
                    formula if value is None and isinstance(formula, str) and formula.startswith("=") else value
                    for value, formula in zip(row[:MAX_TABLE_COLUMNS], formula_row[:MAX_TABLE_COLUMNS])
                ])
            # A sheet's trailing empty rows say nothing.
            while rows and all(value is None for value in rows[-1]):
                rows.pop()
            parts.append(f"<h2>{_esc(sheet.title)}</h2>")
            parts.append(_table(rows, header=False))
            if truncated:
                parts.append(_more(f"the first {MAX_TABLE_ROWS} rows of this sheet"))
    finally:
        values.close()
        formulas.close()
    return "\n".join(parts)


def _docx(data: bytes) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises zip, XML and key errors for a broken file
        raise FilePageRefused("not a Word document that can be read") from exc
    parts = []
    for block in document.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(block, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            level = re.match(r"Heading (\d)", style)
            if style == "Title":
                parts.append(f"<h1>{_esc(text)}</h1>")
            elif level:
                # The page's own h1 is the file's name, so the document's headings sit one below.
                depth = min(int(level.group(1)) + 1, 6)
                parts.append(f"<h{depth}>{_esc(text)}</h{depth}>")
            elif style.startswith("List"):
                parts.append(f"<ul><li>{_esc(text)}</li></ul>")
            else:
                parts.append(f"<p>{_esc(text)}</p>")
        elif tag == "tbl":
            table = Table(block, document)
            rows = [[cell.text for cell in row.cells[:MAX_TABLE_COLUMNS]] for row in table.rows[:MAX_TABLE_ROWS]]
            parts.append(_table(rows, header=True))
    return "\n".join(parts) or '<p class="file-note"><em>This document has no text.</em></p>'


_SLIDE = re.compile(r"ppt/slides/slide(\d+)\.xml$")
_DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _pptx(data: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        slides = sorted(
            ((int(match.group(1)), name) for name in archive.namelist() if (match := _SLIDE.match(name))),
        )
    except zipfile.BadZipFile as exc:
        raise FilePageRefused("not a deck that can be read") from exc
    if not slides:
        raise FilePageRefused("not a deck that can be read")
    parts = []
    for count, (number, name) in enumerate(slides):
        if count >= MAX_SLIDES:
            parts.append(_more(f"the first {MAX_SLIDES} slides"))
            break
        try:
            root = ElementTree.fromstring(archive.read(name))
        except (ElementTree.ParseError, KeyError) as exc:
            raise FilePageRefused("not a deck that can be read") from exc
        lines = []
        for paragraph in root.iter(f"{_DRAWING}p"):
            text = "".join(run.text or "" for run in paragraph.iter(f"{_DRAWING}t")).strip()
            if text:
                lines.append(text)
        title, rest = (lines[0], lines[1:]) if lines else ("", [])
        heading = f"Slide {number}" + (f": {_esc(title)}" if title else "")
        parts.append(f"<h2>{heading}</h2>")
        parts.extend(f"<p>{_esc(line)}</p>" for line in rest)
    return "\n".join(parts)


_RENDERERS = {
    ".pdf": _pdf,
    ".csv": _csv,
    ".txt": _txt,
    ".xlsx": _xlsx,
    ".docx": _docx,
    ".pptx": _pptx,
}
