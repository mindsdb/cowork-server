"""Sharing files that aren't web pages (ENG-4687).

A PDF, Word document, spreadsheet, deck, CSV or text file publishes as a
read-only page, rendered in the conversion process like a Markdown page, with
the file itself beside it for the page's download link. The registry keeps
keying off the original file.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook
from reportlab.pdfgen import canvas

from cowork.services import file_page, publish
from test_publish_markdown import _wire_publish

# The renders through publish start a real conversion process.
pytestmark = pytest.mark.usefixtures("child_runs_this_checkout")


@pytest.fixture(autouse=True)
def _no_cached_pages(monkeypatch):
    monkeypatch.setattr(
        publish, "_RENDERED_PAGES", publish._RenderedPages(max_bytes=8 * 1024 * 1024, max_refused=256)
    )


# ─── Files, built in memory ──────────────────────────────────────────────


def _pdf(pages: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _xlsx() -> bytes:
    book = Workbook()
    sheet = book.active
    sheet.title = "Revenue"
    sheet.append(["Quarter", "Total"])
    sheet.append(["Q1", 120])
    sheet.append(["Q2", "=B2*2"])
    book.create_sheet("Costs").append(["Rent", 30])
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def _docx() -> bytes:
    document = Document()
    document.add_heading("Board update", level=1)
    document.add_paragraph("Revenue grew <fast> this quarter.")
    document.add_paragraph("Hire two engineers", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Region"
    table.rows[0].cells[1].text = "Sales"
    table.rows[1].cells[0].text = "EMEA"
    table.rows[1].cells[1].text = "42"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


_SLIDE_XML = (
    '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>'
    "{paragraphs}</p:spTree></p:cSld></p:sld>"
)


def _pptx(slides: list[list[str]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        for number, lines in enumerate(slides, start=1):
            paragraphs = "".join(f"<a:p><a:r><a:t>{line}</a:t></a:r></a:p>" for line in lines)
            archive.writestr(f"ppt/slides/slide{number}.xml", _SLIDE_XML.format(paragraphs=paragraphs))
    return buffer.getvalue()


# ─── The page body, per kind ─────────────────────────────────────────────


def test_every_page_names_the_file_and_links_to_download_it():
    body = file_page.file_page_body(data=b"a,b\n1,2\n", suffix=".csv", name="Q3 sales: final?.csv")

    assert '<header class="file-bar"><h1>Q3 sales: final?.csv</h1>' in body
    # A button, not a bare link: the name is already in the bar.
    assert '<a class="file-download" href="Q3_sales_final.csv" download="Q3 sales: final?.csv">' in body
    assert "Download</a>" in body


def test_a_csv_is_a_table_with_its_first_row_as_the_header():
    body = file_page.file_page_body(data=b"name,total\nAcme,<b>9</b>\n", suffix=".csv", name="t.csv")

    assert "<th>name</th><th>total</th>" in body
    # Everything from the file is escaped: the page carries no markup of its own.
    assert "<td>&lt;b&gt;9&lt;/b&gt;</td>" in body


def test_a_long_csv_shows_its_start_and_says_the_download_has_the_rest(monkeypatch):
    monkeypatch.setattr(file_page, "MAX_TABLE_ROWS", 3)
    body = file_page.file_page_body(data=b"h\n1\n2\n3\n4\n", suffix=".csv", name="t.csv")

    assert body.count("<tr>") == 3
    assert "Showing the first 3 rows. Download the file for the rest." in body


def test_text_is_shown_as_it_is():
    body = file_page.file_page_body(data="line <1>\nlíne 2".encode(), suffix=".txt", name="n.txt")

    assert "<pre>line &lt;1&gt;\nlíne 2</pre>" in body


def test_a_spreadsheet_shows_each_sheet_with_its_values():
    body = file_page.file_page_body(data=_xlsx(), suffix=".xlsx", name="model.xlsx")

    assert "<h2>Revenue</h2>" in body and "<h2>Costs</h2>" in body
    assert "<td>Quarter</td>" in body and "<td>120</td>" in body
    assert "<td>Rent</td>" in body
    # openpyxl saves no result for a formula: the cell shows the formula, not a blank.
    assert "<td>=B2*2</td>" in body


def test_a_word_document_keeps_its_headings_paragraphs_lists_and_tables():
    body = file_page.file_page_body(data=_docx(), suffix=".docx", name="update.docx")

    # The page's own h1 is the file's name, so the document's headings sit one below.
    assert "<h2>Board update</h2>" in body
    assert "<p>Revenue grew &lt;fast&gt; this quarter.</p>" in body
    assert "<li>Hire two engineers</li>" in body
    assert "<th>Region</th>" in body and "<td>EMEA</td>" in body


def test_a_deck_shows_each_slide_with_its_title_and_text():
    body = file_page.file_page_body(
        data=_pptx([["Q3 review", "Revenue up"], ["Next steps"]]), suffix=".pptx", name="deck.pptx"
    )

    assert "<h2>Slide 1: Q3 review</h2>" in body
    assert "<p>Revenue up</p>" in body
    assert "<h2>Slide 2: Next steps</h2>" in body


def test_a_pdf_shows_in_the_browsers_viewer_with_its_text_below():
    body = file_page.file_page_body(data=_pdf(["First page", "Second page"]), suffix=".pdf", name="report.pdf")

    # The whole width and the rest of the window, fit to the width, no thumbnails.
    assert '<object class="file-pdf" data="report.pdf#view=FitH&amp;navpanes=0" type="application/pdf">' in body
    assert "max-width: none" in body
    assert "<h3>Page 1</h3>" in body and "First page" in body
    assert "Second page" in body


@pytest.mark.parametrize("suffix", [".pdf", ".xlsx", ".docx", ".pptx"])
def test_a_file_that_isnt_its_kind_is_refused(suffix):
    with pytest.raises(file_page.FilePageRefused):
        file_page.file_page_body(data=b"not really", suffix=suffix, name=f"x{suffix}")


def test_a_kind_without_a_page_is_refused():
    with pytest.raises(file_page.FilePageRefused):
        file_page.file_page_body(data=b"PK", suffix=".zip", name="x.zip")


# ─── Publishing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["report.pdf", "update.docx", "model.xlsx", "deck.pptx", "data.csv", "notes.txt"])
def test_files_that_have_a_page_are_publishable_content(tmp_path, name):
    art = tmp_path / "art"
    art.mkdir()
    target = art / name
    target.write_bytes(b"x")

    assert publish.is_publishable_static(target, tmp_path)


def test_a_zip_is_still_not_publishable(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "bundle.zip").write_bytes(b"PK")

    assert not publish.is_publishable_static(art / "bundle.zip", tmp_path)


def test_the_page_renders_in_the_conversion_process_with_the_file_beside_it(tmp_path):
    source = tmp_path / "Board update.docx"
    source.write_bytes(_docx())
    out = tmp_path / "out"
    out.mkdir()

    page = publish._render_to_page(source, out)

    assert page == out / "index.html"
    html = page.read_text()
    assert "<title>Board update</title>" in html
    assert "<h2>Board update</h2>" in html
    assert 'href="Board_update.docx"' in html
    # The bar's look rides in the body, over the shared page template.
    assert '<header class="file-bar">' in html and ".file-download {" in html
    assert (out / "Board_update.docx").read_bytes() == source.read_bytes()


def test_publishing_a_file_uploads_its_page_and_bundles_the_file(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    sheet = art / "model.xlsx"
    sheet.write_bytes(_xlsx())
    captured = _wire_publish(monkeypatch, tmp_path, sheet, "model.xlsx")
    bundled: list[str] = []

    def fake_publish(src, **kw):
        from anton.publisher import _zip_html

        captured["src"] = Path(src)
        captured["src_html"] = Path(src).read_text()
        bundled.extend(zipfile.ZipFile(io.BytesIO(_zip_html(Path(src)))).namelist())
        return {"view_url": "https://4nton.ai/view/u/abc", "report_id": "rid-1", "md5": "m1"}

    monkeypatch.setattr("anton.publisher.publish", fake_publish)

    publish.publish_artifact(sheet, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    assert captured["src"].name == "index.html"
    assert "<h2>Revenue</h2>" in captured["src_html"]
    # The page and the file its link downloads go up together.
    assert sorted(bundled) == ["index.html", "model.xlsx"]
    registry = json.loads((art / ".published.json").read_text())
    assert registry["model.xlsx"]["report_id"] == "rid-1"


def test_a_file_too_large_for_a_page_is_refused_before_any_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "PUBLISH_FILE_MAX_BYTES", 10)
    source = tmp_path / "big.csv"
    source.write_bytes(b"a,b\n" * 10)

    with pytest.raises(publish.FileNotPublishable) as refused:
        publish._render_to_page(source, tmp_path)

    assert refused.value.status_code == 413
    assert refused.value.detail["code"] == "file_not_publishable"


def test_a_file_the_conversion_cant_read_is_refused_and_not_checked_again(tmp_path):
    source = tmp_path / "broken.xlsx"
    source.write_bytes(b"not a workbook")

    with pytest.raises(publish.FileNotPublishable):
        publish._render_to_page(source, tmp_path)
    # The "changed since publish" check doesn't run it again.
    with pytest.raises(publish.FileNotPublishable):
        publish._render_to_page(source, tmp_path, background=True)


def test_both_page_refusals_are_one_kind_for_their_callers():
    assert issubclass(publish.FileNotPublishable, publish.PageNotPublishable)
    assert issubclass(publish.MarkdownNotPublishable, publish.PageNotPublishable)
