"""Convert a document artifact (markdown or HTML) to PDF / Word / HTML.

Pure-Python pipeline — no native deps, no external binaries — so the
desktop install stays clean across platforms:

    markdown → HTML   (markdown)
    HTML     → PDF    (xhtml2pdf / reportlab)
    HTML     → DOCX   (htmldocx / python-docx)

The output is written next to the source (same artifact folder) as
``<stem>.<ext>`` and the path is returned so the caller can open or
download it. Conversion fidelity is good for ordinary reports/memos; it
is not publication-grade typesetting (that would need pandoc + LaTeX,
which we deliberately don't ship).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Target formats we can produce, and the source extensions we accept.
SUPPORTED_FORMATS = ("pdf", "docx", "html")
_SOURCE_EXTS = (".md", ".markdown", ".html", ".htm", ".txt")


class ExportError(Exception):
    """Raised for an unsupported source/target or a conversion failure."""


def _read_source_html(source: Path) -> str:
    """Return the source document as an HTML string.

    Markdown is rendered to HTML; HTML is used as-is; plain text is wrapped
    in a <pre>. Anything else is rejected by the caller via extension check.
    """
    text = source.read_text(encoding="utf-8", errors="replace")
    ext = source.suffix.lower()
    if ext in (".html", ".htm"):
        return text
    if ext == ".txt":
        import html as _html

        return f"<pre>{_html.escape(text)}</pre>"
    # markdown
    import markdown as _markdown

    body = _markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    return body


def _wrap_html_document(body_html: str, title: str) -> str:
    """Wrap rendered body HTML in a minimal, print-friendly document."""
    import html as _html

    safe_title = _html.escape(title or "Document")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title>"
        "<style>"
        "body{font-family:'Helvetica','Arial',sans-serif;font-size:11pt;"
        "line-height:1.5;color:#1a1a1a;margin:2.5em;}"
        "h1,h2,h3{line-height:1.25;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}"
        "code,pre{font-family:'Courier New',monospace;background:#f5f5f5;}"
        "pre{padding:10px;overflow-x:auto;white-space:pre-wrap;}"
        "</style></head><body>"
        f"{body_html}"
        "</body></html>"
    )


def _to_pdf(html_doc: str, dest: Path) -> None:
    from xhtml2pdf import pisa

    with dest.open("wb") as fh:
        result = pisa.CreatePDF(src=html_doc, dest=fh)
    if result.err:
        raise ExportError("PDF conversion failed")


def _to_docx(body_html: str, dest: Path) -> None:
    from docx import Document
    from htmldocx import HtmlToDocx

    document = Document()
    HtmlToDocx().add_html_to_document(body_html, document)
    document.save(str(dest))


def export_artifact(source_path: Path, fmt: str) -> Path:
    """Convert `source_path` to `fmt`, writing the result beside the source.

    Returns the output Path. Raises ExportError on an unsupported
    source/target or a conversion failure.
    """
    fmt = (fmt or "").lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(f"Unsupported export format: {fmt!r}")
    if source_path.suffix.lower() not in _SOURCE_EXTS:
        raise ExportError(
            f"Can only export markdown or HTML documents, not {source_path.suffix!r}"
        )
    if not source_path.is_file():
        raise ExportError("Source document not found")

    dest = source_path.with_suffix(f".{fmt}")
    title = source_path.stem

    try:
        if fmt == "html":
            dest.write_text(
                _wrap_html_document(_read_source_html(source_path), title),
                encoding="utf-8",
            )
        elif fmt == "pdf":
            _to_pdf(_wrap_html_document(_read_source_html(source_path), title), dest)
        elif fmt == "docx":
            _to_docx(_read_source_html(source_path), dest)
    except ExportError:
        raise
    except Exception as exc:  # converter blew up — surface a clean message
        logger.warning("Artifact export to %s failed for %s", fmt, source_path, exc_info=True)
        raise ExportError(f"Could not convert document to {fmt.upper()}") from exc

    return dest
