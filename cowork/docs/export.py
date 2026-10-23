"""A doc as a file to download: Markdown, its page as HTML, PDF or Word."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

from cowork.services.artifact_export import docx_from_html, pdf_from_html

from .markdown import doc_to_markdown
from .model import Doc
from .render import doc_html, export_body_html

MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
# The converters fetch the http(s) images a doc names, from the server.
FETCHING_FORMATS = ("pdf", "docx")


def export_bytes(doc: Doc, fmt: str) -> bytes:
    if fmt == "md":
        body = doc_to_markdown(doc)
        if not body.lstrip().startswith("# "):
            body = f"# {doc['title']}\n\n{body}"
        return body.encode("utf-8")
    if fmt == "html":
        return doc_html(doc).encode("utf-8")
    # A one-tab doc has no tab heading to carry its name; the export starts with it.
    title_html = f"<h1>{html.escape(doc['title'])}</h1>" if len(doc["tabs"]) == 1 else ""
    body = title_html + export_body_html(doc)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"doc.{fmt}"
        if fmt == "pdf":
            pdf_from_html(body, doc["title"], dest)
        else:
            docx_from_html(body, dest)
        return dest.read_bytes()


def export_filename(doc: Doc, fmt: str) -> str:
    """The doc's title as a file name (Unicode kept; the header encodes it)."""
    keep = "".join(ch if ch.isalnum() or ch in " -_" else " " for ch in doc["title"]).strip()
    return ("-".join(keep.split())[:80] or "doc") + f".{fmt}"
