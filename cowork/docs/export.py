"""A doc as a file to download: Markdown, its page as HTML, Word or PDF.

Word and PDF are made as an artifact's export makes them
(cowork.services.artifact_export): the doc's page is first made self-contained
by the fetch-free inliner, so an image comes from the doc itself or its folder
and a web image is left out, never fetched. Word is written by the Word writer;
a PDF comes from the render service, which only a deployment that configures
one offers. While org export is off, an organization deployment offers
neither. Markdown and HTML need no conversion and are always offered.
"""

from __future__ import annotations

import html

from cowork.common.paths import PinnedDir
from cowork.schemas.artifact_export import InlineLimits, InlineMode
from cowork.services import artifact_export
from cowork.services.artifact_inline import inline_artifact_html
from cowork.services.docx_writer import write_docx

from .markdown import doc_to_markdown
from .model import Doc
from .render import doc_html, export_body_html

MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
# The formats that convert, in a worker thread and one at a time (export_slot).
CONVERTED_FORMATS = ("docx", "pdf")
# The order the doc's Export menu lists them in.
_MENU_ORDER = ("docx", "pdf", "md", "html")


def offered_formats(*, org_mode: bool) -> list[str]:
    """The formats a doc exports to on this deployment, in the menu's order."""
    pdf = artifact_export.export_pdf_settings()
    pdf_offered = pdf.engine == "render" and pdf.render_url is not None
    converted = not artifact_export.org_export_refused(org_mode=org_mode)
    return [fmt for fmt in _MENU_ORDER
            if fmt not in CONVERTED_FORMATS or (converted and (fmt != "pdf" or pdf_offered))]


def _document(doc: Doc) -> bytes:
    """The doc as one HTML document for the inliner, its title first when no tab heading carries it."""
    title = html.escape(doc["title"])
    # A one-tab doc has no tab heading to carry its name; the export starts with it.
    title_html = f"<h1>{title}</h1>" if len(doc["tabs"]) == 1 else ""
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title></head>"
            f"<body>{title_html}{export_body_html(doc)}</body></html>").encode("utf-8")


def export_bytes(doc: Doc, fmt: str, *, folder: PinnedDir | None = None, org_mode: bool = False) -> bytes:
    """``doc`` as ``fmt``. Word and PDF need the doc's pinned ``folder``, which
    an image beside the doc is read from, and run in an anyio worker thread: a
    PDF reaches the event loop from there for the render service."""
    if fmt == "md":
        body = doc_to_markdown(doc)
        if not body.lstrip().startswith("# "):
            body = f"# {doc['title']}\n\n{body}"
        return body.encode("utf-8")
    if fmt == "html":
        return doc_html(doc).encode("utf-8")
    if folder is None:
        raise ValueError(f"a {fmt} export reads the doc's folder")
    if fmt == "pdf":
        budget = artifact_export.PDF_INLINE_BUDGET_BYTES
    else:
        budget = artifact_export.ORG_INLINE_BUDGET_BYTES if org_mode else artifact_export.INLINE_BUDGET_BYTES
    inlined = inline_artifact_html(
        html=_document(doc),
        folder=folder,
        base_parts=(),
        mode=InlineMode.WORD if fmt == "docx" else InlineMode.PDF,
        limits=InlineLimits(max_total_bytes=budget),
    )
    if fmt == "docx":
        return write_docx(document=inlined, title=doc["title"])
    # The one way the server gets a PDF: the render service, as an artifact's export does.
    return artifact_export._render_pdf(source_name="doc.html", html=inlined.html)


def export_filename(doc: Doc, fmt: str) -> str:
    """The doc's title as a file name (Unicode kept; the header encodes it)."""
    keep = "".join(ch if ch.isalnum() or ch in " -_" else " " for ch in doc["title"]).strip()
    return ("-".join(keep.split())[:80] or "doc") + f".{fmt}"
